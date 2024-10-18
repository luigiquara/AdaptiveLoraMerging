'''
Perform LoRA merging on class-incremental scenario.
'''

import argparse
from collections import defaultdict
import copy
import os

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, confusion_matrix_metrics, forgetting_metrics, loss_metrics, StreamConfusionMatrix, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from peft import get_peft_model, LoraConfig
from tabulate import tabulate
import timm
from torch import nn
from torch.optim import Adam
from torchvision import transforms

def load_benchmark(dataset_name, n_classes_per_exp, dataset_root, seed):
    # read the benchmark
    if dataset_name.lower() == 'splitcifar100':
        print('Loading SplitCIFAR100')

        # normalize values from avalanche docs
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        ])

        benchmark = SplitCIFAR100(
            n_experiences = int(100/n_classes_per_exp),
            seed = seed, dataset_root = dataset_root,
            train_transform = transform, eval_transform = transform
        )

    # the dataset name is wrong or it is not yet implemented
    else: raise NotImplementedError

    return benchmark

def single_seed_train(args, seed):
    benchmark = load_benchmark(args.dataset_name, args.n_classes_per_exp, args.dataset_root, seed)
    breakpoint()

    # load the model
    base_model = timm.create_model(args.base_model, pretrained=True, num_classes=benchmark.n_classes)

    # create peft model
    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = ['qkv', 'fc1', 'fc2']
    )
    peft_model = get_peft_model(base_model, lora_config)

    test_results = []
    for experience in benchmark.train_stream:
        print(f'Start of task {experience.task_label}')
        print(f'Classes in this task: {experience.classes_in_this_experience}')

        #lora_name = '_'.join(experience.classes_in_this_experience)
        lora_name = '_'.join([str(c) for c in experience.classes_in_this_experience])
        lora_path = os.path.join(args.lora_root, str(seed), lora_name)

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            #cpu_usage_metrics(experience=True),
            forgetting_metrics(experience=True, stream=True),
            StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
            #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[InteractiveLogger()]
        )

        if os.path.exists(lora_path):
            print(f'Load lora adapter from {lora_path}')
            peft_model.load_adapter(lora_path, lora_name)

        else:
            print(f'Create new lora adapter for {lora_name}')
            peft_model.add_adapter(lora_name, lora_config)
            peft_model.set_adapter(lora_name)

            # train adapter
            training_strategy = Naive(
                peft_model, Adam(peft_model.parameters(), lr=args.lr), nn.CrossEntropyLoss(),
                train_mb_size = args.batch_size, train_epochs = args.epochs, eval_mb_size = args.batch_size,
                evaluator = eval_plugin, device = args.device
            )
            res = training_strategy.train(experience)
            print(f'Completed training for {lora_name}')

            # save the adapter
            # remove lora_name from the path, since save_pretrained creates that dir
            lora_path = os.path.join(args.lora_root, str(seed))
            peft_model.save_pretrained(lora_path)
            print(f'{lora_name} saved to {lora_path}')

        print('Evaluating on the whole test set')
        # need to define a strategy only for evaluation
        # maybe there is a smarter way
        training_strategy = Naive(
            peft_model, Adam(peft_model.parameters(), lr=args.lr), nn.CrossEntropyLoss(),
            train_mb_size = args.batch_size, train_epochs = args.epochs, eval_mb_size = args.batch_size,
            evaluator = eval_plugin, device = args.device
        )
        test_results.append(training_strategy.eval(benchmark.test_stream))

def single_run_class_incremental_testing(args, seed):
    '''
    'Diagonal' evaluation.
    For experience at timestep t, merge all the adaptars from 0 to t and test the merged model on all experiences up to t.

    We assume to have all the adapters already trained and stored on disk
    '''

    benchmark = load_benchmark(args.dataset_name, args.n_classes_per_exp, args.dataset_root, seed)

    # load the model
    base_model = timm.create_model(args.base_model, pretrained=True, num_classes=benchmark.n_classes)

    # create peft model
    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = ['qkv', 'fc1', 'fc2']
    )
    peft_model = get_peft_model(base_model, lora_config)

    res = defaultdict(list)
    lora_names = []
    combination_types = ['linear', 'ties', 'dare_ties']

    eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            #cpu_usage_metrics(experience=True),
            forgetting_metrics(experience=True, stream=True),
            #StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
            #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[InteractiveLogger()]
        )

    for i, experience in enumerate(benchmark.test_stream):
        print(f'Testing on classes {experience.classes_in_this_experience}\n')

        lora_name = '_'.join([str(c) for c in experience.classes_in_this_experience])
        lora_path = os.path.join(args.lora_root, str(seed), lora_name)

        # load the adapter related to the current classes
        if os.path.exists(lora_path):
            peft_model.load_adapter(lora_path, lora_name)
            print(f'Loaded adapter from {lora_path}')
        else:
            raise FileNotFoundError(f'{lora_path} do not exists!')

        lora_names.append(lora_name)
        print(f"I'm going to merge these adapters: {lora_names}")
        density = 0.2
        weights = [1.0]*len(lora_names)

        for combination_type in combination_types:
            print(f'\nUsing {combination_type} merging alg')

            merged_model = copy.deepcopy(peft_model)
            merged_model.add_weighted_adapter(lora_names, weights, f'merge_{combination_type}', combination_type, density=density)
            merged_model.set_adapter(f'merge_{combination_type}')

            training_strategy = Naive(
                merged_model, Adam(merged_model.parameters(), lr=args.lr), nn.CrossEntropyLoss(),
                train_mb_size = args.batch_size, train_epochs = args.epochs, eval_mb_size = args.batch_size,
                evaluator = eval_plugin, device = args.device
            )
            # test on current and previous experiences
            res[combination_type].append(training_strategy.eval(benchmark.test_stream[:i+1]))

    # save the results in a dict
    # to later create a md table
    merged_alg_res = defaultdict(list)
    for combination_type, alg_res in res.items():
        key = 'Top1_Acc_Exp/eval_phase/test_stream/Task000'
        
        # alg_res_at_t are the results at timestep t
        # the model used is the one with merged adapters up to t
        # the model is evaluated up to experience t
        for t, alg_res_at_t in enumerate(alg_res):
            merged_alg_res[combination_type].append({
                'experience': f'E{t}',
                'W': weights[0] if weights[0] == weights[1] else [f'w:.2f' for w in weights],
                'accs': []
            })

            for idx in range(t+1):
                exp_num = f'0{idx}'
                if len(exp_num) == 2: exp_num = f'0{exp_num}'
                merged_alg_res[combination_type][-1]['accs'].append(alg_res_at_t[f'{key}/Exp{exp_num}'])
    
    return merged_alg_res

def get_result_table(res_dict):
    for combination_type, results in res_dict.items():
        print(f"----- {combination_type.upper()} -----")
        headers = ['Exp', 'W'] + [f'E{t}' for t in range(len(results))]
        
        table = []
        for entry in results:
            row = [entry['experience'], str(entry['W'])]
            for exp_id in range(len(headers[2:])):
                if exp_id < len(entry['accs']):
                    row.append(f'{entry["accs"][exp_id]}')
                else: row.append('-')
            table.append(row)
        
        pretty_table = tabulate(table, headers=headers, tablefmt='pipe')
        print(pretty_table + '\n')
        return(pretty_table)

def average_merged_res(merged_res):
    return merged_res[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--dataset_name', type=str, default='SplitCIFAR100')
    parser.add_argument('--dataset_root', type=str, default='/raid/l.quarantiello/datasets')
    parser.add_argument('--n_classes_per_exp', type=int, default=5)

    # model arguments
    parser.add_argument('--base_model', type=str, default='vit_base_patch16_224')

    # lora arguments
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_root', type=str, default='/raid/l.quarantiello/adaptive_lora_merging/loras')

    # training arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    # train or test?
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_false', dest='training')

    args = parser.parse_args()

    if args.training:
        for seed in [1, 10, 42, 101, 3333]: single_seed_train(args, seed)
    else:
        merged_res = []
        for seed in [1]:#, 10, 42, 101, 3333]:
            merged_res.append(single_run_class_incremental_testing(args, seed))

        avg_merged_res = average_merged_res(merged_res)
        table = get_result_table(avg_merged_res)