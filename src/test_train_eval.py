import os
import avalanche
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, confusion_matrix_metrics, forgetting_metrics, loss_metrics, StreamConfusionMatrix, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from peft import LoraConfig, get_peft_model
import timm

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        ])
benchmark = SplitCIFAR100(n_experiences=20, seed=1, dataset_root='/raid/l.quarantiello/datasets', train_transform=transform, eval_transform=transform)


base_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=benchmark.n_classes)

# create peft model
lora_config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.1,
    target_modules = ['qkv', 'fc1', 'fc2']
)
peft_model = get_peft_model(base_model, lora_config)

for exp in benchmark.train_stream:
    print(f'Training on {exp.classes_in_this_experience}')

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

    #peft_model.add_adapter('prova', lora_config)
    #peft_model.set_adapter('prova')
    training_strategy = Naive(
        peft_model, Adam(peft_model.parameters(), lr=1e-4), nn.CrossEntropyLoss(),
        train_mb_size = 32, train_epochs = 10, eval_mb_size = 32,
        evaluator = eval_plugin, device = 'cuda' 
    )
    #res = training_strategy.train(exp)
    break

for exp in benchmark.test_stream:
    print(f'Testing on {exp.classes_in_this_experience}')
    lora_name = '_'.join([str(c) for c in exp.classes_in_this_experience])
    lora_path = os.path.join('/raid/l.quarantiello/adaptive_lora_merging/loras', str(1), lora_name)

    # load the adapter related to the current classes
    if os.path.exists(lora_path):
        peft_model.load_adapter(lora_path, lora_name)
        print(f'Loaded adapter from {lora_path}')

    weights = [1.0]
    peft_model.add_weighted_adapter([lora_name], weights, 'ties', 'ties', density=0.2)
    peft_model.set_adapter('ties')

    training_strategy = Naive(
        peft_model, Adam(peft_model.parameters(), lr=1e-4), nn.CrossEntropyLoss(),
        train_mb_size = 32, train_epochs = 10, eval_mb_size = 32,
        evaluator = eval_plugin, device = 'cuda' 
    )

    res = training_strategy.eval(exp)
    break

breakpoint()