import torch
import torch.nn as nn
import torchvision
from torch.nn.utils import prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
percent = 0.3

# Model definition
net = torchvision.models.resnet50(pretrained=False).to(device)
net.fc = nn.Linear(512 * 4, 10).to(device)
net.load_state_dict(torch.load('./heritage_key_model/net_128.pth'))

parameters_to_prune = []
def add_paras(parameters_to_prune):
    for i, j in list(net.named_modules()):
        if isinstance(j, nn.BatchNorm2d) or isinstance(j, nn.Conv2d):
            if not j.bias == None:
                parameters_to_prune.append((j, 'bias'))
            if not j.weight == None:
                parameters_to_prune.append((j, 'weight'))
add_paras(parameters_to_prune)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=percent,
)
