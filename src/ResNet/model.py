import torch
import torch.nn as nn
from torchvision import models

def create_resnet18(num_classes=6, pretrained=True):
    """
    Initializes ResNet-18 with ImageNet weights and replaces the final FC layer.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_cost_sensitive_weights(device="cpu"):
    """
    Returns normalized cost-sensitive class weights:
    [10k: 0.36, 20k: 0.39, 50k: 0.49, 100k: 0.69, 200k: 1.17, 500k: 2.67]
    Derived from the 6x6 Cost Matrix normalized against train set distribution.
    """
    weights = [0.36, 0.39, 0.49, 0.69, 1.17, 2.67]
    return torch.tensor(weights, dtype=torch.float32, device=device)

