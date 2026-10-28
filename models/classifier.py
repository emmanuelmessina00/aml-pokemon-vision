import torch.nn as nn
from torchvision import models


def get_pokemon_resnet(num_classes=5, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model