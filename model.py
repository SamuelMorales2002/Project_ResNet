# model.py
"""
Módulo para definición y construcción del modelo ResNet-50.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Retorna ResNet-50 con capa FC reemplazada para clasificación binaria
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # Salida: 1 logit

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """Carga un checkpoint guardado"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint