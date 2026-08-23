import torch.nn as nn
import torchvision.models as torchvision_models
from models.efficientnet import EfficientNetB3Medical

def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    backbone = model_cfg.get("backbone", "efficientnet_b3")
    num_classes = model_cfg.get("num_classes", 3)
    pretrained = model_cfg.get("pretrained", False)
    dropout = model_cfg.get("dropout", 0.3)

    if backbone == "efficientnet_b3":
        return EfficientNetB3Medical(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif backbone == "resnet18":
        weights = torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision_models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone}")
