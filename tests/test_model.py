import torch
from models.efficientnet import EfficientNetB3Medical
from models.classifier import build_model
from utils.config import load_config

def test_efficientnet_b3_architecture():
    model = EfficientNetB3Medical(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

def test_build_model_from_config():
    config = load_config("configs/fast_validation.yaml")
    model = build_model(config)
    x = torch.randn(2, 3, config["dataset"]["image_size"], config["dataset"]["image_size"])
    out = model(x)
    assert out.shape == (2, config["dataset"]["num_classes"])
