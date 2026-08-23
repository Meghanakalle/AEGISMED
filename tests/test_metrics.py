from torch.utils.data import DataLoader
from models.classifier import build_model
from evaluation.metrics import evaluate_model
from data.dataset import MedicalDataset
from data.synthetic import generate_synthetic_medical_xray
from data.preprocessing import get_transforms
from utils.config import load_config

def test_metrics_computation():
    config = load_config("configs/fast_validation.yaml")
    num_classes = config["model"]["num_classes"]
    labels = [i % num_classes for i in range(15)]
    images = [generate_synthetic_medical_xray(lbl, image_size=64) for lbl in labels]

    eval_tf = get_transforms(image_size=64, is_train=False)
    ds = MedicalDataset(images, labels, transform=eval_tf)
    loader = DataLoader(ds, batch_size=5)

    model = build_model(config)
    metrics = evaluate_model(model, loader, device="cpu")

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["confusion_matrix"].shape == (num_classes, num_classes)
