import argparse
import torch
from utils.config import load_config
from utils.checkpointing import load_checkpoint
from models.classifier import build_model
from data.loader import get_hospital_dataloaders
from evaluation.metrics import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Global Model Evaluator")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint .pt file")

    args = parser.parse_args()
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, global_test_loader, _ = get_hospital_dataloaders(config)
    model = build_model(config).to(device)

    chk_round, chk_metric = load_checkpoint(model, args.checkpoint, device=device)
    print(f"[+] Loaded Model Checkpoint from Round {chk_round} (Metric: {chk_metric:.4f})")

    metrics = evaluate_model(model, global_test_loader, device=device)
    
    print("\n=== Global Model Evaluation Summary ===")
    print(f"Test Loss:     {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
