import os
import torch

def save_checkpoint(model, round_num: int, metric_val: float, save_dir: str, filename: str = "best_model.pt"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    checkpoint = {
        "round": round_num,
        "state_dict": model.state_dict(),
        "metric": metric_val
    }
    torch.save(checkpoint, filepath)
    return filepath

def load_checkpoint(model, filepath: str, device: str = "cpu"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("round", 0), checkpoint.get("metric", 0.0)
