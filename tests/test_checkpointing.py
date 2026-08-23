import os
import torch
from models.classifier import build_model
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.config import load_config

def test_checkpoint_save_and_load(tmp_path):
    config = load_config("configs/fast_validation.yaml")
    model = build_model(config)
    save_dir = str(tmp_path)
    
    filepath = save_checkpoint(model, round_num=5, metric_val=0.885, save_dir=save_dir, filename="test_chk.pt")
    assert os.path.exists(filepath)

    new_model = build_model(config)
    round_num, metric_val = load_checkpoint(new_model, filepath)
    assert round_num == 5
    assert metric_val == 0.885
