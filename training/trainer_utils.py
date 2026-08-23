from typing import Tuple
import torch

def compute_loss_and_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module
) -> Tuple[torch.Tensor, int, int]:
    loss = criterion(outputs, targets)
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == targets.data).item()
    total = targets.size(0)
    return loss, corrects, total
