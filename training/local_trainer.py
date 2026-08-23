import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from privacy.dp_optimizer import DPOptimizerWrapper

class LocalTrainer:
    """Local PyTorch Trainer implementing FedProx proximal penalty and Differential Privacy."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.0005,
        mu: float = 0.01, # FedProx parameter (mu=0 gives FedAvg)
        use_dp: bool = False,
        dp_epsilon: float = 3.0,
        dp_delta: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.mu = mu
        self.use_dp = use_dp
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if self.use_dp:
            self.optimizer = DPOptimizerWrapper(
                base_optimizer,
                max_grad_norm=max_grad_norm,
                noise_multiplier=1.1,
                target_epsilon=dp_epsilon,
                target_delta=dp_delta
            )
        else:
            self.optimizer = base_optimizer

    def train_epoch(self, train_loader, global_model_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Snapshot global model weights for FedProx proximal penalty calculation
        global_params = None
        if global_model_weights is not None and self.mu > 0:
            global_params = {
                k: v.clone().detach().to(self.device)
                for k, v in global_model_weights.items()
            }

        for images, labels, _ in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.use_dp:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # FedProx Proximal Penalty term: (mu / 2) * || w - w_global ||^2
            if global_params is not None and self.mu > 0:
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        proximal_term += torch.sum((param - global_params[name]) ** 2)
                loss += (self.mu / 2.0) * proximal_term

            loss.backward()

            if self.use_dp:
                self.optimizer.step(self.model)
            else:
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        privacy_info = (0.0, 0.0)
        if self.use_dp:
            privacy_info = self.optimizer.get_privacy_spent()

        return {
            "loss": float(epoch_loss),
            "accuracy": float(epoch_acc),
            "epsilon": float(privacy_info[0]),
            "delta": float(privacy_info[1])
        }

    def evaluate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        val_loss = total_loss / max(1, total)
        val_acc = correct / max(1, total)
        return {"loss": float(val_loss), "accuracy": float(val_acc)}
