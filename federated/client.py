import torch
import torch.nn as nn
from models.classifier import build_model
from federated.fedprox import compute_fedprox_proximal_term
from privacy.dp_optimizer import DPOptimizerWrapper
from compression.top_k import apply_top_k_sparsification

class HospitalFLClient:
    def __init__(self, hospital_id: int, hospital_name: str, train_loader, config: dict, device: str = "cpu"):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.model = build_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_local_epoch(self, global_weights: dict, round_idx: int = 1):
        self.model.load_state_dict(global_weights)
        self.model.train()

        fed_cfg = self.config["federated"]
        priv_cfg = self.config["privacy"]
        comp_cfg = self.config["compression"]

        epochs = fed_cfg.get("local_epochs", 2)
        lr = fed_cfg.get("lr", 0.001)
        mu_prox = fed_cfg.get("mu_prox", 0.01)

        # Decay learning rate across rounds to improve stability and convergence
        lr_round = lr * (0.95 ** (round_idx - 1))

        # Use AdamW for better fine-tuning generalization
        base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_round, weight_decay=1e-4)

        if priv_cfg.get("enabled", True):
            optimizer = DPOptimizerWrapper(
                base_optimizer,
                max_grad_norm=priv_cfg.get("max_grad_norm", 1.0),
                noise_multiplier=priv_cfg.get("noise_multiplier", 1.5)
            )
        else:
            optimizer = base_optimizer

        total_loss = 0.0
        total_samples = 0

        # Pre-flatten global weights for extremely fast FedProx computation
        global_params_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                g_weight = global_weights[name] if name in global_weights else param.detach()
                global_params_list.append(g_weight.to(self.device).flatten())
        global_weights_flat = torch.cat(global_params_list) if global_params_list else None

        for _ in range(epochs):
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(x_batch)
                cls_loss = self.criterion(outputs, y_batch)
                prox_loss = compute_fedprox_proximal_term(self.model, global_weights_flat, mu=mu_prox)
                loss = cls_loss + prox_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        updated_weights = self.model.state_dict()

        # Apply Top-K sparsification if enabled
        if comp_cfg.get("enabled", True):
            k_ratio = comp_cfg.get("top_k_ratio", 0.20)
            from compression.top_k import _NEVER_SPARSIFY_SUBSTRINGS
            
            # Calculate weight updates (differences) for sparsifiable layers
            weight_updates = {}
            for name, param in updated_weights.items():
                if param.is_floating_point() and not any(s in name for s in _NEVER_SPARSIFY_SUBSTRINGS):
                    weight_updates[name] = param - global_weights[name].to(self.device)
            
            # Apply Top-K compression on the updates
            compressed_updates = apply_top_k_sparsification(weight_updates, k_ratio=k_ratio)
            
            # Reconstruct the sparsified weights: global_weights + compressed_updates
            for name in updated_weights.keys():
                if name in compressed_updates:
                    updated_weights[name] = global_weights[name].to(self.device) + compressed_updates[name]

        avg_loss = total_loss / max(1, total_samples)
        return updated_weights, len(self.train_loader.dataset), avg_loss
