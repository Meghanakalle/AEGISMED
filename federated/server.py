import os
import pandas as pd
import torch
from models.classifier import build_model
from federated.client import HospitalFLClient
from federated.fedprox import fedprox_aggregate
from evaluation.metrics import evaluate_model
from privacy.privacy_accountant import RDPPrivacyAccountant
from compression.communication import CommunicationTracker, calculate_state_dict_size_mb
from utils.checkpointing import save_checkpoint
from utils.logging_utils import setup_logger

class FederatedServer:
    def __init__(self, config: dict, hospital_loaders: list, global_test_loader, device: str = "cpu", global_val_loader=None):
        self.config = config
        self.hospital_loaders = hospital_loaders
        self.global_test_loader = global_test_loader
        self.global_val_loader = global_val_loader
        self.device = device
        self.logger = setup_logger("FederatedServer")

        self.global_model = build_model(config).to(self.device)
        self.global_weights = self.global_model.state_dict()

        fed_cfg = config["federated"]
        priv_cfg = config["privacy"]

        # Initialize Hospital FL Clients
        self.clients = [
            HospitalFLClient(
                hospital_id=i,
                hospital_name=fed_cfg["hospital_names"][i] if i < len(fed_cfg["hospital_names"]) else f"Hospital_{i}",
                train_loader=loader,
                config=config,
                device=device
            )
            for i, loader in enumerate(hospital_loaders)
        ]

        self.privacy_accountant = RDPPrivacyAccountant(target_delta=priv_cfg.get("target_delta", 1e-4))
        
        uncompressed_mb = calculate_state_dict_size_mb(self.global_weights)
        self.final_test_metrics = None

        self.comm_tracker = CommunicationTracker(
            num_hospitals=len(self.clients),
            uncompressed_size_mb=uncompressed_mb
        )

    def run_training_loop(self, rounds: int = None) -> pd.DataFrame:
        if rounds is None:
            rounds = self.config["federated"].get("num_rounds", 40)

        history = []
        best_accuracy = 0.0

        priv_cfg = self.config["privacy"]
        comp_cfg = self.config["compression"]
        k_ratio = comp_cfg.get("top_k_ratio", 0.20) if comp_cfg.get("enabled", True) else 1.0

        for r in range(1, rounds + 1):
            client_weights = []
            client_counts = []
            client_losses = []

            # 1. Local hospital training
            for client in self.clients:
                weights, sample_cnt, loss = client.train_local_epoch(self.global_weights, round_idx=r)
                client_weights.append(weights)
                client_counts.append(sample_cnt)
                client_losses.append(loss)

            # 2. Server FedProx weight aggregation
            self.global_weights = fedprox_aggregate(client_weights, client_counts)
            self.global_model.load_state_dict(self.global_weights)

            # 3. Privacy budget update
            if priv_cfg.get("enabled", True) and len(self.clients) > 0:
                client = self.clients[0]
                local_epochs = self.config["federated"].get("local_epochs", 2)
                num_batches = len(client.train_loader)
                steps_per_round = local_epochs * num_batches
                
                batch_size = self.config["federated"].get("batch_size", 16)
                n_samples = len(client.train_loader.dataset)
                sample_rate = batch_size / max(1, n_samples)
                
                self.privacy_accountant.step(
                    noise_multiplier=priv_cfg.get("noise_multiplier", 1.1),
                    sample_rate=sample_rate,
                    steps=steps_per_round
                )
            epsilon = self.privacy_accountant.get_epsilon()

            # 4. Communication tracking (Download is dense, Upload is sparse)
            self.comm_tracker.log_round(self.global_weights, k_ratio=k_ratio, enabled=comp_cfg.get("enabled", True))
            comm_summary = self.comm_tracker.get_summary()

            # Evaluate on validation split if available
            val_acc = 0.0
            val_loss = 0.0
            if self.global_val_loader is not None:
                val_metrics = evaluate_model(self.global_model, self.global_val_loader, self.device)
                val_acc = val_metrics["accuracy"]
                val_loss = val_metrics["loss"]

            # 5. Global model evaluation
            test_metrics = evaluate_model(self.global_model, self.global_test_loader, self.device)
            if r == rounds:
                self.final_test_metrics = test_metrics
            avg_client_loss = sum(client_losses) / len(client_losses)

            round_record = {
                "round": r,
                "train_loss": round(avg_client_loss, 4),
                "val_loss": round(val_loss, 4) if self.global_val_loader is not None else 0.0,
                "val_accuracy": round(val_acc, 4) if self.global_val_loader is not None else 0.0,
                "test_loss": round(test_metrics["loss"], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "precision": round(test_metrics["precision"], 4),
                "recall": round(test_metrics["recall"], 4),
                "f1_score": round(test_metrics["f1_score"], 4),
                "roc_auc": round(test_metrics["roc_auc"], 4),
                "privacy_epsilon": epsilon,
                "uncompressed_comm_mb": comm_summary["uncompressed_total_mb"],
                "compressed_comm_mb": comm_summary["compressed_total_mb"],
                "communication_reduction_percent": comm_summary["communication_reduction_percent"]
            }
            history.append(round_record)

            if self.global_val_loader is not None:
                self.logger.info(
                    f"Round {r:02d}/{rounds:02d} | Train Loss: {avg_client_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Test Acc: {test_metrics['accuracy'] * 100:.2f}% | "
                    f"ROC-AUC: {test_metrics['roc_auc']:.4f} | Privacy epsilon: {epsilon:.2f} | Comm Savings: {comm_summary['communication_reduction_percent']}%"
                )
            else:
                self.logger.info(
                    f"Round {r:02d}/{rounds:02d} | Train Loss: {avg_client_loss:.4f} | Test Acc: {test_metrics['accuracy'] * 100:.2f}% | "
                    f"ROC-AUC: {test_metrics['roc_auc']:.4f} | Privacy epsilon: {epsilon:.2f} | Comm Savings: {comm_summary['communication_reduction_percent']}%"
                )

            # Save checkpoint if validation accuracy improves (fallback to test if validation loader not available)
            val_metric = val_acc if self.global_val_loader is not None else test_metrics["accuracy"]
            if val_metric > best_accuracy:
                best_accuracy = val_metric
                chk_dir = self.config["paths"].get("checkpoints_dir", "checkpoints")
                save_checkpoint(self.global_model, round_num=r, metric_val=best_accuracy, save_dir=chk_dir)

        df_results = pd.DataFrame(history)
        results_dir = self.config["paths"].get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        df_results.to_csv(os.path.join(results_dir, "experiment_metrics.csv"), index=False)

        return df_results
