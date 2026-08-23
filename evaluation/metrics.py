import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def evaluate_model(model: nn.Module, data_loader, device: str = "cpu") -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * len(y_batch)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_batch.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    num_samples = len(all_targets)
    avg_loss = total_loss / max(1, num_samples)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    
    try:
        classes_present = np.unique(all_targets)
        if len(classes_present) < 2:
            roc_auc = 0.5000
        else:
            probs_arr = np.array(all_probs)
            if probs_arr.shape[1] > len(classes_present):
                probs_present = probs_arr[:, classes_present]
                row_sums = probs_present.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1e-8, row_sums)
                probs_present = probs_present / row_sums
                roc_auc = roc_auc_score(all_targets, probs_present, multi_class="ovr", average="macro", labels=classes_present)
            else:
                roc_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        import logging
        logging.getLogger("metrics").warning(f"Could not compute ROC-AUC: {e}. Falling back to 0.5000.")
        roc_auc = 0.5000

    cm = confusion_matrix(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": np.array(all_probs)
    }
