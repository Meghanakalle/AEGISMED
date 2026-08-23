import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Use high-quality visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_publication_plots(results_df: pd.DataFrame, hospital_stats: list, save_dir: str = "figures", test_metrics: dict = None, class_names: list = None, alpha: float = 0.3):
    os.makedirs(save_dir, exist_ok=True)
    if class_names is None:
        class_names = ["Normal", "Lung_Opacity", "Viral Pneumonia", "COVID"]

    # 1. Global Accuracy vs. FL Rounds
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["round"], results_df["test_accuracy"] * 100, marker='o', color='#0284c7', linewidth=2.5, label='Global Model Test Accuracy')
    plt.title("Federated Learning Convergence (40 Rounds)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("FL Communication Round", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # 2. Loss Curves (Train Loss vs Test Loss)
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["round"], results_df["train_loss"], label='Avg Hospital Train Loss', color='#e11d48', linewidth=2.0)
    plt.plot(results_df["round"], results_df["test_loss"], label='Global Test Loss', color='#2563eb', linestyle='--', linewidth=2.0)
    plt.title("Training & Test Loss Trajectory", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("FL Round", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=300)
    plt.close()

    # 3. Privacy Epsilon Budget Growth over Rounds
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["round"], results_df["privacy_epsilon"], color='#8b5cf6', linewidth=2.5, marker='s', label='RDP Cumulative Epsilon (ε)')
    plt.axhline(y=3.0, color='#dc2626', linestyle=':', label='Target Privacy Guarantee (ε = 3.0)')
    plt.title("Differential Privacy Epsilon (ε) Budget Expenditure", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("FL Round", fontsize=11)
    plt.ylabel("Epsilon (ε)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "privacy_epsilon_growth.png"), dpi=300)
    plt.close()

    # 4. Communication Overhead Reduction Comparison
    plt.figure(figsize=(7, 5))
    uncompressed = results_df["uncompressed_comm_mb"].iloc[-1]
    compressed = results_df["compressed_comm_mb"].iloc[-1]
    categories = ['Uncompressed Standard FL', 'Proposed Top-K Sparse FL']
    values = [uncompressed, compressed]
    colors = ['#94a3b8', '#10b981']
    
    bars = plt.bar(categories, values, color=colors, width=0.5)
    plt.ylabel("Total Transmitted Data (MB)", fontsize=11)
    plt.title("Total Inter-Hospital Communication Overhead (40 Rounds)", fontsize=13, fontweight='bold', pad=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values)*0.02), f"{yval:.1f} MB", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "communication_comparison.png"), dpi=300)
    plt.close()

    # 5. Multiclass ROC Curves
    plt.figure(figsize=(7, 6))
    colors_roc = ['#2563eb', '#16a34a', '#dc2626', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f43f5e']
    
    if test_metrics is not None and "targets" in test_metrics and "probabilities" in test_metrics:
        try:
            targets = np.array(test_metrics["targets"])
            probs = np.array(test_metrics["probabilities"])
            num_classes = len(class_names)
            
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(targets, probs[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors_roc[0], label=f'{class_names[1]} (AUC = {roc_auc:.3f})', linewidth=2.0)
            else:
                y_one_hot = label_binarize(targets, classes=list(range(num_classes)))
                for idx in range(num_classes):
                    if idx < y_one_hot.shape[1]:
                        if len(np.unique(y_one_hot[:, idx])) > 1:
                            fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probs[:, idx])
                            roc_auc = auc(fpr, tpr)
                            color = colors_roc[idx % len(colors_roc)]
                            plt.plot(fpr, tpr, color=color, label=f'{class_names[idx]} (AUC = {roc_auc:.3f})', linewidth=2.0)
                        else:
                            plt.plot([0, 1], [0, 0], color=colors_roc[idx % len(colors_roc)], label=f'{class_names[idx]} (No samples in split)', linewidth=2.0)
        except Exception as e:
            print(f"[!] Warning: Failed to compute actual ROC curves: {e}. Plotting diagonal.")
    else:
        print("[!] Warning: test_metrics is missing or empty. Cannot plot actual ROC curves.")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Multiclass ROC Curves (Global Model)", fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 6. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    if test_metrics is not None and "confusion_matrix" in test_metrics:
        cm = test_metrics["confusion_matrix"]
    else:
        cm = np.zeros((len(class_names), len(class_names)), dtype=int)
        print("[!] Warning: test_metrics is missing confusion_matrix. Plotting empty matrix.")
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Global Test Set Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 7. Non-IID Dirichlet Class Heterogeneity Stacked Bar
    if hospital_stats:
        plt.figure(figsize=(8, 5))
        rows = []
        for h in hospital_stats:
            row = {"Hospital": h["hospital_name"]}
            for idx, name in enumerate(class_names):
                row[name] = h["class_distribution"].get(idx, 0)
            rows.append(row)
        df_stats = pd.DataFrame(rows)
        
        colors_bar = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f43f5e']
        df_stats.set_index("Hospital").plot(kind='bar', stacked=True, color=colors_bar[:len(class_names)], figsize=(8,5))
        plt.title(f"Non-IID Hospital Data Heterogeneity Distribution (Dirichlet α={alpha})", fontsize=12, fontweight='bold')
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=15, ha='right')
        plt.legend(title="Disease Category")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "non_iid_hospital_distribution.png"), dpi=300)
        plt.close()

    return save_dir
