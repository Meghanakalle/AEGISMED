import os
import pandas as pd
import numpy as np

def generate_markdown_research_report(
    results_df: pd.DataFrame,
    config: dict = None,
    save_dir: str = "reports",
    test_metrics: dict = None,
    hospital_stats: list = None,
    dataset_sizes: dict = None
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "final_experiment_report.md")

    final_row = results_df.iloc[-1]
    final_acc = final_row["test_accuracy"] * 100.0
    final_eps = final_row["privacy_epsilon"]
    comm_reduction = final_row["communication_reduction_percent"]
    uncompressed_mb = final_row["uncompressed_comm_mb"]
    compressed_mb = final_row["compressed_mb"] if "compressed_mb" in final_row else final_row.get("compressed_comm_mb", 0.0)

    # Dynamic status evaluations
    acc_status = f"Exceeded (+{final_acc - 85.0:.2f}%)" if final_acc >= 85.0 else f"Failed Target ({final_acc - 85.0:.2f}%)"
    auc_status = "Exceeded" if final_row['roc_auc'] >= 0.880 else "Failed Target"
    dp_status = "Verified Compliant" if final_eps <= 3.5 else "Non-Compliant (Exceeds Target)"
    comm_status = "Exceeded" if comm_reduction >= 60.0 else "Failed Target"

    # Default parameters if config is missing
    num_classes = 4
    classes_list = ["Normal", "Lung_Opacity", "Viral Pneumonia", "COVID"]
    class_names = ", ".join(classes_list)
    backbone = "Resnet18"
    top_k_ratio = 0.20
    noise_multiplier = 1.5
    max_grad_norm = 1.0
    mu_prox = 0.01
    dirichlet_alpha = 0.3
    batch_size = 16
    local_epochs = 2

    if config is not None:
        dataset_cfg = config.get("dataset", {})
        num_classes = dataset_cfg.get("num_classes", num_classes)
        classes_list = dataset_cfg.get("classes", classes_list)
        class_names = ", ".join(classes_list)
        backbone = config.get("model", {}).get("backbone", backbone).replace("_", "-").title()
        top_k_ratio = config.get("compression", {}).get("top_k_ratio", top_k_ratio)
        noise_multiplier = config.get("privacy", {}).get("noise_multiplier", noise_multiplier)
        max_grad_norm = config.get("privacy", {}).get("max_grad_norm", max_grad_norm)
        mu_prox = config.get("federated", {}).get("mu_prox", mu_prox)
        dirichlet_alpha = dataset_cfg.get("dirichlet_alpha", dirichlet_alpha)
        batch_size = config.get("federated", {}).get("batch_size", batch_size)
        local_epochs = config.get("federated", {}).get("local_epochs", local_epochs)

    top_k_pct = top_k_ratio * 100.0

    # Dataset split sizes
    train_size = dataset_sizes.get("train", 0) if dataset_sizes else 0
    val_size = dataset_sizes.get("val", 0) if dataset_sizes else 0
    test_size = dataset_sizes.get("test", 0) if dataset_sizes else 0

    # Class distribution table across hospitals
    hospital_dist_table = ""
    if hospital_stats is not None:
        h_rows = []
        h_header = "| Hospital | Total Samples | " + " | ".join(classes_list) + " |"
        h_divider = "| :--- | :---: | " + " | ".join([":---:"] * len(classes_list)) + " |"
        for h in hospital_stats:
            row_str = f"| {h['hospital_name']} | {h['sample_count']} | "
            counts = [str(h['class_distribution'].get(c_idx, 0)) for c_idx in range(len(classes_list))]
            row_str += " | ".join(counts) + " |"
            h_rows.append(row_str)
        hospital_dist_table = "\n".join([h_header, h_divider] + h_rows)

    # Confusion matrix and per-class metrics
    cm_table = "Confusion matrix not available."
    per_class_table = "Per-class metrics not available."
    if test_metrics is not None and "confusion_matrix" in test_metrics:
        cm = test_metrics["confusion_matrix"]
        cm_header = "| Actual \\ Predicted | " + " | ".join(classes_list) + " |"
        cm_divider = "| :--- | " + " | ".join([":---:"] * len(classes_list)) + " |"
        cm_rows = []
        for i, row in enumerate(cm):
            cm_rows.append(f"| **{classes_list[i]}** | " + " | ".join(map(str, row)) + " |")
        cm_table = "\n".join([cm_header, cm_divider] + cm_rows)

        # Calculate per-class metrics
        per_class_rows = []
        for i in range(len(classes_list)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_rows.append(f"| **{classes_list[i]}** | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
        per_class_table = "\n".join([
            "| Class | Precision | Recall | F1-Score |",
            "| :--- | :---: | :---: | :---: |"
        ] + per_class_rows)

    # DP steps calculation
    dp_steps_str = "Not calculated"
    if hospital_stats is not None and len(hospital_stats) > 0:
        h0_samples = hospital_stats[0]["sample_count"]
        h0_batches = (h0_samples + batch_size - 1) // batch_size
        total_rounds = len(results_df)
        total_steps = total_rounds * local_epochs * h0_batches
        dp_steps_str = f"{total_steps} steps ({total_rounds} rounds x {local_epochs} local epochs x {h0_batches} local batches)"
        sampling_rate = batch_size / max(1, h0_samples)
    else:
        sampling_rate = 0.0

    # Load ablation summary
    ablation_table = "Ablation study data not found. Run ablation study to generate."
    ablation_csv = "results/ablation_study_summary.csv"
    if os.path.exists(ablation_csv):
        try:
            df_ablation = pd.read_csv(ablation_csv)
            ablation_table = df_ablation.to_markdown(index=False)
        except Exception as e:
            ablation_table = f"Error loading ablation study: {e}"

    # Load Non-IID Dirichlet alpha study
    non_iid_table = "Non-IID study data not found. Run Non-IID study to generate."
    non_iid_csv = "results/non_iid_alpha_study.csv"
    if os.path.exists(non_iid_csv):
        try:
            df_non_iid = pd.read_csv(non_iid_csv)
            non_iid_table = df_non_iid.to_markdown(index=False)
        except Exception as e:
            non_iid_table = f"Error loading Non-IID study: {e}"

    # Generate snapshot rows
    table_indices = sorted(list(set([0, len(results_df) - 1] + list(range(0, len(results_df), max(1, len(results_df) // 4))))))
    table_df = results_df.iloc[table_indices]

    report_content = rf"""# Major Research Project Report
## Communication-Efficient Federated Learning with Differential Privacy Guarantees for Distributed Image Classification Across Heterogeneous Hospital Networks

### Executive Summary
This major project develops a privacy-preserving, communication-efficient federated learning system tailored for multi-hospital clinical networks. By combining **FedProx** parameter aggregation with **Renyi Differential Privacy (RDP)** ($\epsilon={final_eps:.2f}, \delta=10^{{-4}}$) and **Top-K Update Sparsification** (top {top_k_pct:.0f}% magnitude tensor selection), our architecture achieves **{final_acc:.2f}% Global Test Accuracy** and **{final_row['roc_auc']:.4f} ROC-AUC** on {backbone} while reducing inter-hospital bandwidth overhead by **{comm_reduction:.1f}%**.

---

### Core Experimental Findings

| Metric | Target Baseline | Achieved Result | Evaluation Status |
| :--- | :--- | :--- | :--- |
| **Global Test Accuracy** | 85.0% | **{final_acc:.2f}%** | {acc_status} |
| **ROC-AUC (Macro OVR)** | 0.880 | **{final_row['roc_auc']:.4f}** | {auc_status} |
| **Differential Privacy ($\epsilon$)** | $\le 3.5$ | **$\epsilon = {final_eps:.2f}$** ($\delta=10^{{-4}}$) | {dp_status} |
| **Bandwidth Overhead Reduction** | $\ge 60.0\%$ | **{comm_reduction:.1f}% Reduction** | {comm_status} |
| **Uncompressed Transmitted Data** | ~3,840 MB | **{uncompressed_mb:.1f} MB** | Baseline |
| **Compressed Transmitted Data** | < 1,500 MB | **{compressed_mb:.1f} MB** | Compressed Payload |

---

### Dataset Details
- **Dataset Name:** COVID-19 Radiography Dataset
- **Total Split Sizes:**
  - **Training Set:** {train_size} samples
  - **Validation Set:** {val_size} samples
  - **Held-out Test Set:** {test_size} samples

#### Hospital Client Sample & Class Distributions
{hospital_dist_table}

---

### Model & Training Configurations
- **Model Backbone:** Pretrained **{backbone}**
- **Image Size:** {config.get('dataset', {}).get('image_size', 128)}x{config.get('dataset', {}).get('image_size', 128)}
- **Optimizer:** AdamW with Weight Decay (`1e-4`)
- **Initial Learning Rate:** {config.get('federated', {}).get('lr', 0.001)}
- **FedProx Regularization ($\mu$):** {mu_prox}
- **Top-K Update Sparsification Keep Ratio:** {top_k_ratio} (i.e., top {top_k_pct:.0f}% updates kept)
- **Differential Privacy Config:**
  - **Clipping Norm ($C$):** {max_grad_norm}
  - **Noise Multiplier ($\sigma$):** {noise_multiplier}
  - **Client Sampling Rate ($q$):** {sampling_rate:.4f}
  - **Total Accountant Steps:** {dp_steps_str}
  - **Target Delta ($\delta$):** 0.0001
  - **Final Privacy Epsilon ($\epsilon$):** {final_eps:.4f}

---

### Per-Class Performance Metrics
{per_class_table}

#### Confusion Matrix
{cm_table}

---

### Ablation Study Results
{ablation_table}

---

### Non-IID Dirichlet Alpha Sensitivity Study
{non_iid_table}

---

### Baseline Comparisons
We compare our proposed privacy-preserving, communication-sparse framework against the standard federated learning baselines reported in the reference literature:

| Method | Learning Rate | Accuracy | Loss | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fed-Avg | 5e-3 | 87.87% | 0.436 | 0.878 | 0.878 | 0.878 |
| Fed-Avg | 5e-4 | 90.92% | 0.380 | 0.909 | 0.909 | 0.909 |
| Fed-Avg | 1e-5 | 86.36% | 0.457 | 0.863 | 0.863 | 0.863 |
| Edge-Avg | 5e-3 | 89.13% | 0.408 | 0.891 | 0.891 | 0.891 |
| **Edge-Avg (Best)** | **5e-4** | **93.94%** | **0.370** | **0.939** | **0.939** | **0.939** |
| Edge-Avg | 1e-5 | 87.39% | 0.443 | 0.873 | 0.873 | 0.873 |
| **Proposed Framework** | **{config.get('federated', {}).get('lr', 0.0005)}** | **{final_acc:.2f}%** | **{final_row['test_loss']:.4f}** | **{final_row['precision']:.4f}** | **{final_row['recall']:.4f}** | **{final_row['f1_score']:.4f}** |

---

### Limitations & Analysis
1. **CPU Execution Limitations:** Due to the lack of hardware GPU/CUDA acceleration on the client's side, training takes significantly longer. Subsampling the dataset (e.g. 600 per class) represents a practical trade-off to allow scientific execution of the entire ablation/sensitivity matrix.
2. **Privacy vs. Utility Tradeoff:** Stronger differential privacy guarantees (higher noise multiplier, lower epsilon) naturally degrade the accuracy compared to non-private baselines like Edge-Avg.
3. **Statistical Skew (Non-IID):** Extremely small Dirichlet alpha ($\alpha = 0.1$) causes extreme client data imbalance which impacts convergence stability, requiring FedProx proximal term adjustment.

---
*Report automatically generated by Major Project Evaluation Engine.*
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    return report_path
