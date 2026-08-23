import os
import copy
import pandas as pd
from utils.config import load_config
from experiments.runner import run_federated_experiment

def run_ablation_study(config_path: str = "configs/research.yaml"):
    """
    Executes a 5-variant ablation study comparing:
    1. Base FedAvg (No DP, No Compression, No Proximal)
    2. FedAvg + FedProx (mu = 0.01)
    3. FedAvg + Differential Privacy (eps = 3.0)
    4. FedAvg + Top-K Sparsification (Top 20%)
    5. Proposed Combined Framework (FedProx + DP + Top-K)
    """
    variants = [
        {"name": "FedAvg_Base", "mu": 0.0, "dp": False, "topk": False, "k_ratio": 1.0},
        {"name": "FedProx_Only", "mu": 0.01, "dp": False, "topk": False, "k_ratio": 1.0},
        {"name": "FedAvg_DP_Only", "mu": 0.0, "dp": True, "topk": False, "k_ratio": 1.0},
        {"name": "FedAvg_TopK_Only", "mu": 0.0, "dp": False, "topk": True, "k_ratio": 0.20},
        {"name": "Proposed_FedProx_DP_TopK", "mu": 0.01, "dp": True, "topk": True, "k_ratio": 0.20}
    ]

    base_cfg = load_config(config_path)
    ablation_summary = []

    for v in variants:
        cfg = copy.deepcopy(base_cfg)  # deepcopy: base_cfg's nested dicts must not be shared/mutated across variants
        cfg["federated"]["mu_prox"] = v["mu"]
        cfg["privacy"]["enabled"] = v["dp"]
        cfg["compression"]["enabled"] = v["topk"]
        cfg["compression"]["top_k_ratio"] = v["k_ratio"]
        cfg["federated"]["num_rounds"] = 5  # Ablation quick run

        print(f"\n--- Ablation variant: {v['name']} ---")
        df = run_federated_experiment(config_path, mode="ablation", config_override=cfg)
        final_row = df.iloc[-1]

        ablation_summary.append({
            "Variant": v["name"],
            "Test Accuracy (%)": round(final_row["test_accuracy"] * 100, 2),
            "ROC-AUC": final_row["roc_auc"],
            "Epsilon": final_row["privacy_epsilon"],
            "Comm Reduction (%)": final_row["communication_reduction_percent"]
        })

    df_ablation = pd.DataFrame(ablation_summary)
    os.makedirs("results", exist_ok=True)
    df_ablation.to_csv("results/ablation_study_summary.csv", index=False)
    print("\n=== Ablation Study Completed ===")
    print(df_ablation)
    return df_ablation
