import os
import copy
import pandas as pd
from utils.config import load_config
from experiments.runner import run_federated_experiment

def run_non_iid_alpha_study(config_path: str = "configs/research.yaml"):
    """
    Evaluates sensitivity to statistical heterogeneity across Dirichlet alpha parameters:
    alpha = [0.1 (extreme non-iid), 0.3 (moderate non-iid), 0.8 (mild non-iid), 10.0 (near-iid)]
    """
    alphas = [0.1, 0.3, 0.8, 10.0]
    base_cfg = load_config(config_path)
    alpha_results = []

    for alpha in alphas:
        cfg = copy.deepcopy(base_cfg)  # deepcopy: base_cfg's nested dicts must not be shared/mutated across variants
        cfg["dataset"]["dirichlet_alpha"] = alpha
        cfg["federated"]["num_rounds"] = 5  # Quick sensitivity evaluation

        print(f"\n--- Non-IID alpha: {alpha} ---")
        df = run_federated_experiment(config_path, mode="non_iid", config_override=cfg)
        final_row = df.iloc[-1]

        alpha_results.append({
            "Dirichlet Alpha (alpha)": alpha,
            "Heterogeneity Level": "Extreme" if alpha <= 0.1 else ("Moderate" if alpha <= 0.3 else ("Mild" if alpha <= 0.8 else "Near-IID")),
            "Final Test Acc (%)": round(final_row["test_accuracy"] * 100, 2),
            "Final Loss": final_row["test_loss"],
            "Macro Precision": final_row["precision"],
            "Macro Recall": final_row["recall"],
            "Macro F1": final_row["f1_score"],
            "ROC-AUC": final_row["roc_auc"]
        })

    df_alpha = pd.DataFrame(alpha_results)
    os.makedirs("results", exist_ok=True)
    df_alpha.to_csv("results/non_iid_alpha_study.csv", index=False)
    print("\n=== Non-IID Dirichlet Alpha Study Completed ===")
    print(df_alpha)
    return df_alpha
