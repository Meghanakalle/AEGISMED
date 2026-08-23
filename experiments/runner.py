import os
import torch
import pandas as pd
from utils.seed import set_seed
from utils.config import load_config
from data.loader import get_hospital_dataloaders
from federated.server import FederatedServer
from evaluation.visualizer import generate_publication_plots
from evaluation.report_generator import generate_markdown_research_report

def run_federated_experiment(config_path: str = "configs/research.yaml", mode: str = "proposed", config_override: dict = None):
    # config_override lets callers (e.g. ablation/non-IID studies) pass an
    # already-modified config dict so their per-variant settings actually take
    # effect, instead of this always reloading the unmodified file from disk.
    config = config_override if config_override is not None else load_config(config_path)
    set_seed(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build hospital data loaders
    hospital_loaders, global_val_loader, global_test_loader, hospital_stats = get_hospital_dataloaders(config, return_val=True)

    # Persist real per-hospital sample counts / class distributions so the
    # dashboard's Hospital Network tab can show genuine data instead of
    # hardcoded placeholder numbers.
    class_names = config["dataset"].get("classes", [])
    hospital_rows = []
    for h in hospital_stats:
        row = {
            "hospital_id": h["hospital_id"],
            "hospital_name": h["hospital_name"],
            "sample_count": h["sample_count"],
        }
        for cls_idx, count in h["class_distribution"].items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            row[cls_name] = count
        hospital_rows.append(row)
    results_dir = config["paths"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(hospital_rows).to_csv(os.path.join(results_dir, "hospital_stats.csv"), index=False)

    # Initialize Federated Server
    server = FederatedServer(
        config=config,
        hospital_loaders=hospital_loaders,
        global_test_loader=global_test_loader,
        device=device,
        global_val_loader=global_val_loader
    )

    rounds = config["federated"].get("num_rounds", 40)
    df_results = server.run_training_loop(rounds=rounds)

    # Generate Figures
    fig_dir = config["paths"].get("figures_dir", "figures")
    class_names = config["dataset"].get("classes", [])
    dirichlet_alpha = config["dataset"].get("dirichlet_alpha", 0.3)
    generate_publication_plots(
        df_results,
        hospital_stats,
        save_dir=fig_dir,
        test_metrics=server.final_test_metrics,
        class_names=class_names,
        alpha=dirichlet_alpha
    )

    # Generate Markdown Report
    rep_dir = config["paths"].get("reports_dir", "reports")
    train_size = sum(len(loader.dataset) for loader in hospital_loaders)
    val_size = len(global_val_loader.dataset) if global_val_loader is not None else 0
    test_size = len(global_test_loader.dataset)
    dataset_sizes = {"train": train_size, "val": val_size, "test": test_size}

    report_path = generate_markdown_research_report(
        df_results,
        config=config,
        save_dir=rep_dir,
        test_metrics=server.final_test_metrics,
        hospital_stats=hospital_stats,
        dataset_sizes=dataset_sizes
    )

    # Save final raw predictions and labels for auditable metrics
    if server.final_test_metrics is not None:
        preds = server.final_test_metrics["predictions"]
        targets = server.final_test_metrics["targets"]
        df_preds = pd.DataFrame({
            "prediction": preds,
            "label": targets
        })
        df_preds.to_csv(os.path.join(results_dir, "final_predictions.csv"), index=False)

    print(f"\n[+] Experiment Pipeline Completed Successfully!")
    print(f"    - Results saved to: {os.path.join(config['paths']['results_dir'], 'experiment_metrics.csv')}")
    print(f"    - Raw predictions saved to: {os.path.join(config['paths']['results_dir'], 'final_predictions.csv')}")
    print(f"    - Publication figures saved to: {fig_dir}/")
    print(f"    - Report generated at: {report_path}")

    return df_results
