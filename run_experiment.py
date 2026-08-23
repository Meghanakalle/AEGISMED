import argparse
from utils.config import load_config
from experiments.runner import run_federated_experiment
from experiments.ablation import run_ablation_study
from experiments.non_iid_study import run_non_iid_alpha_study

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Medical Research Experiment Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (defaults depend on --mode)")
    parser.add_argument("--mode", type=str, default="proposed", choices=["proposed", "fast", "ablation", "non_iid"], help="Execution mode")
    
    args = parser.parse_args()

    if args.mode == "fast":
        config_path = args.config or "configs/fast_validation.yaml"
        print("[+] Executing Fast Validation FL Pipeline...")
        run_federated_experiment(config_path=config_path, mode="fast")
    elif args.mode == "proposed":
        config_path = args.config or "configs/research.yaml"
        try:
            num_rounds = load_config(config_path)["federated"].get("num_rounds", "?")
        except Exception:
            num_rounds = "?"
        print(f"[+] Executing Federated Learning Pipeline ({num_rounds} Rounds)...")
        run_federated_experiment(config_path=config_path, mode="proposed")
    elif args.mode == "ablation":
        config_path = args.config or "configs/research.yaml"
        print("[+] Executing 5-Variant Ablation Study...")
        run_ablation_study(config_path=config_path)
    elif args.mode == "non_iid":
        config_path = args.config or "configs/research.yaml"
        print("[+] Executing Non-IID Dirichlet Sensitivity Study...")
        run_non_iid_alpha_study(config_path=config_path)

if __name__ == "__main__":
    main()
