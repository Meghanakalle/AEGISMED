import os
import pandas as pd
from evaluation.visualizer import generate_publication_plots
from evaluation.report_generator import generate_markdown_research_report

def main():
    csv_path = "results/experiment_metrics.csv"
    if not os.path.exists(csv_path):
        print("[-] Metrics file results/experiment_metrics.csv not found. Running fast validation experiment first...")
        from run_experiment import main as run_exp
        import sys
        sys.argv = ["run_experiment.py", "--config", "configs/fast_validation.yaml", "--mode", "fast"]
        run_exp()

    results_df = pd.read_csv(csv_path)
    
    # Dummy stats if standalone
    hospital_stats = [
        {"hospital_name": "Hospital_A_Metropolitan", "class_distribution": {0: 120, 1: 30, 2: 10}},
        {"hospital_name": "Hospital_B_Regional", "class_distribution": {0: 20, 1: 110, 2: 25}},
        {"hospital_name": "Hospital_C_University", "class_distribution": {0: 15, 1: 25, 2: 120}},
        {"hospital_name": "Hospital_D_Community", "class_distribution": {0: 60, 1: 60, 2: 40}}
    ]

    fig_dir = generate_publication_plots(results_df, hospital_stats, save_dir="figures")
    report_path = generate_markdown_research_report(results_df, save_dir="reports")

    print(f"[+] Publication plots generated in: {fig_dir}/")
    print(f"[+] Markdown research report generated at: {report_path}")

if __name__ == "__main__":
    main()
