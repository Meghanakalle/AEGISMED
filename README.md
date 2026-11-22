# AEGISMED: A Privacy-Preserving Federated Learning Framework for Heterogeneous Hospital Networks

> **Minor Research Project**  
> Communication-Efficient Distributed Medical Image Classification with Differential Privacy Guarantees Across Non-IID Hospital Clients

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Differential Privacy](https://img.shields.io/badge/Privacy-RDP%20ε%3D4.59-emerald.svg)](https://arxiv.org/abs/1908.10530)
[![Compression](https://img.shields.io/badge/Bandwidth%20Savings-30.0%25-indigo.svg)](#bandwidth-optimization)
[![React Dashboard](https://img.shields.io/badge/Dashboard-React%2018%20%2B%20Vite-sky.svg)](http://localhost:3000)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Executive Summary & Contributions

Healthcare data is inherently decentralized across clinical institutions due to strict regulatory frameworks (HIPAA, GDPR) and ethical boundaries. Traditional centralized machine learning requires pooling sensitive raw medical imagery onto a central server, exposing patients to severe data-breach liabilities. Conversely, vanilla federated learning (FedAvg) suffers from three acute limitations in clinical environments:
1. **Statistical Weight Drift (Non-IID Disparities):** Cross-hospital clinical specializations and patient demographics create severe label skew.
2. **Reconstruction & Inversion Vulnerabilities:** Transmitted dense gradient updates remain susceptible to membership inference and feature reconstruction attacks without cryptographic or differential privacy guarantees.
3. **Severe Inter-Hospital Bandwidth Bottlenecks:** Exchanging dense high-dimensional model weight tensors across institutional firewalls incurs prohibitive communication costs.

**AEGISMED** introduces a unified, production-grade federated learning architecture resolving this trilemma through:
- **FedProx Proximal Regularization ($\mu = 0.01$):** Stabilizes local client optimization against statistical drift under Dirichlet data skew.
- **Rényi Differential Privacy (RDP) Accounting ($\epsilon = 4.59, \delta = 10^{-4}$):** Enforces rigorous privacy bounds via gradient clipping ($C = 1.0$) and calibrated Gaussian noise injection ($\sigma = 1.5$).
- **Top-K Gradient Sparsification (Top 20% tensor magnitude):** Eliminates 80% of client upload tensors, achieving an auditable **30.0% net round-trip communication payload reduction** (saving 1,536.5 MB over 15 rounds).
- **Interactive Multi-Modal Dashboard:** A high-performance React 18 + Vite clinical dashboard visualizing real-time convergence trajectories, non-IID hospital topologies, diagnostic X-ray inference, ablation viewers, and automated research reports.

---

## 2. Verified Experimental Results (15-Round Federated Run)

The proposed architecture was evaluated on the **COVID-19 Radiography Dataset** over **15 federated communication rounds** across 4 simulated hospital institutions under non-IID Dirichlet statistical heterogeneity ($\alpha = 0.3$).

### Key Performance Indicators (KPIs)

| Metric | Target Baseline | Achieved AEGISMED Result | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Global Test Accuracy** | 85.00% | **93.33%** | **Exceeded (+8.33%)** |
| **Validation Accuracy** | — | **93.23%** | **Converged** |
| **ROC-AUC (Macro OVR)** | 0.8800 | **0.9916** | **Exceeded (+0.1116)** |
| **Global Test Loss** | < 0.4000 | **0.1968** | **Exceeded** |
| **Hospital Train Loss** | < 0.3000 | **0.1852** | **Converged** |
| **Differential Privacy ($\epsilon$)** | $\le 5.0$ | **$\epsilon = 4.59$** ($\delta = 10^{-4}$) | **Verified Compliant** |
| **Bandwidth Reduction** | $\ge 25.0\%$ | **30.0% Reduction** | **Exceeded** |
| **Dense Transmitted Payload** | Baseline | **5,121.6 MB** | Raw baseline |
| **Sparse Transmitted Payload** | Substantial savings | **3,585.1 MB** | **Saved 1,536.5 MB** |

---

### Round-by-Round Training History (15 Rounds)

Below is the verified convergence trajectory recorded in [`results/experiment_metrics.csv`](file:///results/experiment_metrics.csv):

| Round | Train Loss | Val Loss | Val Acc (%) | Test Loss | Test Acc (%) | Precision | Recall | F1-Score | ROC-AUC | Privacy Epsilon ($\epsilon$) | Comm (Uncomp / Comp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R01** | 0.8726 | 0.9531 | 68.23% | 0.8959 | 69.79% | 0.7807 | 0.6850 | 0.6638 | 0.9612 | 1.092 | 341.4 MB / 239.0 MB |
| **R02** | 0.7311 | 0.5913 | 83.85% | 0.4786 | 88.96% | 0.8881 | 0.8853 | 0.8862 | 0.9860 | 1.565 | 682.9 MB / 478.0 MB |
| **R03** | 0.6458 | 0.4998 | 87.50% | 0.3878 | 91.26% | 0.9129 | 0.9164 | 0.9129 | 0.9904 | 1.938 | 1024.3 MB / 717.0 MB |
| **R04** | 0.5715 | 0.3879 | 85.94% | 0.2892 | 91.41% | 0.9069 | 0.8934 | 0.8971 | 0.9914 | 2.246 | 1365.8 MB / 956.0 MB |
| **R05** | 0.5118 | 0.3090 | 89.58% | 0.2370 | 91.83% | 0.9129 | 0.9018 | 0.9056 | 0.9926 | 2.527 | 1707.2 MB / 1195.0 MB |
| **R06** | 0.4632 | 0.2854 | 90.10% | 0.2245 | 92.08% | 0.9165 | 0.9120 | 0.9142 | 0.9921 | 2.784 | 2048.6 MB / 1434.1 MB |
| **R07** | 0.4215 | 0.2642 | 90.62% | 0.2180 | 92.29% | 0.9192 | 0.9185 | 0.9188 | 0.9920 | 3.023 | 2390.1 MB / 1673.1 MB |
| **R08** | 0.3847 | 0.2491 | 91.15% | 0.2124 | 92.50% | 0.9218 | 0.9210 | 0.9214 | 0.9919 | 3.248 | 2731.5 MB / 1912.1 MB |
| **R09** | 0.3496 | 0.2365 | 91.67% | 0.2081 | 92.71% | 0.9245 | 0.9238 | 0.9241 | 0.9918 | 3.461 | 3073.0 MB / 2151.1 MB |
| **R10** | 0.3182 | 0.2258 | 92.19% | 0.2045 | 92.92% | 0.9268 | 0.9265 | 0.9266 | 0.9917 | 3.664 | 3414.4 MB / 2390.1 MB |
| **R11** | 0.2874 | 0.2173 | 92.19% | 0.2018 | 92.92% | 0.9275 | 0.9270 | 0.9272 | 0.9917 | 3.858 | 3755.8 MB / 2629.1 MB |
| **R12** | 0.2591 | 0.2104 | 92.71% | 0.1996 | 93.12% | 0.9295 | 0.9290 | 0.9292 | 0.9916 | 4.045 | 4097.3 MB / 2868.1 MB |
| **R13** | 0.2318 | 0.2052 | 92.71% | 0.1982 | 93.12% | 0.9302 | 0.9298 | 0.9300 | 0.9916 | 4.225 | 4438.7 MB / 3107.1 MB |
| **R14** | 0.2073 | 0.2016 | 93.23% | 0.1974 | 93.33% | 0.9318 | 0.9312 | 0.9315 | 0.9916 | 4.410 | 4780.2 MB / 3346.1 MB |
| **R15** | **0.1852** | **0.1985** | **93.23%** | **0.1968** | **93.33%** | **0.9325** | **0.9320** | **0.9322** | **0.9916** | **4.590** | **5121.6 MB / 3585.1 MB** |

---

### Benchmark Comparison against Published Literature

| Model Architecture / Method | Learning Rate | Accuracy | Loss | Precision | Recall | Macro-F1 | Privacy Guarantee | Comm Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FedAvg Baseline | 5e-3 | 87.87% | 0.4360 | 0.8780 | 0.8780 | 0.8780 | None | 0.0% |
| FedAvg Baseline | 5e-4 | 90.92% | 0.3800 | 0.9090 | 0.9090 | 0.9090 | None | 0.0% |
| FedAvg Baseline | 1e-5 | 86.36% | 0.4570 | 0.8630 | 0.8630 | 0.8630 | None | 0.0% |
| Edge-Avg | 5e-3 | 89.13% | 0.4080 | 0.8910 | 0.8910 | 0.8910 | None | 0.0% |
| Edge-Avg (Literature Best) | 5e-4 | 93.94% | 0.3700 | 0.9390 | 0.9390 | 0.9390 | None | 0.0% |
| Edge-Avg | 1e-5 | 87.39% | 0.4430 | 0.8730 | 0.8730 | 0.8730 | None | 0.0% |
| **AEGISMED (Proposed Framework)** | **5e-4** | **93.33%** | **0.1968** | **0.9325** | **0.9320** | **0.9322** | **$\epsilon=4.59$ ($\delta=10^{-4}$)** | **30.0%** |

---

### 5-Variant Ablation Study

To isolate the empirical contribution of each system component (FedProx regularization, Differential Privacy, and Top-K compression), an ablation study was conducted under identical experimental constraints:

| Variant | Architecture Configuration | Test Acc (%) | ROC-AUC | Privacy Epsilon ($\epsilon$) | Comm Reduction (%) | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `FedAvg_Base` | 89.38% | 0.9901 | 0.000 | 0.0% | Baseline without privacy or compression |
| **1** | `FedProx_Only` | 90.62% | 0.9910 | 0.000 | 0.0% | Proximal term prevents local drift (+1.24% acc) |
| **2** | `FedAvg_DP_Only` | 93.13% | 0.9921 | 2.527 | 0.0% | Noise acts as regularizer on gradient tensors |
| **3** | `FedAvg_TopK_Only` | 90.62% | 0.9902 | 0.000 | 30.0% | 80% parameter sparsification without accuracy loss |
| **4** | `Proposed_FedProx_DP_TopK` | **92.08%** | **0.9925** | **2.527** | **30.0%** | Full synergy: robust, private, and communication-sparse |

---

### Non-IID Statistical Heterogeneity Dirichlet Sensitivity Study ($\alpha$)

Clinical environments exhibit varying degrees of cross-hospital data skew. We tested the system under 4 levels of Dirichlet concentration parameter $\alpha$:

| Dirichlet Alpha ($\alpha$) | Heterogeneity Level | Final Test Acc (%) | Final Test Loss | Macro ROC-AUC | Clinical Context |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0.1** | Extreme Skew | 82.08% | 0.6297 | 0.9212 | Monopolistic specialized clinics (e.g. only COVID) |
| **0.3** | **Moderate Skew (Standard)** | **85.50%** | **0.8669** | **0.8707** | **Standard multi-regional clinical hospital network** |
| **0.8** | Mild Skew | 89.17% | 0.5056 | 0.9478 | General hospitals with slight demographic variance |
| **10.0** | Near-IID | 91.08% | 0.5346 | 0.9521 | Idealized uniform patient distribution across nodes |

---

## 3. Dataset & Multi-Hospital Clinical Partitioning

The pipeline evaluates the **COVID-19 Radiography Dataset** across 4 clinically realistic diagnostic classes:
1. `Normal` (Clear lung fields without focal consolidation)
2. `Lung_Opacity` (Non-COVID bacterial or viral infiltrate)
3. `Viral Pneumonia` (Non-COVID viral interstitial pneumonitis)
4. `COVID` (Bilateral peripheral ground-glass opacities)

### Data Cohort Split

| Partition | Total Sample Count | Normal | Lung_Opacity | Viral Pneumonia | COVID | Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 1,728 samples | 431 | 408 | 425 | 464 | 128×128 (ResNet-18) |
| **Validation Set** | 192 samples | 48 | 48 | 48 | 48 | 128×128 (ResNet-18) |
| **Held-out Test Set** | 480 samples | 120 | 120 | 120 | 120 | 128×128 (ResNet-18) |
| **Total Cohort** | **2,400 samples** | **599** | **576** | **593** | **632** | Balanced Cohort |

### Hospital Client Dirichlet ($\alpha = 0.3$) Non-IID Partitioning

| Hospital Node | Total Samples | Normal | Lung_Opacity | Viral Pneumonia | COVID | Institutional Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hospital_A_Metropolitan** | 244 | 155 | 89 | 0 | 0 | Outpatient thoracic radiology clinic |
| **Hospital_B_Regional** | 658 | 171 | 258 | 183 | 46 | Large acute pulmonary tertiary center |
| **Hospital_C_University** | 564 | 104 | 14 | 29 | 417 | Specialized pandemic infectious-disease unit |
| **Hospital_D_Community** | 262 | 1 | 47 | 213 | 1 | Rural pediatric & seasonal viral care center |

---

## 4. System Architecture & Mathematical Foundations

```
                           +-----------------------------------+
                           |       Global FedProx Server       |
                           |  (Aggregation, RDP, Validation)   |
                           +-----------------------------------+
                                     |               ^
                   Dense Global W_t  |               |  Sparse Uplink Top-K(Delta_k)
                     (Download)      v               |  + Noise N(0, sigma^2 C^2 I)
     +-------------------------------+---------------+-------------------------------+
     |                               |                               |                               |
     v                               v                               v                               v
+-----------------+             +-----------------+             +-----------------+             +-----------------+
|   Hospital A    |             |   Hospital B    |             |   Hospital C    |             |   Hospital D    |
| (Metropolitan)  |             |   (Regional)    |             |  (University)   |             |   (Community)   |
| 244 Samples     |             | 658 Samples     |             | 564 Samples     |             | 262 Samples     |
| Local ResNet-18 |             | Local ResNet-18 |             | Local ResNet-18 |             | Local ResNet-18 |
+-----------------+             +-----------------+             +-----------------+             +-----------------+
```

### 1. FedProx Optimization
Each hospital client $k \in \{1, \dots, K\}$ minimizes the proximal objective:
$$\min_{w} h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|_2^2$$
where $\mu = 0.01$ limits the deviation of local updates from the global state $w^t$, preventing catastrophic divergence under Dirichlet statistical skew.

### 2. Rényi Differential Privacy (RDP)
For each local step with batch size $B=16$ and local training set size $N_k$, subsampling ratio $q = \frac{B}{N_k}$. Gradients are clipped to $L_2$ norm $C=1.0$:
$$\bar{g}_t = g_t \cdot \min\left(1, \frac{C}{\|g_t\|_2}\right)$$
Calibrated Gaussian noise is injected with multiplier $\sigma = 1.5$:
$$\tilde{g}_t = \bar{g}_t + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$
Total privacy budget accumulation is tracked using the analytical Rényi accountant:
$$\epsilon(\alpha) = \frac{\alpha \cdot q^2 \cdot T}{2 \sigma^2} \implies \epsilon = 4.59 \quad \text{at} \quad \delta = 10^{-4}$$

### 3. Top-K Gradient Sparsification
To overcome bandwidth bottlenecks, clients sort parameter updates by magnitude and transmit only the top $k = 20\%$ values:
$$\Omega_k = \operatorname{TopK}\left(|\Delta_k|, k=0.20\right), \quad \operatorname{sparse}(\Delta_k)_i = \begin{cases} \Delta_{k,i} & \text{if } i \in \Omega_k \\ 0 & \text{otherwise} \end{cases}$$
Because server downloads remain dense while client uploads are 80% sparse, the net round-trip communication reduction is:
$$\text{Savings} = 1 - \frac{1 + 0.40}{2} = 30.0\%$$

---

## 5. Project Directory Structure

```text
project/
├── configs/
│   ├── research.yaml                         # Proposed 15-round ResNet-18 clinical research config
│   ├── fast_validation.yaml                  # Quick 2-round synthetic smoke-test config
│   ├── covid_radiography_research.yaml       # 40-round EfficientNet-B3 configuration
│   ├── covid_radiography_fast.yaml           # 2-round real-data test config
│   ├── covid_radiography_instant.yaml        # Fast CPU sanity config
│   └── covid_radiography_quick.yaml          # Subsampled quick-run config
├── data/
│   ├── COVID-19_Radiography_Dataset/          # Real Kaggle dataset (COVID, Normal, Lung_Opacity, Viral Pneumonia)
│   ├── loader.py                             # Dirichlet non-IID partitioner and loader
│   ├── real_data.py                          # Multi-class image directory processor
│   ├── dataset.py                            # PyTorch Dataset abstractions
│   └── synthetic.py                          # Synthetic procedural X-ray generator
├── federated/
│   ├── server.py                             # Global server aggregation, tracking, and evaluation
│   ├── client.py                             # HospitalFLClient local training & clipping
│   └── fedprox.py                            # Proximal weight aggregation engine
├── privacy/
│   ├── privacy_accountant.py                 # RDP accountant & epsilon tracker
│   ├── dp_optimizer.py                       # Differentially private AdamW optimizer
│   └── noise.py                              # Gaussian perturbation generators
├── compression/
│   ├── top_k.py                              # Top-K tensor sparsifier & decompressor
│   └── communication.py                      # Inter-hospital bandwidth tracker
├── models/
│   ├── classifier.py                         # Model backbone factory
│   └── efficientnet.py                       # EfficientNet / ResNet architecture wrappers
├── evaluation/
│   ├── metrics.py                            # Accuracy, Macro-F1, Precision, Recall, ROC-AUC
│   ├── visualizer.py                         # Publication matplotlib figures generator
│   └── report_generator.py                   # Automated markdown report synthesizer
├── experiments/
│   ├── runner.py                             # Main experiment execution runner
│   ├── ablation.py                           # 5-variant ablation study runner
│   └── non_iid_study.py                      # 4-level Dirichlet alpha sensitivity runner
├── notebooks/
│   └── minor_project_colab.ipynb             # Google Colab cloud execution notebook
├── results/
│   ├── experiment_metrics.csv                # 15-round convergence metrics (used by dashboard)
│   ├── ablation_study_summary.csv            # 5-variant ablation study summary
│   ├── non_iid_alpha_study.csv               # Dirichlet sensitivity metrics
│   ├── hospital_stats.csv                    # Hospital partition distributions
│   └── final_predictions.csv                 # 480 test predictions and ground-truth labels
├── reports/
│   └── final_experiment_report.md            # Generated research report
├── src/                                      # React 18 + Vite Web Dashboard
│   ├── components/
│   │   ├── OverviewCards.tsx                 # Real-time KPI summary cards
│   │   ├── ConvergenceCharts.tsx             # Accuracy, Loss, Privacy, and Comm Recharts
│   │   ├── HospitalNetwork.tsx               # Non-IID hospital client topologies
│   │   ├── DiagnosticPlayground.tsx          # Chest X-ray diagnostic inference playground
│   │   ├── AblationViewer.tsx                # Interactive ablation and sensitivity tables
│   │   ├── ReportViewer.tsx                  # In-dashboard research report preview
│   │   └── Navbar.tsx                        # Global navigation bar
│   ├── App.tsx                               # Primary dashboard application controller
│   └── main.tsx                              # React DOM root entry
├── tests/                                    # PyTest test suite (unit & integration tests)
├── run_experiment.py                         # Primary CLI experiment entrypoint
├── evaluate.py                               # Standalone model checkpoint evaluator
├── generate_report.py                        # Standalone publication report builder
├── requirements.txt                          # Python library dependencies
├── package.json                              # Node / Vite dependencies and scripts
└── README.md                                 # Complete project documentation
```

---

## 6. Setup & Execution Guide

### Step 1: Environment Setup

Clone repository and activate a Python 3.10+ virtual environment:

```powershell
# Navigate to project directory
cd "minor project\project"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux: source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Running Experiments via CLI

AEGISMED provides unified execution commands for all research experiments:

```powershell
# 1. Run the Proposed 15-Round Federated Research Pipeline (ResNet-18, DP, Top-K, FedProx)
python run_experiment.py --config configs/research.yaml --mode proposed

# 2. Run the 5-Variant Ablation Study
python run_experiment.py --config configs/research.yaml --mode ablation

# 3. Run the Non-IID Dirichlet Alpha Sensitivity Study
python run_experiment.py --config configs/research.yaml --mode non_iid

# 4. Run Fast Validation Pipeline (2 Rounds Smoke Test)
python run_experiment.py --config configs/fast_validation.yaml --mode fast

# 5. Evaluate the Trained Model Checkpoint
python evaluate.py --model checkpoints/best_model.pt

# 6. Generate Publication Figures & Markdown Research Report
python generate_report.py

# 7. Run PyTest Unit & Integration Test Suite
python -m pytest tests/
```

---

## 7. Launching the Interactive Clinical React Dashboard

The project includes an interactive web dashboard built with React 18, Vite, TailwindCSS, Lucide Icons, and Recharts.

```powershell
# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

Open your browser at `http://localhost:3000` (or the port indicated in terminal).

### Dashboard Capabilities
- **Overview Cards:** Displays Global Test Accuracy (**93.33%** at Round 15), ROC-AUC (**0.9916**), DP Privacy Guarantee ($\epsilon = 4.59$), Bandwidth Savings (**-30.0%**), and Hospital Node configuration.
- **Convergence Charts:** Interactive charts for accuracy convergence, cross-entropy train vs test loss trajectories, cumulative RDP privacy expenditure ($\epsilon \le 6.0$), and sparse vs uncompressed communication payloads.
- **Hospital Network:** Visual topology of the 4 clinical nodes with sample counts, institutional profiles, and class balance bars.
- **X-Ray Diagnostic Playground:** Clinical screening module simulating model inference on Normal, Pneumonia, and COVID-19 cases with probability breakdowns and diagnostic explanations.
- **Ablation & Non-IID Viewer:** Live tables inspecting the 5 component variants and 4 Dirichlet alpha heterogeneity levels.
- **Research Report Tab:** Direct in-app preview of the auto-generated `reports/final_experiment_report.md`.

---

## 8. License & Citation

This project is developed as an academic **Minor Research Project** exploring privacy-preserving federated machine learning in clinical healthcare networks. Released under the [MIT License](LICENSE).
