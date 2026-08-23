# Major Research Project Report
## Communication-Efficient Federated Learning with Differential Privacy Guarantees for Distributed Image Classification Across Heterogeneous Hospital Networks

### Executive Summary
This major project develops a privacy-preserving, communication-efficient federated learning system tailored for multi-hospital clinical networks. By combining **FedProx** parameter aggregation with **Renyi Differential Privacy (RDP)** ($\epsilon=2.53, \delta=10^{-4}$) and **Top-K Update Sparsification** (top 20% magnitude tensor selection), our architecture achieves **91.83% Global Test Accuracy** and **0.9926 ROC-AUC** on Resnet18 while reducing inter-hospital bandwidth overhead by **30.0%**.

---

### Core Experimental Findings

| Metric | Target Baseline | Achieved Result | Evaluation Status |
| :--- | :--- | :--- | :--- |
| **Global Test Accuracy** | 85.0% | **91.83%** | Exceeded (+6.83%) |
| **ROC-AUC (Macro OVR)** | 0.880 | **0.9926** | Exceeded |
| **Differential Privacy ($\epsilon$)** | $\le 3.5$ | **$\epsilon = 2.53$** ($\delta=10^{-4}$) | Verified Compliant |
| **Bandwidth Overhead Reduction** | $\ge 60.0\%$ | **30.0% Reduction** | Failed Target |
| **Uncompressed Transmitted Data** | ~3,840 MB | **1707.2 MB** | Baseline |
| **Compressed Transmitted Data** | < 1,500 MB | **1195.5 MB** | Compressed Payload |

---

### Round-by-Round Federated Training History
Below is the full history of the federated learning training run across the 5 rounds:

| Round | Train Loss | Val Loss | Val Acc | Test Loss | Test Acc | Precision | Recall | F1-Score | ROC-AUC | Privacy Epsilon ($\epsilon$) | Comm Vol (Uncomp/Comp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.8726 | 0.9531 | 68.23% | 0.8959 | 69.79% | 0.7807 | 0.6850 | 0.6638 | 0.9612 | 1.092 ($\delta=10^{-4}$) | 341.4 MB / 239.1 MB |
| **2** | 0.7311 | 0.5913 | 83.85% | 0.4786 | 88.96% | 0.8881 | 0.8853 | 0.8862 | 0.9860 | 1.565 ($\delta=10^{-4}$) | 682.9 MB / 478.2 MB |
| **3** | 0.6458 | 0.4998 | 87.50% | 0.3878 | 91.26% | 0.9129 | 0.9164 | 0.9129 | 0.9904 | 1.938 ($\delta=10^{-4}$) | 1024.3 MB / 717.3 MB |
| **4** | 0.5715 | 0.3879 | 85.94% | 0.2892 | 91.41% | 0.9069 | 0.8934 | 0.8971 | 0.9914 | 2.246 ($\delta=10^{-4}$) | 1365.8 MB / 956.4 MB |
| **5** | 0.5118 | 0.3090 | 89.58% | 0.2370 | 91.83% | 0.9129 | 0.9018 | 0.9056 | 0.9926 | 2.527 ($\delta=10^{-4}$) | 1707.2 MB / 1195.5 MB |

---

### Dataset Details
- **Dataset Name:** COVID-19 Radiography Dataset
- **Total Split Sizes:**
  - **Training Set:** 1728 samples
  - **Validation Set:** 192 samples
  - **Held-out Test Set:** 480 samples

#### Hospital Client Sample & Class Distributions
| Hospital | Total Samples | Normal | Lung_Opacity | Viral Pneumonia | COVID |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hospital_A_Metropolitan | 244 | 155 | 89 | 0 | 0 |
| Hospital_B_Regional | 658 | 171 | 258 | 183 | 46 |
| Hospital_C_University | 564 | 104 | 14 | 29 | 417 |
| Hospital_D_Community | 262 | 1 | 47 | 213 | 1 |

---

### Model & Training Configurations
- **Model Backbone:** Pretrained **Resnet18**
- **Image Size:** 128x128
- **Optimizer:** AdamW with Weight Decay (`1e-4`)
- **Initial Learning Rate:** 0.0005
- **FedProx Regularization ($\mu$):** 0.01
- **Top-K Update Sparsification Keep Ratio:** 0.2 (i.e., top 20% updates kept)
- **Differential Privacy Config:**
  - **Clipping Norm ($C$):** 1.0
  - **Noise Multiplier ($\sigma$):** 1.5
  - **Client Sampling Rate ($q$):** 0.0656
  - **Total Accountant Steps:** 160 steps (5 rounds x 2 local epochs x 16 local batches)
  - **Target Delta ($\delta$):** 0.0001
  - **Final Privacy Epsilon ($\epsilon$):** 2.5270

---

### Per-Class Performance Metrics
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.8824 | 0.8824 | 0.8824 |
| **Lung_Opacity** | 0.8600 | 0.9485 | 0.9021 |
| **Viral Pneumonia** | 0.9683 | 0.9683 | 0.9683 |
| **COVID** | 0.9412 | 0.8081 | 0.8696 |

#### Confusion Matrix
| Actual \ Predicted | Normal | Lung_Opacity | Viral Pneumonia | COVID |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 105 | 8 | 4 | 2 |
| **Lung_Opacity** | 4 | 129 | 0 | 3 |
| **Viral Pneumonia** | 3 | 1 | 122 | 0 |
| **COVID** | 7 | 12 | 0 | 80 |

---

### Ablation Study Results
| Variant                  |   Test Accuracy (%) |   ROC-AUC |   Epsilon |   Comm Reduction (%) |
|:-------------------------|--------------------:|----------:|----------:|---------------------:|
| FedAvg_Base              |               91.04 |    0.9911 |     2.527 |                   30 |
| FedProx_Only             |               91.04 |    0.9919 |     2.527 |                   30 |
| FedAvg_DP_Only           |               91.04 |    0.9936 |     2.527 |                   30 |
| FedAvg_TopK_Only         |               89.80 |    0.9890 |     2.527 |                   30 |
| Proposed_FedProx_DP_TopK |               91.04 |    0.9926 |     2.527 |                   30 |

---

### Non-IID Dirichlet Alpha Sensitivity Study
|   Dirichlet Alpha (alpha) | Heterogeneity Level   |   Final Test Acc (%) |   Final Loss |   ROC-AUC |
|--------------------------:|:----------------------|---------------------:|-------------:|----------:|
|                       0.1 | Extreme               |                82.08 |       0.6297 |    0.9212 |
|                       0.3 | Moderate              |                85.50 |       0.8669 |    0.8707 |
|                       0.8 | Mild                  |                89.17 |       0.5056 |    0.9478 |
|                      10   | Near-IID              |                91.08 |       0.5346 |    0.9521 |

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
| **Proposed Framework** | **0.0005** | **91.83%** | **0.2370** | **0.9129** | **0.9018** | **0.9056** |

---

### Limitations & Analysis
1. **CPU Execution Limitations:** Due to the lack of hardware GPU/CUDA acceleration on the client's side, training takes significantly longer. Subsampling the dataset (e.g. 600 per class) represents a practical trade-off to allow scientific execution of the entire ablation/sensitivity matrix.
2. **Privacy vs. Utility Tradeoff:** Stronger differential privacy guarantees (higher noise multiplier, lower epsilon) naturally degrade the accuracy compared to non-private baselines like Edge-Avg.
3. **Statistical Skew (Non-IID):** Extremely small Dirichlet alpha ($\alpha = 0.1$) causes extreme client data imbalance which impacts convergence stability, requiring FedProx proximal term adjustment.

---
*Report automatically generated by Major Project Evaluation Engine.*
