# Communication-Efficient Federated Learning with Differential Privacy Guarantees

This project contains both:
1. **Python FL Framework**: Federated learning training loops, EfficientNet-B3 backbone, FedProx, Differential Privacy (RDP accountant), Top-K compression, evaluation scripts, and pytest suite.
2. **React Web Dashboard**: High-level visual dashboard with FL metrics, convergence graphs, hospital network, diagnostic playground, ablation study viewer, and research report preview.

---

## 🚀 How to Run in VS Code

### Step 1: Export / Download Project
- Click **Settings** (gear icon) in the top right of the AI Studio interface.
- Click **Export Project** or **Download ZIP**.
- Extract the ZIP archive onto your local machine.

---

### Step 2: Open in VS Code
1. Open VS Code.
2. Go to `File` -> `Open Folder...` (or `Ctrl+K Ctrl+O` / `Cmd+O`) and select the extracted project folder.

---

### Step 3: Run the Python FL Experiments

#### Option A: Using VS Code Debugger (F5)
1. Open the **Run and Debug** tab in VS Code sidebar (`Ctrl+Shift+D` or `Cmd+Shift+D`).
2. Select one of the pre-configured configurations from the dropdown:
   - **Python: Fast Validation** (runs quick 1-round test loop)
   - **Python: Full FL Research Experiment** (runs 40-round training loop)
   - **Python: Ablation Study** (runs 5-variant ablation study)
   - **Python: Evaluate Checkpoint**
3. Press **F5** to start execution.

#### Option B: Using Terminal
Open VS Code Integrated Terminal (`Ctrl+\`` or `Cmd+\``) and run:

```bash
# 1. Create and activate a Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run fast integration test
python run_experiment.py --config configs/fast_validation.yaml --mode fast

# 4. Run full 40-round federated training experiment
python run_experiment.py --config configs/research.yaml --mode proposed

# 5. Run 5-variant ablation study
python run_experiment.py --config configs/research.yaml --mode ablation

# 6. Run PyTest unit tests
python -m pytest tests/
```

---

### Step 4: Run the Interactive React Dashboard

In VS Code Terminal:

```bash
# 1. Install Node modules
npm install

# 2. Launch Vite development server
npm run dev
```

Then open your browser at `http://localhost:3000` (or the local port printed in the terminal).

---

### Using Your Own (Real) COVID Chest X-ray Dataset

By default the pipeline generates synthetic X-ray-like images so it runs out of the box with
no data download. To train on a real dataset instead, set `dataset.data_dir` in a config file.

**If you downloaded the Kaggle "COVID-19 Radiography Database"** (the one with `COVID/`,
`Lung_Opacity/`, `Normal/`, `Viral Pneumonia/` folders, each containing an `images/`
subfolder and a `masks/` subfolder) — this is supported directly. Two ready-made configs
are included:

- `configs/covid_radiography_fast.yaml` — 2-round smoke test, small images, use this first.
- `configs/covid_radiography_research.yaml` — full 40-round EfficientNet-B3 run.

Open either file and edit this one line to point at your extracted dataset folder:

```yaml
dataset:
  data_dir: "/path/to/COVID-19_Radiography_Dataset"   # <- edit this
```

Then run:

```bash
python run_experiment.py --config configs/covid_radiography_fast.yaml --mode fast
python run_experiment.py --config configs/covid_radiography_research.yaml --mode proposed
```

The loader automatically looks inside each class folder for an `images/` subfolder
(ignoring `masks/`); it also works with a flatter `data_dir/ClassName/*.png` layout if
your dataset doesn't have the nested `images/`/`masks/` structure.

**For any other real dataset**, just arrange it as one folder per class matching
`dataset.classes` in the config (case/space/underscore-insensitive), e.g.:

```
/path/to/your/dataset/
├── Normal/
│   ├── img001.png
│   └── ...
├── Pneumonia/
│   └── ...
└── COVID-19/
    └── ...
```

and set `data_dir` accordingly, adjusting `dataset.classes`/`dataset.num_classes` and
`model.num_classes` to match how many classes you actually have. Leave `data_dir: null`
to keep using synthetic data. Real images are larger than the 64px synthetic samples in
`fast_validation.yaml`, so bump `image_size` back up (e.g. 224) for real runs.

---

## 🖥️ About the React Dashboard

The dashboard under `src/` is a **self-contained visual demo** — it does not read
`results/experiment_metrics.csv` or `reports/final_experiment_report.md` produced by the
Python pipeline. Its charts are generated from a simulated 40-round curve baked into
`src/App.tsx`, and the "Diagnostic Playground" classifies images using canned
preset logic in-browser, not your trained PyTorch model. It needs no API key and no
backend running — `npm install && npm run dev` is enough to view it standalone.

If you want the dashboard to show your *actual* training results instead of the
simulated curve, that requires a small integration step (e.g. a script to convert
`results/experiment_metrics.csv` into the `RoundRecord[]` shape `App.tsx` expects, or a
tiny local API endpoint serving it) — ask if you'd like this wired up.

---

## 📁 Project Directory Structure

```
├── .vscode/             # VS Code launch.json & tasks.json configurations
├── configs/             # Yaml experiment configs (fast_validation.yaml, research.yaml)
├── models/              # EfficientNet-B3 backbone classifier
├── federated/           # Federated Server & Local Client training logic
├── privacy/             # Differential privacy noise & RDP accountant
├── compression/         # Top-K sparsification & communication metrics
├── evaluation/          # Metrics calculation (Accuracy, ROC-AUC, Loss)
├── tests/               # PyTest unit tests suite
├── notebooks/           # Jupyter / Google Colab notebook
├── src/                 # React UI components & dashboard
├── run_experiment.py    # Main CLI entrypoint for FL experiments
├── evaluate.py          # Standalone checkpoint evaluator
├── generate_report.py   # Final markdown report generator
├── requirements.txt     # Python package requirements
└── package.json         # Node.js dependencies
```
