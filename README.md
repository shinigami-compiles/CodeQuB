You’re right to think about this now – environment + versions will make or break this project.

First I’ll be very clear about one important limitation, then I’ll give you:

1. `requirements.txt`
2. `README.md` (full markdown code)

---

## ⚠️ Important: Python 3.13 vs TensorFlow

As of late 2025, **TensorFlow does *not* yet provide official wheels for Python 3.13** – it’s only tested/supported on Python **3.9–3.12**. ([TensorFlow][1])

That means:

> If you insist on *system* Python 3.13, pip install of `tensorflow` / `tensorflow-cpu` will fail (or require messy source builds, which are not officially supported on Windows).

✅ **Recommended setup for your project:**

* **OS:** Windows 8.1 (OK, TensorFlow still supports Windows 7+ for CPU). ([TensorFlow][1])
* **Python inside a virtualenv:** **Python 3.11** (or 3.10 / 3.12)
* **TensorFlow:** 2.16.x CPU-only (supports Python 3.12 and below). ([blog.tensorflow.org][2])

So in the README I’ll explicitly tell: “Use Python 3.11 venv for this project, even if your system Python is 3.13.”

---

## ✅ `requirements.txt`

Below is a `requirements.txt` that’s appropriate for:

* Python **3.11** on **Windows 8.1**
* Your current codebase (`generate_dataset.py`, `train_model.py`, `app.py`, explainability, recommendations, report generation)

```txt
# ===============================
# Core Deep Learning / ML stack
# ===============================

# CPU-only TensorFlow (recommended on Windows 8.1)
# Tested for Python 3.9–3.12, use a virtualenv with Python 3.11.
tensorflow-cpu==2.16.2

# Numeric + data handling
numpy>=1.26.0,<3.0
pandas==2.2.3

# Classical ML utilities (train/val/test split, metrics, scaler, etc.)
scikit-learn==1.6.0
joblib>=1.3.0

# ===============================
# Web API / Flask app
# ===============================

Flask==3.0.3

# Optional but recommended for better JSON handling & Werkzeug compatibility
itsdangerous>=2.1.0
Jinja2>=3.1.0
Werkzeug>=3.0.0

# ===============================
# Misc utilities
# ===============================

# For type hints if you want to expand / refactor later.
typing_extensions>=4.8.0
```

> 🔑 **How to handle Python 3.13:**
>
> * Keep Python 3.13 installed globally if you want.
> * Create a **separate virtualenv using Python 3.11** just for this project, and use the above `requirements.txt` inside it.

I explain exactly how in the README below.

---

## ✅ `README.md` (project documentation)

Here’s a complete `README.md` you can drop in your repo root.

````markdown
# ANN-Based Code Quality & Bug Risk Prediction System  
### Using Static & Process Metrics + Multi-Task Neural Networks

---

## 1. Overview

**CodePulse AI** is an ANN-powered system that predicts:

1. **Code Quality Score** (continuous, `0–100`)  
2. **Bug Risk Class** (6 classes: `0–5`)

for a **single code file / module**, based on **22 numeric metrics** extracted from:

- Static analysis (LOC, complexity, nesting, lint)
- Testing (coverage, test files, test failures)
- Process metrics (code churn, contributors, file age, fan-in)
- Bug/defect history (bug counts, density, recent bugs, bug trend)

No raw source code is passed into the model – only **tabular numeric features**.

The system includes:

- **Synthetic dataset generation** with heuristic label formulas  
- **Multi-task ANN** in TensorFlow/Keras  
- Prediction of **quality (regression)** and **bug risk (classification)**  
- **Explainability** (feature contributions)  
- **Recommendation engine** (how to improve quality & reduce risk)  
- **Flask UI** with a **dark cyberpunk theme** (manual input + CSV upload)  
- **HTML report generation** for each analysis  

---

## 2. Environment & Requirements

### 2.1. Python & OS

> ⚠️ **Important about Python 3.13**

As of now, TensorFlow officially supports **Python 3.9–3.12**, not 3.13.  
To avoid installation issues, this project is designed to run inside a **virtual environment with Python 3.11** (or 3.10 / 3.12). :contentReference[oaicite:3]{index=3}  

You can keep Python 3.13 installed globally, but **create a dedicated venv** with Python 3.11.

- **Recommended:**
  - OS: Windows 8.1 (64-bit)
  - Python: **3.11.x** (for this project’s virtualenv)
  - CPU-only TensorFlow (no GPU needed)

### 2.2. Installing dependencies

1. Create and activate a virtual environment (example for Python 3.11):

   ```bash
   # Ensure python3.11 is installed
   python3.11 -m venv venv
   # On Windows:
   venv\Scripts\activate
````

2. Upgrade pip:

   ```bash
   python -m pip install --upgrade pip
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Contents of `requirements.txt` are:

```text
tensorflow-cpu==2.16.2
numpy>=1.26.0,<3.0
pandas==2.2.3
scikit-learn==1.6.0
joblib>=1.3.0
Flask==3.0.3
itsdangerous>=2.1.0
Jinja2>=3.1.0
Werkzeug>=3.0.0
typing_extensions>=4.8.0
```

---

## 3. Project Structure

A suggested layout for this project:

```text
project_root/
│
├── generate_dataset.py        # Synthetic data generator (15k rows, balanced bug classes)
├── train_model.py             # Builds & trains multi-task ANN, saves model/scaler/metadata
├── explainability.py          # Gradient × input feature contributions
├── recommendations.py         # Rule-based recommendations based on metrics & predictions
├── report_generator.py        # Builds full HTML report for a single analysis
├── app.py                     # Flask web app (UI + APIs)
│
├── models/
│   ├── multitask_ann_model/   # Saved Keras model (from train_model.py)
│   ├── feature_scaler.pkl     # StandardScaler for 22 features
│   └── metadata.json          # Feature metadata, label mapping, stats, descriptions
│
├── templates/
│   ├── base.html              # Base layout (nav, shell)
│   ├── landing.html           # Landing page (title + description + Start button)
│   ├── input.html             # Manual form + single-row CSV upload
│   ├── result.html            # High-level result (quality score + bug risk class)
│   ├── breakdown.html         # Detailed breakdown, feature impacts, recommendations
│   └── about.html             # Docs, CSV structure, feature list, template CSV download
│
├── static/
│   ├── css/
│   │   └── style.css          # Dark cyberpunk theme with neon cyan accents
│   ├── js/
│   │   └── main.js            # Mode toggle, report download logic
│   └── files/
│       └── template_metrics.csv   # CSV template for single-row upload
│
├── data/
│   └── code_quality_bug_risk_dataset.csv  # (generated by generate_dataset.py)
│
├── requirements.txt
└── README.md
```

---

## 4. Pipeline: End-to-End Flow

### 4.1. Step 1 – Generate synthetic dataset

Script: **`generate_dataset.py`**

* Randomly generates **15,000 rows** of realistic feature values for the 22 metrics.

* Uses heuristic formulas to compute:

  * `code_quality_score` (0–100)
  * `bug_risk_score` (0–100)
  * `bug_risk_class` (0–5), via thresholds:

    ```text
    [0, 15)   -> 0 (Very Low)
    [15, 30)  -> 1 (Low)
    [30, 45)  -> 2 (Slightly Risky)
    [45, 60)  -> 3 (Medium)
    [60, 80)  -> 4 (High)
    [80, 100] -> 5 (Critical)
    ```

* Ensures **class balance** across the 6 bug risk classes as much as possible.

* Saves: `data/code_quality_bug_risk_dataset.csv`

Run:

```bash
python generate_dataset.py
```

---

### 4.2. Step 2 – Train the multi-task ANN

Script: **`train_model.py`**

* Loads `data/code_quality_bug_risk_dataset.csv`

* Splits into **train/validation/test** (e.g., 70/15/15)

* Standardizes features using `StandardScaler`

* Builds a **multi-task ANN** in Keras:

  **Shared trunk:**

  * Dense(128, activation="relu")
  * Dense(64, activation="relu")
  * Dropout(0.3)
  * Dense(32, activation="relu")

  **Head 1 (Quality regression):**

  * Dense(1, activation="linear")

  **Head 2 (Bug risk classification):**

  * Dense(6, activation="softmax")

* Uses **combined loss**:

  ```python
  loss = 0.7 * MSE(quality) + 0.3 * CategoricalCrossentropy(bug_class)
  ```

* Tracks metrics:

  * Regression: MAE, MSE
  * Classification: Accuracy

* Uses **EarlyStopping** on validation loss.

* Evaluates on test set:

  * Prints MAE, RMSE for quality
  * Prints accuracy and confusion matrix for bug risk class

* Saves:

  * `models/multitask_ann_model/`
  * `models/feature_scaler.pkl`
  * `models/metadata.json` (feature names, descriptions, stats, label mapping, num classes)

Run:

```bash
python train_model.py
```

---

### 4.3. Step 3 – Run the Flask app

Script: **`app.py`**

* Loads:

  * Keras model from `models/multitask_ann_model`
  * Scaler from `models/feature_scaler.pkl`
  * Metadata from `models/metadata.json`
* Uses helper modules:

  * `explainability.py`
  * `recommendations.py`
  * `report_generator.py`

Run:

```bash
python app.py
```

Open in browser:

```text
http://localhost:5000/
```

---

## 5. UI Features

### 5.1. Landing page (`/` → `landing.html`)

* Dark **cyberpunk** theme (black background, neon cyan glow)
* Centered **title** and **description**
* **"Start Code Analysis"** button → goes to `/input`

### 5.2. Input page (`/input` → `input.html`)

Two modes on the same page:

1. **Manual Input Mode**

   * Guided form grouped into:

     * Structural & Complexity
     * Style & Readability
     * Testing & Coverage
     * Stability & Ownership
     * Bug & Instability
   * You manually enter the 22 metrics.

2. **Single-Row CSV Upload Mode**

   * Upload a CSV with **exactly one row** representing a single code file.
   * Header must match the feature names.
   * Uses `/predict-csv-single` route:

     * Reads first row → `features`
     * Performs prediction
     * Redirects to result page

JS logic (`static/js/main.js`) handles toggling between modes.

### 5.3. Result page (`/result/<analysis_id>` → `result.html`)

Shows:

* **Code Quality Score** out of 100
* **Bug Risk Class** 0–5 + descriptive label (Very Low … Critical)
* **Probability distribution** over all 6 classes
* **Top 3 features** influencing:

  * Quality
  * Bug Risk

Also includes:

* Button: **“View Detailed Breakdown”** → `/breakdown/<analysis_id>`
* Button: Analyze another file → `/input`
* Button: Back to home → `/`

### 5.4. Breakdown page (`/breakdown/<analysis_id>` → `breakdown.html`)

Shows:

* Summary of:

  * Quality score
  * Bug risk class + label

* Detailed **feature impact tables**:

  * For quality: each feature + value + contribution (↑ improves / ↓ reduces)
  * For bug risk: each feature + value + contribution (↑ increases / ↓ reduces)

* **Recommendations**:

  * How to improve code quality (e.g., reduce complexity, increase tests)
  * How to reduce bug risk (e.g., lower churn, reduce test failures)

* Full **input metrics table**:

  * Feature name
  * Value
  * Human-readable description (from `metadata.json`)

* **“Download Full HTML Report”** button:

  * JS collects features from the breakdown table.
  * Calls `/api/report` (POST JSON `{features: {...}}`).
  * Opens the generated report HTML in a new tab.

### 5.5. About page (`/about` → `about.html`)

Contains:

* What CodePulse AI is
* How the pipeline works
* Detailed CSV structure explanation
* Full feature list grouped by categories
* **"Download Template CSV"** button → `/download-template-csv`

---

## 6. Template CSV

Place this file at:

```text
static/files/template_metrics.csv
```

Example content:

```csv
loc,num_functions,avg_function_length,cyclomatic_complexity,max_nesting_depth,max_function_complexity,std_dev_function_complexity,lint_warning_count,comment_to_code_ratio,review_comment_count,test_coverage_percent,num_test_files_linked,code_churn_recent,num_contributors,file_age_days,fan_in,bug_count_total,bug_density,bugs_last_30_days,bug_trend,test_fail_count_related,days_since_last_change
800,25,32.0,12,4,25,4.0,30,0.25,18,75.0,3,120,5,500,12,8,1.2,2,0,3,30
```

This exactly matches the header in the About page and the model’s expected feature columns.

---

## 7. API Endpoints

### 7.1. `/api/predict` (POST, JSON)

**Request:**

```json
{
  "features": {
    "loc": 800,
    "num_functions": 25,
    "avg_function_length": 32.0,
    "...": 0
  }
}
```

**Response:**

```json
{
  "prediction": {
    "code_quality_score": 72.5,
    "bug_risk_class": 2,
    "bug_risk_label": "Slightly Risky",
    "bug_risk_probabilities": {
      "0": 0.05,
      "1": 0.18,
      "2": 0.40,
      "3": 0.25,
      "4": 0.10,
      "5": 0.02
    }
  },
  "explanations": {
    "quality": [...],
    "risk": [...]
  },
  "recommendations": {
    "quality": [...],
    "risk": [...]
  }
}
```

### 7.2. `/api/report` (POST, JSON)

Same request structure as `/api/predict`, but returns **HTML** instead of JSON:

* Generated via `report_generator.build_report_html(...)`
* Used by the **Download Full HTML Report** button on the breakdown page.

---

## 8. Notes & Future Enhancements

Possible future upgrades:

* Replace synthetic data with **real repository metrics** (from Git, SonarQube, etc.)
* Integrate directly with:

  * `pylint`, `flake8`, `radon`, `bandit` for automatic metric extraction
  * CI pipelines (GitHub Actions, GitLab CI) to auto-analyze changed files
* Add repository-level dashboard:

  * Aggregate quality and risk per module / directory
  * Trend charts over time (per commit or per release)
* Switch to PyTorch or ONNX deployment once **Python 3.13 support is mature** for all frameworks.

---

## 9. License / Credits

* ANN model & code logic: your implementation
* TensorFlow/Keras, scikit-learn, pandas, Flask: respective open-source licenses


