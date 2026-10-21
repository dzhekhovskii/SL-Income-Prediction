# Supervised Learning Project: Adult / Census Income

**Dataset:** Adult (Census Income), UCI Machine Learning Repository, loaded from OpenML (`fetch_openml(name="adult", version=2)`).

**Task:** binary classification. Predict whether a person earns more than $50K a year (`>50K`) from 14 attributes of the 1994 US Census. Only 23.9% of people are positive, so the main metric is **PR-AUC**.

---

## Results

Final model: **XGBoost**, tuned with Optuna, trained **without class re-weighting**, used with a **decision threshold of 0.35**.

| Model (after Optuna) | PR-AUC, 5-fold CV | PR-AUC, validation |
|---|---|---|
| Logistic regression | 0.778 | 0.772 |
| Random forest | 0.814 | 0.810 |
| **XGBoost (selected)** | **0.827** | **0.824** |

**Test set** (used once, at the end):

| PR-AUC | ROC-AUC | Recall @0.35 | Precision @0.35 | Brier | ECE |
|---|---|---|---|---|---|
| 0.813 | 0.923 | 0.79 | 0.65 | 0.092 | 0.004 |

Main findings:
- `class_weight` and SMOTE leave the PR-AUC unchanged (0.770 vs 0.772): they only shift the threshold. They also inflate probabilities (34.7% predicted vs 23.9% real). We therefore keep the unweighted model and choose the threshold explicitly.
- The unweighted XGBoost is already well calibrated (ECE 0.011). Platt and isotonic calibration do not improve it.
- Threshold 0.35 is chosen for a target recall of 80%: in our scenario, missing a high-income client costs more than an unnecessary contact.

---

## Repository structure

```
Supervised-learning-project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── SL_PROJECT_V4.ipynb     # full analysis (all steps, with outputs)
├── src/                        # Python modules (to be added)
├── report/
│   ├── report.tex              # LaTeX report
│   ├── report.pdf
│   └── figures/                # plots exported from the notebook
└── mlflow.db, mlruns/          # experiment tracking (not versioned)
```

---

## Installation

### Google Colab (recommended)

1. Open `notebooks/SL_PROJECT_V4.ipynb` in Colab.
2. Run the first cell to mount Google Drive.
3. Install the dependencies:
   ```python
   !pip install -q -r requirements.txt
   ```
4. **Runtime → Run all.** It takes about 25 minutes, most of it for the Optuna search.

### Local

```bash
git clone https://github.com/nayekanoute/Supervised-learning-project.git
cd Supervised-learning-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/SL_PROJECT_V4.ipynb
```

On macOS, XGBoost needs OpenMP: `brew install libomp`. Skip the Google Drive cell when running locally.

---

## Pipeline steps

| Step | What is done |
|---|---|
| 1. EDA and cleaning | Types, missing values, 52 exact duplicates removed, `fnlwgt` and `education` dropped |
| 2. Feature engineering | `is_married`, `has_capital`, `net_capital`, `hours_group`, regrouped `workclass`, `is_us` |
| 3. Class imbalance | No correction vs `class_weight` vs SMOTE; decision: no re-weighting, tune the threshold |
| 4. Model progression | Logistic regression → random forest → XGBoost |
| 5. Pipeline | Feature engineering, imputation, scaling, one-hot encoding and model in one scikit-learn `Pipeline` |
| 6–7. Optuna and MLflow | 15 trials per model, stratified 5-fold CV on train only; 45 runs logged in MLflow |
| 8. SHAP | Global importance, beeswarm, 4 local explanations (TP, TN, FP, FN) |
| 9. Calibration | Frozen model, raw vs Platt vs isotonic, reliability diagram, Brier, ECE |
| Threshold | `precision_recall_curve`, target recall 80% → threshold 0.35 |
| Test | One-shot evaluation of the final model |

**Data split:** 60% train / 20% validation / 20% test, stratified, `random_state=42`. The validation set is split in half for calibration and threshold selection. The test set is only used in the last cell.

---

## MLflow

In Colab:

```python
get_ipython().system_raw("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &")
from google.colab import output
output.serve_kernel_port_as_window(5000)
```

Locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 and select the `adult-income` experiment.

---

## Limitations

- No fairness audit: SHAP shows that sex influences predictions, both directly (`sex_Female`) and through `relationship_Husband`.
- Data from 1994, US population only.
- No prediction API yet; the code still lives in the notebook and has to be exported to `src/`.

---

## Report and authors

The full report is in [`report/report.pdf`](report/report.pdf).

Authors: [Name 1], [Name 2], [Name 3]. See the contributions table in the report.
