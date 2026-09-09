# Credit Card Fraud Detection

A supervised classification project on the classic ULB credit card fraud dataset (Kaggle: `mlg-ulb/creditcardfraud`) — 284,807 transactions, 492 fraudulent (~0.17%). Built to compare modeling approaches on a severely imbalanced, real-world classification problem and land on a deployable, cost-justified model rather than just a leaderboard score.

## Result

**Random Forest**, evaluated at a cost-optimal decision threshold, is the recommended model.

| Model | PR-AUC | Cost-optimal total cost* |
|-------------------|-------|---------------|
| **Random Forest** | 0.873 | **$1,969.72** |
| XGBoost | 0.883 | $2,043.10 |
| Neural Network (focal loss) | ~0.73 | ~$2,100–2,180 |

\*Total cost = sum(missed-fraud transaction amounts) + $10 × false positives, on the held-out test set. See `notebooks/06_model_comparison.ipynb` for the full comparison and `notebooks/05_shap_explainability.ipynb` for why the model makes the predictions it does.

Notably, XGBoost has a marginally higher raw PR-AUC, but Random Forest wins at the specific operating point that actually matters once real dollar costs are considered — a reminder that an aggregate metric doesn't always predict which model is best to deploy.

## Repo Structure

```
data/                            gitignored -- see Setup below
src/
    data_prep.py                 shared load/scale/split, used by every notebook
notebooks/
    01_eda_preprocessing.ipynb   exploration, class imbalance, statistical significance check
    02_random_forest.ipynb       baseline, tuning attempt, threshold analysis
    03_xgboost_tuned.ipynb       same treatment as RF
    04_neural_network_experiment.ipynb   class-weighted / focal loss / deeper NN -- documented negative result
    05_shap_explainability.ipynb SHAP analysis on the winning model
    06_model_comparison.ipynb    side-by-side comparison and final verdict
models/                          saved trained models + chosen thresholds (per model)
predictions/                     saved test-set predicted probabilities (per model)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Download the dataset from Kaggle (`mlg-ulb/creditcardfraud`) and place `creditcard.csv` in `data/`.
3. Run the notebooks in order (01 → 06). Each model notebook saves its trained model, predictions, and chosen thresholds to `models/` and `predictions/`, so `06_model_comparison.ipynb` can be re-run on its own afterward without retraining anything.

## Methodology Notes

- **Threshold selection isn't arbitrary.** Every model notebook evaluates three thresholds: the naive default (0.5), the F1-maximizing threshold (best generic precision/recall balance), and a cost-optimal threshold (minimizes `sum(missed fraud $) + $10 * false positives`, using the assumption that a missed fraud costs its transaction amount and a false alarm costs a fixed review cost).
- **Hyperparameter tuning is cross-validated against the baseline, not assumed to help.** Both RF and XGBoost ran `RandomizedSearchCV`, but in both cases the plain default configuration beat the tuning search under cross-validation — so both notebooks keep the untuned baseline rather than a "tuned" model that's actually worse.
- **The Neural Network section is a kept negative result, not a mistake to hide.** It documents why a class-weighted NN, focal loss, and a deeper architecture all still lost to the tree-based models, which is a useful finding in its own right for this kind of PCA-compressed, severely imbalanced tabular data.
