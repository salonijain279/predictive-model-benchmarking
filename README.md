# Predictive Model Benchmarking: Classification & Regression

**Python, scikit-learn, XGBoost**

Two independent model-comparison studies, each running a battery of classical
ML algorithms through the same rigorous pipeline: `GridSearchCV` hyperparameter
tuning under stratified/k-fold cross-validation on the training set only, then
a single held-out test-set evaluation to select and report the winner.

## Study 1: Bank Marketing — Term Deposit Subscription (Classification)

**Question:** will a bank customer subscribe to a term deposit, based on
demographics and campaign-contact history?

**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
— Portuguese bank direct-marketing campaign records.

**Models compared:** Logistic Regression, Decision Tree, Random Forest, KNN,
Gradient Boosting, XGBoost — each tuned via `GridSearchCV` (stratified 5-fold
CV, ROC-AUC scoring).

### Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **91.0%** | 65.8% | 48.1% | 0.556 | **0.934** |
| Random Forest | 90.6% | 69.0% | 35.7% | 0.471 | 0.931 |
| Gradient Boosting | 90.9% | 67.0% | 44.2% | 0.533 | 0.930 |
| Logistic Regression | 90.1% | 64.5% | 33.9% | 0.445 | 0.906 |
| Decision Tree | 89.8% | 58.4% | 45.5% | 0.511 | 0.884 |
| KNN | 90.3% | 66.2% | 34.9% | 0.457 | 0.865 |

**Winner: XGBoost** (n_estimators=200, max_depth=5, learning_rate=0.1).
Boosting's sequential error-correction and built-in regularization let it
capture nonlinear feature interactions while controlling overfitting —
reflected in the strongest ROC-AUC and the best overall accuracy/F1 balance.

## Study 2: Real Estate Valuation — Price per Unit Area (Regression)

**Question:** what predicts the price per unit area of a residential
property in New Taipei City?

**Dataset:** [UCI Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
— transaction date, house age, distance to nearest MRT station, number of
nearby convenience stores, and geographic coordinates.

**Models compared:** Ridge, Lasso, Decision Tree, Random Forest, KNN,
Gradient Boosting, XGBoost — each tuned via `GridSearchCV` (5-fold CV, RMSE
scoring).

### Results (held-out test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** | **5.372** | **3.659** | **0.828** |
| XGBoost | 5.729 | 3.912 | 0.804 |
| Gradient Boosting | 5.961 | 4.149 | 0.788 |
| KNN | 6.401 | 4.588 | 0.756 |
| Decision Tree | 6.457 | 4.727 | 0.752 |
| Lasso | 7.284 | 5.305 | 0.684 |
| Ridge | 7.286 | 5.295 | 0.684 |

**Winner: Random Forest** (n_estimators=200, max_depth=10, min_samples_leaf=3,
max_features=sqrt). Averaging predictions across many trees, each built on a
different bootstrap sample and feature subset, reduces variance and captures
nonlinearity without the overfitting risk of a single deep tree — evident in
its clear RMSE/R² lead over every other model, including XGBoost.

## Methodology (both studies)

- Numeric and categorical features identified programmatically and handled
  through a `ColumnTransformer` (standard scaling for numeric, one-hot
  encoding for categorical), wrapped in an sklearn `Pipeline` so
  preprocessing is fit only on training folds — no leakage into CV or test
  scoring.
- Train/test split held out before any tuning; cross-validation used only on
  the training partition.
- Every model tuned with its own `GridSearchCV` over a small, targeted
  hyperparameter grid, scored by the metric appropriate to the task
  (ROC-AUC for classification, RMSE for regression).
- Final model selection based on held-out test performance, not CV score —
  avoiding the common mistake of reporting cross-validation numbers as if
  they were an unbiased final estimate.

## Repo structure

```
predictive-model-benchmarking/
├── notebooks/
│   └── 01_model_benchmarking.ipynb   both studies, executed end to end
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_model_benchmarking.ipynb
```

Both datasets are fetched automatically at runtime via the [`ucimlrepo`](https://pypi.org/project/ucimlrepo/)
package — no manual download needed.

## Limitations

- Hyperparameter grids are intentionally small/targeted rather than
  exhaustive; a wider search could shift rankings, especially among the
  closely-clustered top 3 models in each study.
- Bank Marketing is a class-imbalanced dataset (most customers do not
  subscribe); ROC-AUC was used as the primary tuning metric for exactly this
  reason, but recall on the minority (subscribing) class remains the
  weakest metric across all models — a real business decision would need to
  weigh the cost of a missed subscriber against outreach cost per contact,
  which this study does not model.
- Real Estate Valuation is a small dataset (414 transactions from one city
  and time period); results should not be read as generalizing to other
  markets.
