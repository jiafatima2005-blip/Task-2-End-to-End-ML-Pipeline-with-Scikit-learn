# Task-2-End-to-End-ML-Pipeline-with-Scikit-learn
**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## Task Objective
Build a reusable, production-ready ML pipeline to predict customer churn
using the IBM Telco Customer Churn dataset. Implements preprocessing,
model training, hyperparameter tuning, and pipeline export.

---

## Dataset Used
| Property | Detail |
|---|---|
| Name | IBM Telco Customer Churn |
| Source | Public GitHub URL (no download needed) |
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | Churn (Yes/No) |
| Churn rate | ~26.5% |

---

## Methodology / Approach

### Pipeline Structure
Raw Data
|
v
ColumnTransformer
├── Numeric  : SimpleImputer(median) → StandardScaler
└── Categorical: SimpleImputer(mode) → OneHotEncoder
|
v
Classifier (LR or Random Forest)
|
v
GridSearchCV (5-fold CV, scoring=F1)
|
v
Best Pipeline → joblib export

### Models Trained
| Model | Tuning |
|---|---|
| Logistic Regression | C, penalty, solver |
| Random Forest | n_estimators, max_depth, min_samples_split |

---

## Key Results and Findings

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | ~80% | ~0.58 | ~0.84 |
| Logistic Regression (tuned) | ~81% | ~0.60 | ~0.85 |
| Random Forest (baseline) | ~79% | ~0.57 | ~0.82 |
| Random Forest (tuned) | ~80% | ~0.59 | ~0.83 |

### Key Observations
- Month-to-month contracts have the highest churn rate
- Tenure is the strongest predictor — new customers churn more
- Higher monthly charges increase churn probability
- GridSearchCV improved F1 over baseline for both models
- Exported pipeline loads and predicts in 2 lines of code

---

## How to Run

```bash
pip install -r requirements.txt
```
Open `Task2_ML_Pipeline_Churn.ipynb` and run all cells.
No GPU needed — runs on any laptop in under 5 minutes.

### Load exported pipeline
```python
import joblib
pipeline = joblib.load("best_churn_pipeline.pkl")
predictions = pipeline.predict(new_data)
```
