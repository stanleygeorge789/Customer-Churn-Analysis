# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/Imbalanced-Learning-ff69b4?style=flat-square)



The focus is business-constrained recall, campaign economics, and deployable architecture — not just model accuracy.

---

## 1. Problem Framing

Churn is predictable, but poorly operationalized in many organizations.

Acquiring new customers costs significantly more than retaining existing ones. Retention campaigns are budget constrained and must be precisely targeted.

**Executive Question**

> Who is most likely to leave next, and how many can we realistically save within budget?

### Project Objective

Build a churn prediction system that:

- Identifies high-risk customers within the next 30 days  
- Maximizes recall under campaign capacity constraints  
- Quantifies business impact  
- Produces interpretable churn drivers  

---

## 2. Dataset

Primary benchmark dataset: **Telco Customer Churn**

- Rows: 7,043  
- Churn rate: ~26.5%  
- File: `data/Telco-Customer-Churn.csv`

### Feature Categories

**Demographics**
- Gender  
- SeniorCitizen  
- Partner  
- Dependents  

**Service Usage**
- InternetService  
- StreamingTV  
- OnlineSecurity  
- TechSupport  

**Account Information**
- Tenure  
- Contract  
- MonthlyCharges  
- TotalCharges  

**Target**
- Churn (Yes / No)

Class imbalance: ~27% churn.

---

## 3. Key EDA Insights

- Month-to-month contracts churn 3–4× more than long-term contracts  
- Low tenure + high monthly charges = highest-risk segment  
- Lack of OnlineSecurity and TechSupport increases churn probability  
- Electronic check payment correlates with churn  
- Senior citizens show slightly elevated churn risk  

**Implications**

- Stratified cross-validation required  
- PR-AUC prioritized over ROC-AUC  
- Threshold tuning mandatory  

Default probability threshold of 0.5 is not financially optimal.

---

## 4. Feature Engineering

Enhancements applied:
- Interaction term: tenure × monthly charges  
- Log transformation of skewed monetary variables  

Final feature space: ~20 engineered features.

All preprocessing wrapped inside sklearn pipelines to prevent leakage.

---

## 5. Modeling Strategy

### Validation Design

- 5-fold stratified cross-validation  
- 20% hold-out test set  
- Random seed: 2025  
- Optuna hyperparameter tuning (60–120 trials per model)  
- Full pipeline integration  

### Primary Business Metric

**Recall at ~30% precision**

Interpretation:



---

## 6. Model Performance

| Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (Churn) |
|----------------------|---------|--------|--------------------------|------------|
| CatBoost             | 0.90    | 0.69   | 0.79                     | 0.60       |
| LightGBM             | 0.895   | 0.685  | 0.78                     | 0.595      |
| XGBoost              | 0.892   | 0.678  | 0.77                     | 0.59       |
| HistGradientBoosting | 0.885   | 0.66   | 0.75                     | 0.57       |
| Random Forest        | 0.87    | 0.63   | 0.72                     | 0.54       |
| Logistic Regression  | 0.85    | 0.59   | 0.68                     | 0.51       |

### What Matters

Difference between ROC 0.90 and 0.89 is negligible.

Difference between 79% and 68% recall at fixed campaign budget is financially meaningful.

On 1,500 churners:

- CatBoost captures ~1,185  
- Logistic Regression captures ~1,020  

165 additional customers retained per cycle.

Model choice affects revenue.

---

## 7. Final Model Selection

**Selected Model: CatBoost**

Reasons:

- Highest PR-AUC  
- Strong recall in high-recall region  
- Stable cross-validation variance  
- Native categorical feature handling  
- Minimal preprocessing requirements  

Churn patterns are nonlinear. Boosting models outperform linear baselines consistently.

---

## 8. Business Impact Simulation

Assumptions:

- 7,000 customers  
- 1,500 churn per cycle  
- Recall: 79%  
- Customers contacted: ~2,200  
- Retention offer cost: $20  
- Campaign success rate: 15%  
- ARPU: $60  

Estimated retained customers: ~330  
Estimated monthly revenue saved: ~$19,800  
Campaign cost: ~$44,000  

Conclusion:

Threshold optimization and lifetime value modeling are critical for profitability.

---

## 9. Explainability

SHAP analysis applied to CatBoost predictions.

Top churn drivers:

- Short tenure  
- Month-to-month contract  
- High monthly charges  
- No online security  
- Electronic check payment  

Actionable strategies:

- Early onboarding incentives  
- Contract migration offers  
- Bundled service promotions  
- Payment method optimization  

---

## 10. Production Architecture

Deployment flow:

1. Nightly batch scoring  
2. Store churn probability in CRM  
3. Risk segmentation  
4. Automated retention campaign triggering  
5. Quarterly retraining  
6. Drift monitoring  

Optional tools:

- FastAPI for model serving  
- MLflow for experiment tracking  
- Docker containerization  
- Airflow scheduled retraining  
- Evidently for data drift detection  

Designed for operational deployment, not notebook-only experimentation.

---

## 11. Project Structure

```
Customer-Churn-Analysis/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

Production logic resides in `src/`.  
Notebooks are exploratory only.

Structure emphasizes reproducibility, separation of concerns, and deployment readiness.

---

## 12. Quick Start

```bash
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python src/models/train_model.py --model catboost --save
```

---

## 13. Future Improvements


