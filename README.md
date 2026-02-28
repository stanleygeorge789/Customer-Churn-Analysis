# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

Production-oriented machine learning project to predict customer churn in telecom and subscription businesses such as SaaS and banking.

The focus is not leaderboard metrics. The focus is business-constrained recall, campaign economics, and deployable architecture.

---

## 1. Problem Framing

Churn is predictable. Most companies simply do not operationalize it correctly.

Acquiring new customers costs significantly more than retaining existing ones. Yet retention campaigns are often poorly targeted and budget constrained.

**Executive Question**

> "We are losing customers every month. Who is most likely to leave next, and how many can we realistically save within budget?"

### Objective

Build a churn prediction system that:

- Identifies high-risk customers in the next 30 days  
- Maximizes recall under campaign capacity constraints  
- Quantifies business impact  
- Produces interpretable drivers  

This project optimizes for revenue impact, not cosmetic accuracy gains.

---

## 2. Dataset

**Primary dataset:** Telco Customer Churn  

- Rows: 7,043  
- Churn rate: ~26.5%  

Repository file:
```
data/Telco-Customer-Churn.csv
```

### Feature Groups

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

**Account Attributes**
- Tenure  
- Contract  
- MonthlyCharges  
- TotalCharges  

**Target**
- Churn (Yes / No)

Class imbalance: ~27%.

---

## 3. Key EDA Insights

- Month-to-month contracts churn 3–4× more than long-term contracts  
- Low tenure + high monthly charges is highest-risk segment  
- Lack of OnlineSecurity and TechSupport increases churn  
- Electronic check payment correlates with churn  
- Senior citizens show slightly elevated risk  

### Implications

- Stratified cross-validation required  
- Precision-Recall curves preferred over ROC  
- Threshold tuning mandatory  

Default threshold = 0.5 is financially naive.

---

## 4. Feature Engineering

Enhancements applied:

- Tenure buckets  
- Revenue-to-tenure ratio  
- Service count aggregation  
- Binary encoding for contract  
- Tenure × monthly charges interaction  
- Log transformation of skewed monetary variables  

Final feature space: ~20 engineered variables.

All preprocessing wrapped inside sklearn pipelines to prevent leakage.

---

## 5. Modeling Strategy

### Validation Design

- 5-fold stratified cross-validation  
- 20% hold-out test set  
- Fixed random seed (2025)  
- Optuna hyperparameter tuning (60–120 trials per model)  
- Full preprocessing inside pipeline  

### Primary Business Metric

**Recall at ~30% precision**

Interpretation:

If the company contacts 30% of customers, how many churners can it capture?

PR-AUC prioritized due to class imbalance.

---

## 6. Model Comparison

| Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (Churn) |
|----------------------|---------|--------|--------------------------|------------|
| CatBoost             | 0.90    | 0.69   | 0.79                     | 0.60       |
| LightGBM             | 0.895   | 0.685  | 0.78                     | 0.595      |
| XGBoost              | 0.892   | 0.678  | 0.77                     | 0.59       |
| HistGradientBoosting | 0.885   | 0.66   | 0.75                     | 0.57       |
| Random Forest        | 0.87    | 0.63   | 0.72                     | 0.54       |
| Logistic Regression  | 0.85    | 0.59   | 0.68                     | 0.51       |

### What Matters

Difference between ROC 0.90 and 0.89 is noise.

Difference between 79% and 68% recall under fixed campaign budget is material.

On 1,500 churners:

- CatBoost captures ~1,185  
- Logistic Regression captures ~1,020  

That is 165 additional customers per cycle.

Model choice changes revenue.

---

## 7. Final Model Selection

**Selected Model: CatBoost**

Reasons:

- Highest PR-AUC  
- Strongest recall in high-recall region  
- Stable cross-validation  
- Native categorical handling  
- Minimal preprocessing  

Churn behavior is nonlinear. Boosting models outperform linear baselines.

---

## 8. Business Impact Simulation

Assumptions:

- 7,000 customers  
- 1,500 churn per cycle  
- Recall = 79%  
- 2,200 customers contacted  
- Retention cost = $20  
- Offer success rate = 15%  
- ARPU = $60  

Estimated retained customers: ~330  
Monthly revenue saved: ~$19,800  
Campaign cost: ~$44,000  

Conclusion:

Threshold tuning and lifetime value modeling determine profitability.  
Raw recall alone is insufficient.

---

## 9. Explainability

SHAP applied to CatBoost.

Top churn drivers:

- Short tenure  
- Month-to-month contract  
- High monthly charges  
- No online security  
- Electronic check payment  

These insights support targeted interventions.

---

## 10. Production Architecture

Typical deployment flow:

1. Nightly batch scoring  
2. Store churn probability in CRM  
3. Risk segmentation  
4. Automated campaign triggering  
5. Quarterly retraining  
6. Drift monitoring  

Optional stack:

- FastAPI  
- MLflow  
- Docker  
- Airflow  
- Evidently  

Designed for operational use, not just experimentation.

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

All production logic resides in `src/`.  
Notebooks are exploratory only.

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

## 13. Future Extensions

- Survival analysis for time-to-churn  
- Cost-sensitive learning  
- Uplift modeling  
- Real-time scoring API  
- Customer lifetime value integration  
- Continuous performance monitoring  

---
