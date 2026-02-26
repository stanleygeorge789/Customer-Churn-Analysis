# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

End-to-end machine learning project to **understand, quantify, and predict customer churn** in telecom and other subscription-based businesses such as banking and SaaS.

This project focuses on predictive performance, imbalance handling, business cost optimization, and decision threshold calibration for real retention campaigns.

---

## 1. Business Context

Customer churn is one of the most expensive revenue leaks in subscription businesses. Acquiring a new customer often costs significantly more than retaining an existing one.

**Executive Problem**

> “We are losing customers every month. We do not clearly know who will leave next, nor which segments are most vulnerable.”

### Objective

Develop a predictive system that:

- Identifies customers likely to churn in the next 30 days  
- Optimizes recall while controlling campaign cost  
- Enables targeted retention campaigns  
- Produces interpretable churn drivers  

The goal is business impact, not just model accuracy.

---

## 2. Dataset

Primary benchmark dataset: **Telco Customer Churn**

| Dataset                          | Rows  | Churn Rate | Use Case             | Source |
|----------------------------------|-------|------------|----------------------|--------|
| WA_Fn-UseC_-Telco-Customer-Churn | 7043  | ~26.5%     | Telecom benchmark    | Kaggle |
| Bank Customer Churn             | ~10k  | ~20%       | Banking churn risk   | Kaggle |
| E-commerce Churn                | 5–20k | 15–30%     | Retail churn         | Kaggle |

**Repository file:** `data/Telco-Customer-Churn.csv`

### Feature Categories

**Demographics**
- Gender  
- SeniorCitizen  
- Partner  
- Dependents  

**Service-related**
- InternetService  
- PhoneService  
- StreamingTV  
- OnlineSecurity  
- TechSupport  

**Account-related**
- Tenure  
- Contract  
- MonthlyCharges  
- TotalCharges  

**Target**
- Churn (Yes / No)

---

## 3. Exploratory Data Analysis

Key observations:

- Month-to-month contracts churn 3–4× more than long-term contracts  
- High monthly charges + low tenure represent highest-risk segment  
- Customers without OnlineSecurity or TechSupport show elevated churn  
- Electronic check payment correlates with churn  
- Senior citizens have slightly higher churn probability  

Dataset imbalance: ~27% churn.

Implications:

- Stratified cross-validation required  
- PR-AUC prioritized over ROC-AUC  
- Recall-based threshold optimization  

---

## 4. Feature Engineering

Enhancements applied:

- Tenure buckets (0–6, 6–12, 12–24, 24+ months)  
- Revenue-to-tenure ratio  
- Service count aggregation  
- Binary encoding for contract type  
- Interaction term: tenure × monthly charges  
- Log transform of skewed monetary variables  

Final feature space: 18–26 engineered features.

---

## 5. Modeling Strategy

### Validation Design

- 5-fold stratified cross-validation  
- 20% hold-out test set  
- Random seed 2025  
- Full pipeline to prevent data leakage  
- Hyperparameter tuning via Optuna (60–120 trials per model)

### Evaluation Metric

Primary metric:

**Recall at ~30% precision**

Interpretation:

Contact approximately 30% of customers and capture 75–80% of churners.

PR-AUC is prioritized due to class imbalance.

---

## 6. Model Performance Comparison

(5-fold CV + hold-out validation)

| Rank | Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (Churn) |
|------|----------------------|---------|--------|--------------------------|------------|
| 1    | CatBoost             | 0.90    | 0.69   | 0.79                     | 0.60       |
| 2    | LightGBM             | 0.895   | 0.685  | 0.78                     | 0.595      |
| 3    | XGBoost              | 0.892   | 0.678  | 0.77                     | 0.59       |
| 4    | HistGradientBoosting | 0.885   | 0.66   | 0.75                     | 0.57       |
| 5    | Random Forest        | 0.87    | 0.63   | 0.72                     | 0.54       |
| 6    | Logistic Regression  | 0.85    | 0.59   | 0.68                     | 0.51       |

### Why CatBoost Performs Best

- Native categorical handling  
- Minimal preprocessing  
- Stable under tuning  
- Strong bias-variance balance  

---

## 7. Business Impact Simulation

Assumptions:

- 7,000 customers  
- ~1,500 churn per cycle  
- Model recall = 79%  
- Customers contacted ≈ 2,200  
- Retention offer cost = $20  
- Campaign success rate = 15%  
- ARPU = $60  

Estimated retained customers:
~330

Potential saved monthly revenue:
~$19,800

Campaign cost:
~$44,000

Conclusion:
Threshold selection and lifetime value modeling are critical for profitability.

---

## 8. Model Explainability

SHAP applied to CatBoost predictions.

Top churn drivers:

- Short tenure  
- Month-to-month contract  
- High monthly charges  
- Lack of online security  
- Electronic check payment  

This supports actionable interventions such as:

- Early onboarding offers  
- Contract migration incentives  
- Bundled service promotions  
- Security feature upselling  

---

## 9. Production Architecture

Typical deployment flow:

1. Nightly batch scoring  
2. Store churn probability in CRM  
3. Risk segmentation  
4. Automated campaign triggering  
5. Quarterly retraining and drift monitoring  

Optional enhancements:

- FastAPI model serving  
- MLflow model registry  
- Docker containerization  
- Airflow scheduled retraining  
- Drift monitoring with Evidently  

---

## 10. Project Structure

```
Customer-Churn-Analysis/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-modeling.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate.py
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## 11. Quick Start

```bash
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

jupyter lab notebooks/01-eda.ipynb

python src/models/train_model.py --model catboost --save
```

---

## 12. Future Improvements

- Survival analysis for time-to-churn modeling  
- Cost-sensitive learning  
- Uplift modeling  
- Real-time scoring pipeline  
- Customer lifetime value integration  
- Continuous drift monitoring  
