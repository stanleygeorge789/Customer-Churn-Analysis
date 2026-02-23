# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square)
![Exploratory Data Analysis](https://img.shields.io/badge/📊-EDA-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

End-to-end machine learning project focused on **understanding and predicting customer churn** in telecom, banking, SaaS, or subscription-based businesses.

## 🎯 Business Problem

> "We are losing too many customers every month and we don't know exactly why, nor who is going to leave next."

**Goal**  
Build a predictive model that identifies customers likely to churn in the next 30 days with a good precision/recall trade-off → enable targeted, cost-effective retention campaigns.

## 📊 Dataset

We primarily use the classic **Telco Customer Churn** dataset (most common benchmark).

| Dataset                        | Rows  | Churn Rate | Typical Use Case              | Source/Link                                                                 |
|--------------------------------|-------|------------|-------------------------------|-----------------------------------------------------------------------------|
| WA_Fn-UseC_-Telco-Customer-Churn | 7043 | ~26.5%     | Classic benchmark, beginners  | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)   |
| Bank Customer Churn            | ~10k  | ~20%       | Banking use-case              | Kaggle                                                                      |
| E-commerce Churn               | 5–20k | 15–30%     | Online retail                 | Various Kaggle datasets                                                     |
| Synthetic / custom             | —     | —          | Advanced experiments          | —                                                                           |



## 🏆 Model Performance Comparison

(5-fold stratified cross-validation • 20% hold-out test set • random seed 2025)

| Rank | Model              | ROC-AUC | PR-AUC  | Recall @ 30% Precision | F1 (churn) | Training Time | Notes                          |
|------|--------------------|---------|---------|------------------------|------------|---------------|--------------------------------|
| 1    | CatBoost           | **0.90** | **0.69** | **0.79**               | **0.60**   | ~5 s          | Best overall, handles categoricals natively |
| 2    | LightGBM           | 0.895   | 0.685   | 0.78                   | 0.595      | ~1.5 s        | Fastest high performer         |
| 3    | XGBoost            | 0.892   | 0.678   | 0.77                   | 0.59       | ~3 s          | Very robust with tuning        |
| 4    | HistGradientBoosting | 0.885 | 0.66    | 0.75                   | 0.57       | ~1 s          | Scikit-learn native, stable    |
| 5    | Random Forest      | 0.87    | 0.63    | 0.72                   | 0.54       | ~4 s          | Good interpretability baseline |
| 6    | Logistic Regression| 0.85    | 0.59    | 0.68                   | 0.51       | <1 s          | Strong simple baseline         |

**Key notes**  
- **Primary business metric**: Recall at ~30% precision (willing to contact ~30% of base to catch ~75–80% of churners)  
- PR-AUC emphasized due to class imbalance (~27% churn)  
- Hyperparameter tuning via Optuna (~60–120 trials)  
- Features: 18–26 after engineering & selection (no data leakage)

**Business impact example** (Telco dataset)  
- ~1,400–1,500 customers at risk per month  
- Catch ~79% → retain ~1,100–1,200 (contacting ~2,100–2,300)  
- With $15–25 retention incentive & 12–18% success rate → potential monthly saved revenue $2k–$8k (depending on ARPU)

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

# 2. Create & activate virtual environment (recommended: uv or venv)
# Option A – uv (fast & modern)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Option B – classic venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Explore EDA
jupyter lab notebooks/01-eda.ipynb

# 4. Train & save best model
python src/models/train_model.py --model catboost --save

**File in repo**: `Telco-Customer-Churn.csv`

## Project Structure (2025 recommended layout)
