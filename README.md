# Customer Churn Prediction & Analysis

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/scikit--learn-1.4+-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=for-the-badge"> <img src="https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=for-the-badge">

**End-to-end machine learning project** focused on understanding and predicting customer churn in a **telecom / banking / SaaS / subscription** business.

## 🎯 Business Problem

> "We are losing too many customers every month and we don't know exactly why nor who is going to leave next."

**Goal**:  
Build a model that can **predict which customers are likely to churn** in the next 30 days with good enough precision/recall trade-off → enable **targeted retention campaigns**.

## 📊 Dataset

Common public datasets used in this repository (choose one):

| Dataset                          | Rows   | Churn Rate | Most used for          | Link / Source                                 |
|-------------------------------|--------|------------|------------------------|-----------------------------------------------|
| Telco Customer Churn (IBM)    | ~7k    | ~27%       | beginners & comparison | Kaggle                                        |
| WA_Fn-UseC_-Telco-Customer-Churn | 7043 | 26.5%      | classic benchmark      | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Bank Customer Churn           | 10k    | ~20%       | banking use-case       | Kaggle / Ravel                            |
| E-commerce Churn              | ~5–20k | 15–30%     | online retail          | various Kaggle datasets                       |
| Synthetic / own generated     | —      | —          | advanced experiments   | —                                             |

## Project Structure (2025 recommended layout)

```text
customer-churn-analysis/
├── data/
│   ├── raw/                  ← never modify!
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-baseline.ipynb
│   ├── 04-model-tuning.ipynb
│   └── 05-interpretability.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   └── visualization/
│       └── custom_plots.py
├── models/                   ← saved models (.joblib / .pkl / .cbm / .onnx)
├── reports/
│   ├── figures/
│   └── churn_report.md / churn_dashboard.html
├── requirements.txt
├── environment.yml           (optional – conda)
├── .gitignore
├── README.md                ← you're reading this
└── churn_predictor/          (optional package structure)
    ├── __init__.py
    └── pipeline.py
