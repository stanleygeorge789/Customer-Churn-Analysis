# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

End-to-end machine learning project to **understand and predict customer churn** in telecom (or similar subscription-based businesses like banking/SaaS).

## 🎯 Business Problem

> "We are losing too many customers every month and we don't know exactly why, nor who is going to leave next."

**Goal**  
Develop a predictive model to identify customers likely to churn in the next 30 days, balancing precision and recall → enable targeted, cost-effective retention campaigns.

## 📊 Dataset

Primarily using the classic **Telco Customer Churn** dataset (widely used benchmark).

| Dataset                              | Rows  | Churn Rate | Typical Use Case          | Source/Link                                                                 |
|--------------------------------------|-------|------------|---------------------------|-----------------------------------------------------------------------------|
| WA_Fn-UseC_-Telco-Customer-Churn     | 7043  | ~26.5%     | Classic benchmark, beginners | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)   |
| Bank Customer Churn                  | ~10k  | ~20%       | Banking use-case          | Kaggle                                                                      |
| E-commerce Churn                     | 5–20k | 15–30%     | Online retail             | Various Kaggle datasets                                                     |
| Synthetic / custom                   | —     | —          | Advanced experiments      | —                                                                           |

**File in repo**: `Telco-Customer-Churn.csv`

## Project Structure
