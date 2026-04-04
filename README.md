# 💳 Credit Risk Modeling & Scorecard Project

## 📌 Overview

This project focuses on **Credit Risk Prediction** using machine learning and **Scorecard Generation** techniques used in the banking/fintech industry.

It predicts whether a customer will **default on a loan** and assigns a **credit score** to classify customers into risk categories.

---

## 🚀 Key Highlights

* End-to-end ML pipeline (EDA → Modeling → Evaluation → Scorecard)
* Handles **imbalanced data using SMOTE**
* Uses multiple models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Model evaluation using:

  * Accuracy
  * ROC-AUC
  * Classification Report
* **Cross-validation & consistency checks**
* **Hyperparameter tuning (GridSearchCV)**
* **WOE (Weight of Evidence) & IV (Information Value)**
* Industry-style **Scorecard Generation**
* Model explainability using SHAP (optional extension)
* Model saved using **Pickle (joblib)**

---

## 📂 Dataset

* File used: `large_credit_risk_dataset.csv`
* Target Variable: `default`

  * `1` → Default
  * `0` → Non-default

---

## 🔍 Exploratory Data Analysis (EDA)

* Target imbalance check
* Correlation heatmap
* Outlier detection using boxplots
* Feature engineering:

  * `debt_to_income`
  * `emi_estimate`

---

## ⚙️ Feature Engineering

```python
df['debt_to_income'] = df['loan_amount'] / df['income']
df['emi_estimate'] = df['loan_amount'] / df['loan_term_months']
```

---

## 🧠 Model Pipeline

Pipeline includes:

* **Standard Scaling**
* **SMOTE (handling imbalance)**
* ML Model

```python
Pipeline([
    ("Preprocessor", StandardScaler),
    ("SMOTE", SMOTE),
    ("Model", Model)
])
```

---

## 🤖 Models Used

| Model               | Purpose                   |
| ------------------- | ------------------------- |
| Logistic Regression | Baseline + Scorecard      |
| Random Forest       | Non-linear patterns       |
| XGBoost             | High performance boosting |

---

## 📊 Model Evaluation

Metrics used:

* Accuracy
* ROC-AUC Score
* Classification Report
* ROC Curve

---

## 🔁 Model Validation

* Cross Validation (ROC-AUC)
* Train vs Test score comparison
* Random label test (data leakage check)

---

## ⚡ Hyperparameter Tuning

```python
GridSearchCV(
    param_grid={
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }
)
```

---

## 🏦 Scorecard Generation (Industry-Level)

Steps:

1. Convert variables into **WOE bins**
2. Calculate **Information Value (IV)**
3. Train Logistic Regression on WOE data
4. Generate scorecard

```python
import scorecardpy as sc

bins = sc.woebin(data, y="default")
data_woe = sc.woebin_ply(data, bins)
card = sc.scorecard(bins, model, X.columns)
score = sc.scorecard_ply(data, card)
```

---

## 📈 Risk Segmentation

Customers are categorized into:

| Score Range | Risk Category |
| ----------- | ------------- |
| < 600       | High Risk     |
| 600–650     | Medium Risk   |
| 650–700     | Low Risk      |
| > 700       | Very Safe     |

---

## 📊 Score Distribution

* Histogram used to visualize customer score spread

---

## 💾 Model Saving

```python
import joblib
joblib.dump(model, "credit_model.pkl")
```

---

## 🧰 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* ScorecardPy
* Matplotlib, Seaborn

---

## 🎯 Business Impact

* Helps banks **reduce default risk**
* Enables **automated credit approval**
* Provides **interpretable risk scoring**
* Aligns with real-world **credit underwriting systems**

---

## 🔥 Why This Project Stands Out

* Combines **ML + Finance domain knowledge**
* Includes **Scorecard (used in real banks)**
* Covers **end-to-end pipeline**
* Demonstrates **model validation & robustness checks**
* Interview-ready with strong storytelling

---

## 📌 Future Improvements

* Deploy using Flask / FastAPI
* Add dashboard (Streamlit)
* Use real-world datasets (Home Credit, LendingClub)



