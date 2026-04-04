# 💳 Credit Risk Prediction using Machine Learning

## 📌 Project Overview

This project focuses on building a **Credit Risk Prediction Model** to identify whether a customer is likely to default on a loan.

The model uses financial and behavioral features to help financial institutions make **data-driven lending decisions**.

---

## 🎯 Objective

* Predict whether a customer will **default (1)** or **not default (0)**
* Handle **imbalanced data**
* Improve model performance using **feature engineering**
* Provide **model explainability**

---

## 📊 Dataset Description

The dataset contains customer financial and behavioral information such as:

* Income
* Loan Amount
* Employment Years
* Credit Score
* Existing Loans
* Credit Utilization
* Late Payments
* Savings & Checking Balance
* Loan Term & Interest Rate

### 🎯 Target Variable

* `default`

  * 1 → Customer will default
  * 0 → Customer will not default

---

## 🔍 Exploratory Data Analysis (EDA)

Performed:

* Distribution analysis of target variable
* Correlation heatmap
* Feature vs target analysis (e.g., credit score vs default)
* Detection of **class imbalance**

---

## ⚙️ Feature Engineering

Created new features:

* `debt_to_income = loan_amount / income`
* `emi_estimate = loan_amount / loan_term`

These features improve model understanding of financial behavior.

---

## ⚠️ Handling Imbalanced Data

Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset:

* Prevents bias toward majority class
* Improves recall for default prediction

---

## 🤖 Models Used

* Logistic Regression
* Random Forest
* XGBoost (Primary Model)

---

## 📈 Model Evaluation

Metrics used:

* ROC-AUC Score
* Precision
* Recall
* F1 Score

### Key Focus:

* **Recall** → Important in credit risk to avoid missing defaulters

---

## 📊 Performance Visualization

* ROC Curve
* Precision-Recall Curve
* SHAP Feature Importance

---

## 🧠 Model Explainability (SHAP)

Used SHAP to interpret predictions:

Top important features:

* Late Payments
* Credit Score
* Debt-to-Income Ratio
* Credit Utilization

---

## 🔧 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib, Seaborn

---

## 💾 Model Saving

Model saved using:

```python
joblib.dump(model, "credit_model.pkl")
```

---

## 🌐 Deployment (Optional)

A simple **Streamlit app** can be used for real-time predictions.

---

## 💼 Business Impact

* Helps banks reduce financial losses
* Improves loan approval decisions
* Identifies high-risk customers early

---

## 🚀 Key Learnings

* Handling imbalanced datasets using SMOTE
* Importance of feature engineering in fintech
* Using SHAP for model explainability
* Evaluating models beyond accuracy

---

## 📌 Future Improvements

* Feature selection
* Deployment on cloud (AWS/Render)
