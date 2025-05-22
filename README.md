# 💳 Credit Card Fraud Detection Web App

A Machine Learning-powered web application built as part of the [ML Developer Bootcamp](https://www.notion.so/ML-Developer-Bootcamp-Milestone-1-1f7e0fadd2c880758320e27970a80716). This project identifies fraudulent transactions using advanced ML techniques and is deployed using the **Streamlit** framework.

## 🚀 Project Overview

This project focuses on **credit card fraud detection** using anonymized data with 29 input features (V1–V28 + Amount). It includes end-to-end steps from data exploration and modeling to deployment of the best-performing model via a user-friendly web interface.

---

## 📆 Milestone-Based Development

### ✅ Day 01: Exploratory Data Analysis (EDA)
- Loaded and inspected the dataset from Kaggle.
- Identified key insights about class imbalance (highly skewed toward non-fraud cases).
- Plotted distributions and correlation heatmaps.
- Observed outliers and scaling needs for features like `Amount`.

### ✅ Day 02: Baseline Machine Learning Models
- Applied several machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Naive Bayes
  - Random Forest
  - XGBoost
  - LightGBM
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1-score
  - ROC AUC
  - Confusion Matrix
- Concluded that Random Forest and Gradient Boosted models performed best in early tests.

### ✅ Day 03: Feature Engineering + Model Tuning
- Applied advanced preprocessing:
  - Feature scaling
  - Handling class imbalance using undersampling and SMOTE
  - Feature selection
- Hyperparameter tuning using `GridSearchCV`
- Finalized **Random Forest** as the best-performing model based on cross-validation scores and balanced precision-recall metrics.
- Exported this model using `joblib`.

### ✅ Day 04–05: Deployment with Streamlit
- Built a **Streamlit app**:
  - **App page** where users can upload CSVs and receive fraud predictions.
- Included:
  - Clean user interface
  - Footer for contacts 
- Integrated the pre-trained Random Forest model (`best_model.joblib`) into the Streamlit backend.

---

## 🌐 Web App Features

- Upload a CSV file with columns: `V1–V28 + Amount`
- Instant prediction of fraud probabilities
- Visual fraud count summary
- Model uses the best-performing hypertuned **Random Forest**

---

## 📊 Tech Stack

- **Python 3.11**
- **Scikit-learn** for model building
- **Streamlit** for front-end web app
- **Pandas**, **Joblib**, **Matplotlib/Seaborn** for backend logic
- **Kaggle** for training and tuning

---

## 📽 Demo Video

Watch the complete walkthrough of the deployed web application:

---

## 🧠 Key Learnings

- Hands-on experience in ML pipeline: from EDA → Modeling → Evaluation → Deployment
- Importance of model interpretability in fraud detection
- Streamlit for rapid ML deployment with minimal code
- Understanding trade-offs between different models, thresholds, and metrics
