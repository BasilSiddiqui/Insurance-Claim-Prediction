# 🛡️ Insurance Claim Status Prediction with XGBoost

This project explores the use of machine learning to predict whether an insurance claim will be approved or denied, based on real-world claim, agent, and vendor data. Despite the dataset’s limitations, the pipeline developed is reusable for similar fraud detection or classification tasks in insurance and finance.

---

## 📂 Project Structure

- `employee_data.csv` — Contains data on insurance agents.
- `insurance_data.csv` — Contains claim-level information.
- `vendor_data.csv` — Contains data on third-party vendors.
- `main.py` — Main script with all preprocessing, modeling, and evaluation.
- `README.md` — Project summary and insights.
- 📸 *[space for confusion matrix & feature importance plots]*

---

## 🔍 Objective

Build a binary classifier to predict `CLAIM_STATUS`:
- `0`: Approved
- `1`: Denied

---

## 🧠 Workflow

### 1. 📥 Data Import
Loaded datasets using `pandas`.

### 2. 🧹 Data Cleaning & Merging
- Parsed and standardized date columns.
- Merged agent and vendor data into the claims dataset.

### 3. 🧠 Feature Engineering
Created key derived features:
- `DAYS_TO_REPORT` — Delay between loss and report.
- `CLAIM_RATIO` — Ratio of claim to premium amount.
- `AGENT_TENURE_DAYS` — Agent experience in days.

### 4. 🔤 Encoding
- Used **one-hot encoding** for all categorical variables.

### 5. 🎯 Target Variable
Encoded `CLAIM_STATUS` as the binary target.

### 6. ⚖️ Addressing Class Imbalance
- Tried both **SMOTE** and **scale_pos_weight** approaches.
- Evaluated both using **AUC** and confusion matrix.

### 7. 🚀 Model Training (XGBoost)
Used `XGBClassifier`:
- GridSearchCV with AUC as the scoring metric.
- Evaluated model on unseen test data.

---

## 📊 Key Results

- **Best GridSearch AUC (train/val):** 0.98+
- **Test AUC:** ~0.47 (indicating overfitting)
- **Confusion Matrix:** Model failed to predict any denied claims (Class 1)

Despite trying SMOTE, parameter tuning, and balancing, the model failed to generalize due to **extreme class imbalance and poor signal for denial prediction**.

---

## 📉 Dataset Limitations

> This dataset is not well-suited for claim denial prediction:
- Only ~5% of claims are denied.
- Features offer little separation between classes.
- Label imbalance leads to poor generalization, even with oversampling and tuning.

---

## ✅ Why This Project Still Matters

- Demonstrates full ML pipeline on real insurance data.
- Shows how to:
  - Engineer domain-specific features.
  - Address imbalance with SMOTE or class weighting.
  - Optimize models with GridSearchCV.
  - Evaluate using AUC and confusion matrix.
- Provides a **skeletal, production-ready framework** for classification tasks.

---

## 🧑‍💻 About the Developer

This project was developed by a data science student passionate about applying AI/ML in real-world business contexts like insurance, fraud detection, and finance.

---

## 🖼️ Visualization Placeholders

You can add images for:
1. 📊 Confusion Matrix
2. 📈 Top 15 Feature Importances

---

## 🚀 Future Improvements

- Collect or use datasets with more balanced or explainable outcomes.
- Include NLP or external data for improved signal (e.g., claim descriptions).
- Try alternative models like LightGBM or interpretable models like SHAP.

---

## 🗃️ Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
