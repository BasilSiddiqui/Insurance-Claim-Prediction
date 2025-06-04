# Insurance Claim Status Prediction with XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-red)

This project predicts insurance claim approval/denial status using machine learning. It processes insurance claims data, agent information, and vendor details to build an XGBoost classification model.

## Features

- **Data Integration**: Merges claims, employee, and vendor datasets
- **Feature Engineering**: Creates temporal and financial risk indicators
- **Model Training**: XGBoost classifier with imbalanced data handling
- **Evaluation**: ROC/AUC analysis, feature importance visualization
- **Production-Ready**: Clean pipeline from raw data to predictions

## Data Sources

1. `employee_data.csv`: Agent information (joining dates, etc.)
2. `insurance_data.csv`: Claim transactions and policy details
3. `vendor_data.csv`: Vendor payment information

## Code Structure

```bash
insurance-claims/
├── data/                    # Raw data files
├── notebooks/               # Jupyter notebooks
├── scripts/
│   ├── preprocess.py        # Data cleaning
│   ├── train.py            # Model training
│   └── evaluate.py         # Performance metrics
└── README.md
```

## Key Technical Components

### 1. Feature Engineering

```python
# Temporal features
df['DAYS_TO_REPORT'] = (df['REPORT_DT'] - df['LOSS_DT']).dt.days
df['AGENT_TENURE_DAYS'] = (df['TXN_DATE_TIME'] - df['DATE_OF_JOINING']).dt.days

# Financial ratio
df['CLAIM_RATIO'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
```

### 2. Model Training

```python
model = XGBClassifier(
    scale_pos_weight=ratio_denied_to_approved,  # Handle class imbalance
    eval_metric='logloss',
    early_stopping_rounds=10
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
```

### 3. Evaluation Metrics

```python
# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)

# Feature Importance
plot_importance(model, max_num_features=15)
