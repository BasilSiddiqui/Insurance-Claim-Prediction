import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -----------------------------
# 📥 Load Data
# -----------------------------

employee = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\archive\employee_data.csv")
insurance = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\archive\insurance_data.csv")
vendor = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\archive\vendor_data.csv")

# -----------------------------
# 🧹 Clean & Merge
# -----------------------------

date_cols = ['TXN_DATE_TIME', 'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT']
for col in date_cols:
    insurance[col] = pd.to_datetime(insurance[col])

employee['DATE_OF_JOINING'] = pd.to_datetime(employee['DATE_OF_JOINING'])

df = insurance.merge(employee, on='AGENT_ID', how='left')
df = df.merge(vendor, on='VENDOR_ID', how='left')

# -----------------------------
# 🧠 Feature Engineering
# -----------------------------

df['DAYS_TO_REPORT'] = (df['REPORT_DT'] - df['LOSS_DT']).dt.days.clip(lower=0)
df['CLAIM_RATIO'] = (df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']).replace([float('inf'), -float('inf')], 0).fillna(0)
df['AGENT_TENURE_DAYS'] = (df['TXN_DATE_TIME'] - df['DATE_OF_JOINING']).dt.days.clip(lower=0)

# -----------------------------
# 🎯 Target Encoding (CLAIM_STATUS)
# -----------------------------

df['CLAIM_STATUS'] = df['CLAIM_STATUS'].map({'A': 0, 'D': 1})  # 0 = Approved, 1 = Denied
y = df['CLAIM_STATUS']

# -----------------------------
# 🧹 Drop Unused Columns
# -----------------------------

drop_cols = [
    'CUSTOMER_NAME', 'ADDRESS_LINE1_x', 'ADDRESS_LINE2_x', 'CITY_x', 'STATE_x', 'POSTAL_CODE_x',
    'ADDRESS_LINE1_y', 'ADDRESS_LINE2_y', 'CITY_y', 'STATE_y', 'POSTAL_CODE_y',
    'ADDRESS_LINE1', 'ADDRESS_LINE2', 'CITY', 'STATE', 'POSTAL_CODE',
    'ROUTING_NUMBER', 'ACCT_NUMBER', 'EMP_ROUTING_NUMBER', 'EMP_ACCT_NUMBER',
    'SSN', 'TRANSACTION_ID', 'CUSTOMER_ID', 'POLICY_NUMBER',
    'AGENT_NAME', 'VENDOR_NAME', 'DATE_OF_JOINING', 'TXN_DATE_TIME',
    'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT', 'CLAIM_STATUS'
]

X = df.drop(columns=drop_cols, errors='ignore')

# -----------------------------
# 🔤 Encode Categorical Features
# -----------------------------

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# -----------------------------
# 🧪 Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 🚀 Model Training
# -----------------------------

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
'''
# Add early stopping for better XGBoost performance
model = XGBClassifier(
    eval_metric='logloss',
    early_stopping_rounds=10,
    n_estimators=1000  # Let early stopping choose the best iteration
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
'''
# -----------------------------
# 📊 Model Evaluation
# -----------------------------

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 🔍 Feature Importance
# -----------------------------

feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
plt.title("Top 15 Features Influencing Claim Status")
plt.tight_layout()
plt.show()