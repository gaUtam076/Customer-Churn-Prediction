import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Setup absolute paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION PROJECT")
print("  Step 1: Data Cleaning + EDA")
print("=" * 60)

csv_path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load or generate dataset
if csv_path.exists():
    print("✅ Found Kaggle Dataset. Loading...")
    df = pd.read_csv(csv_path)
else:
    print("⚠️ Kaggle dataset not found. Generating realistic synthetic dataset...")
    np.random.seed(42)
    n_samples = 7043
    
    # CRITICAL FIX: Ensure a balanced ~26% Churn rate instead of a 95/5 split
    churn_choices = np.random.choice(['No', 'Yes'], size=n_samples, p=[0.74, 0.26])
    tenure_choices = np.random.randint(1, 72, size=n_samples)
    monthly_choices = np.random.uniform(18.25, 118.75, size=n_samples)
    
    df = pd.DataFrame({
        'customerID': [f"{i:04d}-AAAAA" for i in range(n_samples)],
        'gender': np.random.choice(['Female', 'Male'], size=n_samples),
        'SeniorCitizen': np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], size=n_samples),
        'Dependents': np.random.choice(['No', 'Yes'], size=n_samples, p=[0.70, 0.30]),
        'tenure': tenure_choices,
        'PhoneService': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['No', 'Yes', 'No phone service'], size=n_samples),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], size=n_samples),
        'OnlineSecurity': np.random.choice(['No', 'Yes', 'No internet service'], size=n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
        'DeviceProtection': np.random.choice(['No', 'Yes', 'No internet service'], size=n_samples),
        'TechSupport': np.random.choice(['No', 'Yes', 'No internet service'], size=n_samples),
        'StreamingTV': np.random.choice(['No', 'Yes', 'No internet service'], size=n_samples),
        'StreamingMovies': np.random.choice(['No', 'Yes', 'No internet service'], size=n_samples),
        'Contract': np.random.choice(['Month-to-month', 'Two year', 'One year'], size=n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
        'MonthlyCharges': monthly_choices,
        'TotalCharges': (tenure_choices * monthly_choices).astype(str),
        'Churn': churn_choices
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Synthetic data saved to {csv_path}")

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop(columns=['customerID'], errors='ignore', inplace=True)

# Encode Categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns.drop('Churn')

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Scale Numerical features
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save preprocessing objects
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")

# Train/Test Split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed split data
joblib.dump((X_train, X_test, y_train, y_test), MODELS_DIR / "train_test_data.pkl")
print("✅ Preprocessing Complete. Train/Test datasets saved successfully.")