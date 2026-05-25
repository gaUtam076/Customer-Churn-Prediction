import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib

# Setup absolute paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION PROJECT")
print("  Step 2: Training Models")
print("=" * 60)

# Load data dynamically using absolute paths
data_file = MODELS_DIR / "train_test_data.pkl"
if not data_file.exists():
    print(f"❌ Error: {data_file} not found. Run step 1 first.")
    exit(1)

X_train, X_test, y_train, y_test = joblib.load(data_file)
print("✅ Train/Test Data Loaded Successfully")
print(f"Training Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}\n")

# Initialize models with balanced class weights to fight data skew
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

best_model_name = None
best_f1 = -1
best_model_obj = None
model_performance = []

# Train and Evaluate
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Check if model supports predict_proba for ROC-AUC calculations
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
    else:
        roc_auc = 0.5

    # CRITICAL FIX: Collect all 5 evaluation metrics required by Streamlit's comparison dataframe
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print(f"✅ {name} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\n")
    
    # Save with exact string labels that streamlit_app.py expects in its subset keys
    model_performance.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

print("=" * 60)
print("  Step 3: Selecting Best Model")
print("=" * 60)
print(f"🏆 Best Model: {best_model_name}")
print(f"🎯 Best F1-Score : {best_f1:.4f}")

# Calculate additional performance metrics for the best model to supply the app sidebar
final_preds = best_model_obj.predict(X_test)
best_accuracy = accuracy_score(y_test, final_preds)

model_info_payload = {
    "model_name": best_model_name,
    "model": best_model_obj,
    "metrics": {
        "Accuracy": round(best_accuracy * 100, 2),  
        "F1-Score": round(best_f1, 4)
    }
}

# Save everything safely using absolute paths
joblib.dump(model_info_payload, MODELS_DIR / "best_model.pkl")
joblib.dump(models, MODELS_DIR / "all_models.pkl")

# Export complete metric framework dataframe to clear the Streamlit KeyError
pd.DataFrame(model_performance).to_csv(MODELS_DIR / "model_comparison.csv", index=False)
print("✅ Structured model payload and comprehensive comparison matrices saved successfully.")

print("=" * 60)
print("  Step 4: Model Evaluation")
print("=" * 60)
print("\nClassification Report:\n")
print(classification_report(y_test, final_preds))

# Save Confusion Matrix array placeholder
cm = confusion_matrix(y_test, final_preds)
joblib.dump(cm, SCREENSHOTS_DIR / "confusion_matrix_data.pkl")
print("✅ Evaluation Completed Successfully.")