import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Load the best model info
best_model_info = joblib.load('./results/best_model/TabPFN-Health-Best-Model.joblib')

# Access components
model = best_model_info['model']
parameters = best_model_info['parameters']
auc_score = best_model_info['auc_score']

print("Best Model Parameters:")
print("=" * 50)
print(f"Best AUC Score: {auc_score:.4f}")
for key, value in parameters.items():
    print(f"{key}: {value}")

# Load and prepare data
df = pd.read_excel("data/AI4healthcare.xlsx")
features = [c for c in df.columns if c.startswith("Feature")]
X = df[features].copy()
y = df["Label"].copy()

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Calculate metrics
acc = accuracy_score(y, predictions)
auc = roc_auc_score(y, probabilities[:, 1])
f1 = f1_score(y, predictions)

print("\nModel Performance on Full Dataset:")
print("=" * 50)
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}") 