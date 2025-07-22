import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load processed data
df = pd.read_csv("crx_processed.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

# Evaluate both models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

rf_report = classification_report(y_test, rf_pred)
xgb_report = classification_report(y_test, xgb_pred)

# Save the evaluation results
os.makedirs("output", exist_ok=True)
with open("output/evaluation_report.txt", "w") as f:
    f.write("Random Forest Model Evaluation:\n")
    f.write(rf_report)
    f.write("\n" + "="*50 + "\n\n")
    f.write("XGBoost (GradientBoostingClassifier) Model Evaluation:\n")
    f.write(xgb_report)

print("Evaluation report saved to output/evaluation_report.txt")
