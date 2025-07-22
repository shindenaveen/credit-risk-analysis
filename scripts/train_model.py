# scripts/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load preprocessed data
data = pd.read_csv("../data/crx_processed.csv")

# Split into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Save models
joblib.dump(rf, "../models/random_forest.pkl")
joblib.dump(xgb, "../models/xgboost.pkl")

# Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("XGBoost Report:\n", classification_report(y_test, y_pred_xgb))
