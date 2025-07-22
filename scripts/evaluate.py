import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os

# Load data
data_path = r'C:\Users\navee\Downloads\credit+approval\processed\crx_processed.csv'
df = pd.read_csv(data_path)

X = df.drop('class', axis=1)
y = df['class']

# Split same as training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
model_dir = r'C:\Users\navee\Downloads\credit+approval\models'
rf = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
xgb = joblib.load(os.path.join(model_dir, 'xgboost.pkl'))

# Predictions
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

# Evaluate
rf_report = classification_report(y_test, rf_preds)
xgb_report = classification_report(y_test, xgb_preds)

# Save evaluation to file
output_dir = r'C:\Users\navee\Downloads\credit+approval\output'
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write("Random Forest Report:\n")
    f.write(rf_report + "\n")
    f.write("\nGradient Boosting Report:\n")
    f.write(xgb_report)

print(f"Evaluation reports saved to {report_path}")
