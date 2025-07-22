import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the preprocessed data
data_path = r'C:\Users\navee\Downloads\credit+approval\processed\crx_processed.csv'
df = pd.read_csv(data_path)

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Train Gradient Boosting (as an XGBoost alternative)
xgb = GradientBoostingClassifier(random_state=42)
xgb.fit(X_train, y_train)

# Save models
model_dir = r'C:\Users\navee\Downloads\credit+approval\models'
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf, os.path.join(model_dir, 'random_forest.pkl'))
joblib.dump(xgb, os.path.join(model_dir, 'xgboost.pkl'))

print("Models trained and saved successfully.")
