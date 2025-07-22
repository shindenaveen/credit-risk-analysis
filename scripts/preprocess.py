import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Load data
data_path = os.path.join('data', 'crx.data')
columns = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16'
]
df = pd.read_csv(data_path, header=None, names=columns, na_values='?')

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save processed data
processed_path = os.path.join('data', 'crx_processed.csv')
df.to_csv(processed_path, index=False)
print(f"Data preprocessing completed. Processed file saved to {processed_path}")
