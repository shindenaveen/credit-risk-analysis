import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Path to your original dataset
data_path = r'C:\Users\navee\Downloads\credit+approval\crx.data'

# Corrected column list
columns = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
    'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class'
]

try:
    df = pd.read_csv(data_path, header=None, names=columns, na_values='?')

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save processed data
    output_dir = r'C:\Users\navee\Downloads\credit+approval\processed'
    os.makedirs(output_dir, exist_ok=True)

    processed_path = os.path.join(output_dir, 'crx_processed.csv')
    df.to_csv(processed_path, index=False)

    print(f"Data preprocessing completed.\nProcessed file saved to: {processed_path}")

except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    print("Please check if the file exists and the name is correct.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
