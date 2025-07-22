import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Update the data path to point to your Downloads folder
data_path = r'C:\Users\navee\Downloads\credit+approval\crx.data' 

columns = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',a
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16'
]

try:
    df = pd.read_csv(data_path, header=None, names=columns, na_values='?')
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)  # Using median instead of mean for robustness
    
    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create output directory if it doesn't exist
    output_dir = r'C:\Users\navee\Downloads\credit+approval\processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    processed_path = os.path.join(output_dir, 'crx_processed.csv')
    df.to_csv(processed_path, index=False)
    print(f"Data preprocessing completed. Processed file saved to {processed_path}")

except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    print("Please check:")
    print("1. The file exists at that location")
    print("2. The filename is exactly 'crx.data' (some browsers rename it)")
    print("3. You have permission to access the file")
except Exception as e:
    print(f"An error occurred: {str(e)}")
