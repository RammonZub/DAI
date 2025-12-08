import pandas as pd
import numpy as np

try:
    df = pd.read_csv("db/db_computers_clean.csv")
    print("Data loaded successfully.")
    print("\nColumn Data Types:")
    print(df.dtypes)
    
    numeric_cols = ['price_euros', 'ram_gb', 'ssd_gb', 'screen_size_inch']
    for col in numeric_cols:
        if col in df.columns:
            print(f"\nChecking column: {col}")
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            print(f"Non-numeric values in {col}: {non_numeric}")
            if non_numeric > 0:
                print("Sample non-numeric values:")
                print(df[pd.to_numeric(df[col], errors='coerce').isna()][col].head())
        else:
            print(f"\nColumn {col} not found!")

except Exception as e:
    print(f"Error loading data: {e}")
