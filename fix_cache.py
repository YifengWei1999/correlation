import pandas as pd
import numpy as np

# Read the corrupted cache file
cache_file = "market_data/^TNX_2023-01-01_2024-12-31.csv"

print("Reading corrupted cache file...")
df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")
print(f"First few rows:")
print(df.head())

# Remove rows that contain the ticker symbol
print("\nCleaning data...")
# Find rows that contain actual numeric data
cleaned_rows = []
for i, (idx, row) in enumerate(df.iterrows()):
    # Check if this row contains numeric data in the Close column
    try:
        close_val = float(row['Close'])
        if not np.isnan(close_val) and close_val > 0:
            cleaned_rows.append(i)
    except (ValueError, TypeError):
        print(f"Skipping row {i}: {row['Close']}")
        continue

# Keep only the rows with valid data
if cleaned_rows:
    start_idx = min(cleaned_rows)
    df_clean = df.iloc[start_idx:].copy()
    
    # Ensure all OHLCV columns are numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop any remaining NaN rows
    df_clean = df_clean.dropna()
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Cleaned columns: {list(df_clean.columns)}")
    print(f"Data types:")
    print(df_clean.dtypes)
    print(f"\nFirst few rows of cleaned data:")
    print(df_clean.head())
    
    # Save the cleaned data
    df_clean.to_csv(cache_file)
    print(f"\nCleaned data saved to {cache_file}")
else:
    print("No valid data rows found!") 