import pandas as pd

datasets = ['train.csv', 'test.csv']

for file in datasets:
    df = pd.read_csv(file)

    # Report missing values
    missing_count = df.isna().sum().sum()
    total_rows = len(df)
    print(f"\n{file}: {missing_count} total missing values across {df.isna().sum().count()} columns")
    if missing_count > 0:
        # Drop any row with at least one missing value
        df = df.dropna(axis=0)
        dropped = total_rows - len(df)
        print(f"Dropped {dropped} rows containing missing values.")
    else:
        print("No missing values found. No rows dropped.")

    # Overwrite original dataset with cleaned data
    df.to_csv(file, index=False)
    print(f"Overwritten {file} with cleaned data (rows: {len(df)})")
