import pandas as pd
import glob

parquet_files = glob.glob("D:/Datasets/synthbuster-plus/data/*.parquet")
if parquet_files:
    df = pd.read_parquet(parquet_files[0])
    print("Columns:", df.columns.tolist())
    print("\nFirst row content:")
    for col in df.columns:
        val = df[col].iloc[0]
        print(f"  {col}: {type(val)}")
        if isinstance(val, dict):
            print(f"    Keys: {val.keys()}")
            for k, v in val.items():
                print(f"      {k}: {type(v)}")