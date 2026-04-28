import pyarrow.parquet as pq
import pandas as pd

# Read the parquet file
df = pd.read_parquet("tx-ad-2026/demo_1000.parquet")

print(f"数据集形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")
print(f"\n前 10 条数据:")
print(df.head(10).to_string())