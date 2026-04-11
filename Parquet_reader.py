import pandas as pd
import numpy as np

df = pd.read_parquet('your_file.parquet')

print("Shape:", df.shape)
print("\nDtypes:")
print(df.dtypes)
print("\nFirst row:")
print(df.iloc[0].to_dict())
