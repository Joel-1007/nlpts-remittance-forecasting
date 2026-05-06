import pandas as pd
import numpy as np

# 1. Check what the sentiment features actually look like
train = pd.read_csv('/kaggle/working/features_train.csv')
test  = pd.read_csv('/kaggle/working/features_test.csv')
sent  = pd.read_csv('/kaggle/working/sentiment_vectors.csv')

print("=== SENTIMENT VECTORS ===")
print(sent.shape)
print(sent.dtypes)
print(sent.head(3).to_string())

print("\n=== SENTIMENT STATS ===")
num_cols = sent.select_dtypes(include='number').columns.tolist()
print(sent[num_cols].describe().to_string())

print("\n=== TRAIN TARGET STATS ===")
print(f"inward_flow: mean={train['inward_flow'].mean():.1f}, std={train['inward_flow'].std():.1f}")
print(f"Train range: {train['date'].min()} → {train['date'].max()}")
print(f"Test range:  {test['date'].min()} → {test['date'].max()}")

print("\n=== TRAIN inward_flow (first 10) ===")
print(train[['date','inward_flow']].head(10).to_string())

print("\n=== TEST inward_flow ===")
print(test[['date','inward_flow']].to_string())

# 2. Check for train/test distribution shift
print("\n=== DISTRIBUTION SHIFT ===")
print(f"Train mean: {train['inward_flow'].mean():.1f}  Test mean: {test['inward_flow'].mean():.1f}")
print(f"Train max:  {train['inward_flow'].max():.1f}  Test max:  {test['inward_flow'].max():.1f}")
print(f"Ratio (test/train mean): {test['inward_flow'].mean()/train['inward_flow'].mean():.2f}x")

# 3. Check EPU overlap
if 'EPU_Index' in train.columns:
    print(f"\n=== EPU INDEX ===")
    print(f"Train EPU: mean={train['EPU_Index'].mean():.1f}, std={train['EPU_Index'].std():.1f}")
    print(f"Test  EPU: mean={test['EPU_Index'].mean():.1f}, std={test['EPU_Index'].std():.1f}")

# 4. Check what baseline forecasts look like vs actuals
fc = pd.read_csv('/kaggle/working/baseline_forecasts.csv')
print("\n=== BASELINE FORECASTS vs ACTUAL ===")
print(fc[['date','actual','sarima','naive_drift']].to_string())