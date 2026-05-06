import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

ft = pd.read_csv('/kaggle/working/features_train.csv').sort_values('date')
data = ft[['inward_flow','EPU_Index']].dropna()
data_diff = data.diff().dropna()

print("=== Granger: EPU → inward_flow (levels, lags 1–8) ===")
grangercausalitytests(data, maxlag=8, verbose=True)

print("\n=== Granger: ΔEPU → Δinward_flow (first differences, lags 1–4) ===")
grangercausalitytests(data_diff, maxlag=4, verbose=True)