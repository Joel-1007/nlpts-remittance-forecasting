import pandas as pd, numpy as np
from scipy.stats import t as t_dist

c7 = pd.read_csv('/kaggle/working/cell7_forecasts.csv')
c9 = pd.read_csv('/kaggle/working/cell9_predictions.csv')

print("C7 cols:", c7.columns.tolist())
print("C9 cols:", c9.columns.tolist())

# Merge — adjust key column names if needed
merged = c7[['quarter','actual','Diff_XGB_diff']].merge(
    c9[['quarter','GARCH_Gated']], on='quarter'
)

e1 = (merged['actual'] - merged['Diff_XGB_diff']).values
e2 = (merged['actual'] - merged['GARCH_Gated']).values
d = e1**2 - e2**2
n = len(d)
dm_stat = d.mean() / (np.std(d, ddof=1) / np.sqrt(n))
p_val = 2 * (1 - t_dist.cdf(abs(dm_stat), df=n-1))
print(f"\nDM test (Diff_XGB vs GARCH_Gated): stat={dm_stat:.4f}  p={p_val:.4f}  N={n}")
print(f"Interpretation: {'significant' if p_val < 0.05 else 'NOT significant'} at 5%")