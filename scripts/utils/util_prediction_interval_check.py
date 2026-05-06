import pandas as pd, numpy as np

fc = pd.read_csv('/kaggle/working/baseline_forecasts.csv')
print("Columns:", fc.columns.tolist())
print(fc.head(3))

for level in ['80', '90', '95']:
    lc = f'lower_{level}' if f'lower_{level}' in fc.columns else f'pi_lower_{level}'
    uc = f'upper_{level}' if f'upper_{level}' in fc.columns else f'pi_upper_{level}'
    if lc in fc.columns and uc in fc.columns:
        cov = ((fc['actual'] >= fc[lc]) & (fc['actual'] <= fc[uc])).mean()
        print(f"  {level}% PI coverage: {cov:.1%}")
    else:
        print(f"  {level}% — columns not found")