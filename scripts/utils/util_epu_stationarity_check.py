import pandas as pd, numpy as np

ft = pd.read_csv('/kaggle/working/features_train.csv').sort_values('date')
epu = ft['EPU_Index'].values

print(f"First 20 EPU values: {epu[:20].round(2)}")
print(f"Value of epu[0]: {epu[0]:.4f}")
print(f"Count of obs equal to epu[0]: {(epu == epu[0]).sum()}")
print(f"First non-backfilled index: {(epu != epu[0]).argmax()}")

try:
    from arch import arch_model
    cutoff = (epu != epu[0]).argmax()
    for label, series in [("Full (N=72)", epu), (f"Trimmed (N={len(epu)-cutoff})", epu[cutoff:])]:
        am = arch_model(series, mean='Constant', vol='GARCH', p=1, q=1)
        res = am.fit(disp='off')
        p = res.params
        print(f"\nGARCH {label}: omega={p['omega']:.4f}  alpha={p['alpha[1]']:.4f}  beta={p['beta[1]']:.4f}  persistence={p['alpha[1]']+p['beta[1]']:.4f}")
except Exception as e:
    print(f"\narch not available: {e}")
    print("Check: import arch; print(arch.__version__)")