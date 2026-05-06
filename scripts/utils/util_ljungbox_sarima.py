# ═══════════════════════════════════════════════════════
# LJUNG-BOX TEST ON SARIMA RESIDUALS
# ═══════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

ft = pd.read_csv('/kaggle/working/features_train.csv').sort_values('date')
y_train = ft['inward_flow'].values

# Refit best SARIMA
model = SARIMAX(y_train, order=(0,1,2), seasonal_order=(1,1,1,4))
result = model.fit(disp=False)
residuals = result.resid

# Ljung-Box at lags 4, 8, 12, 16
lb = acorr_ljungbox(residuals, lags=[4, 8, 12, 16], return_df=True)
print("=== Ljung-Box Test on SARIMA Residuals ===")
print("H0: residuals are white noise (no autocorrelation)")
print(lb.to_string())
print()
for lag, row in lb.iterrows():
    status = "✅ white noise" if row['lb_pvalue'] > 0.05 else "❌ autocorrelation present"
    print(f"  Lag {lag:2d}: stat={row['lb_stat']:.4f}  p={row['lb_pvalue']:.4f}  {status}")

# Normality of residuals
from scipy.stats import shapiro, jarque_bera
sw_stat, sw_p = shapiro(residuals)
jb_stat, jb_p = jarque_bera(residuals)
print(f"\n=== Residual Normality ===")
print(f"  Shapiro-Wilk:  stat={sw_stat:.4f}  p={sw_p:.4f}  {'✅ normal' if sw_p > 0.05 else '⚠️ non-normal'}")
print(f"  Jarque-Bera:   stat={jb_stat:.4f}  p={jb_p:.4f}  {'✅ normal' if jb_p > 0.05 else '⚠️ non-normal'}")

# Heteroscedasticity (ARCH effects in residuals)
from statsmodels.stats.diagnostic import het_arch
arch_stat, arch_p, _, _ = het_arch(residuals, nlags=4)
print(f"\n=== ARCH Effects in Residuals ===")
print(f"  ARCH LM test (lag=4): stat={arch_stat:.4f}  p={arch_p:.4f}  {'⚠️ heteroscedastic' if arch_p < 0.05 else '✅ homoscedastic'}")