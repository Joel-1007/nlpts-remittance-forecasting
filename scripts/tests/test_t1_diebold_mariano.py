"""
TEST T1 — Diebold-Mariano & Advanced Forecast Metrics
======================================================
Reviewer requirement:
  • Diebold-Mariano test to prove forecast superiority is statistically
    significant (not just lower MAPE/RMSE).
  • Rolling-origin bootstrap for stability across time windows.
  • Additional metrics: MAE, sMAPE, MASE, prediction-interval coverage.
  • Formal definition and justification of YoY Accuracy.

HOW TO USE ON KAGGLE:
  • Requires india_remittance_processed.csv (or india_remittance_final.csv)
    in /kaggle/working/ or /kaggle/input/.

Output: t1_forecast_metrics.csv, t1_diebold_mariano.csv
"""

import subprocess, sys
try:
    import statsmodels
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

sep = "=" * 72

SEARCH_DIRS = ["/kaggle/working/", "/kaggle/input/", "./", "../"]

def find(filename):
    for d in SEARCH_DIRS:
        p = Path(d) / filename
        if p.exists():
            return str(p)
    inp = Path("/kaggle/input")
    if inp.exists():
        for sub in inp.iterdir():
            p = sub / filename
            if p.exists():
                return str(p)
    return None

# ── Load real data — abort if not found ───────────────────────────────────────
combined_path = (
    find("inward_quarterly_seasonal.csv") or
    find("features_combined.csv") or
    find("remittances_quarterly.csv") or
    find("india_remittance_processed.csv") or
    find("india_remittance_final.csv")
)

if combined_path is None:
    raise FileNotFoundError(
        "T1 could not find a quarterly remittance CSV.\n"
        "Expected one of: remittances_quarterly.csv, features_combined.csv\n"
        "in /kaggle/working/ or /kaggle/input/."
    )

print(sep)
print("T1 — DIEBOLD-MARIANO & ADVANCED FORECAST METRICS")
print("     Mode: REAL DATA")
print(sep)

df = pd.read_csv(combined_path)
print(f"  Loaded: {combined_path}  (columns: {list(df.columns)})")

if "inward_flow" not in df.columns:
    print(f"❌  'inward_flow' column not found. Available: {list(df.columns)}")
    raise SystemExit(1)

sort_col = next((c for c in ["quarter", "date", "year"] if c in df.columns), df.columns[0])
df = df.sort_values(sort_col)

series = df["inward_flow"].dropna().values.astype(float)
N = len(series)
print(f"\n  Series length: {N}")

# ── Helper: forecast metrics ──────────────────────────────────────────────────
def mae(y, yhat):    return np.mean(np.abs(y - yhat))
def rmse(y, yhat):   return np.sqrt(np.mean((y - yhat)**2))
def mape(y, yhat):   return np.mean(np.abs((y - yhat) / (y + 1e-9))) * 100
def smape(y, yhat):
    return np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-9)) * 100

def mase(y, yhat, y_train, seasonality=4):
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006)"""
    naive_errors = np.abs(np.diff(y_train, n=seasonality))
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return mae(y, yhat) / (scale + 1e-9)

def yoy_accuracy(y, yhat):
    """
    YoY Accuracy — formal definition
    ==================================
    For each t, compare whether year-over-year change direction is correct:
      ΔYoY_actual_t = y_t − y_{t−4}
      ΔYoY_hat_t   = ŷ_t − y_{t−4}  (actual lag used, not forecast)
    YoY Accuracy = mean(sign(ΔYoY_actual) == sign(ΔYoY_hat))
    Economic justification: policy makers care about annual growth direction.
    """
    f = 4
    if len(y) <= f:
        return np.nan
    y_lag     = y[:-f]
    y_curr    = y[f:]
    yhat_curr = yhat[f:]
    delta_actual = y_curr - y_lag
    delta_hat    = yhat_curr - y_lag
    hits = (np.sign(delta_actual) == np.sign(delta_hat)).astype(float)
    return hits.mean()

# ── Diebold-Mariano test ──────────────────────────────────────────────────────
def diebold_mariano(e1, e2, h=1, loss="squared"):
    """
    DM test (Harvey, Leybourne & Newbold 1997 finite-sample correction).
    H0: models have equal forecast accuracy.
    """
    d = e1**2 - e2**2 if loss == "squared" else np.abs(e1) - np.abs(e2)
    n = len(d)
    d_mean = np.mean(d)
    max_lag  = h - 1
    gamma0   = np.var(d, ddof=1)
    gamma_sum = 0
    for lag in range(1, max_lag + 1):
        gamma_sum += (1 - lag / (max_lag + 1)) * np.mean(
            (d[lag:] - d_mean) * (d[:-lag] - d_mean))
    var_d   = (gamma0 + 2 * gamma_sum) / n
    k       = (n + 1 - 2*h + h*(h-1)/n) / n
    dm_stat = d_mean / np.sqrt(k * var_d + 1e-12)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    better  = "Model-1 (SARIMA)" if d_mean < 0 else "Model-2 (Naive)"
    return dm_stat, p_value, better

# ── 1. Rolling-origin evaluation ─────────────────────────────────────────────
print("\n[1] Rolling-origin evaluation (expanding window)")

TRAIN_MIN = max(int(N * 0.5), 8)
horizons  = [1, 2, 4]

results = []
for origin in range(TRAIN_MIN, N):
    y_train = series[:origin]
    try:
        m1  = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,0,1,4),
                      enforce_stationarity=False, enforce_invertibility=False)
        r1  = m1.fit(disp=False, maxiter=200)
        fc1 = r1.forecast(steps=max(horizons))
    except Exception:
        fc1 = np.full(max(horizons), np.nan)

    fc2 = np.full(max(horizons), np.nan)
    for h in horizons:
        idx = origin - 4
        if idx >= 0:
            fc2[h-1] = series[idx]

    for h in horizons:
        if origin + h - 1 < N:
            actual = series[origin + h - 1]
            results.append({
                "origin":     origin,
                "horizon":    h,
                "actual":     actual,
                "fc_sarima":  fc1[h-1] if not np.isnan(fc1[h-1]) else np.nan,
                "fc_naive":   fc2[h-1],
            })

df_roll = pd.DataFrame(results).dropna()
print(f"  Origins evaluated : {df_roll['origin'].nunique()}")
print(f"  Total forecasts   : {len(df_roll)}")

# ── 2. Metrics per horizon ────────────────────────────────────────────────────
print("\n[2] Forecast metrics by horizon")
print(f"  {'Horizon':>8}  {'Model':>12}  {'RMSE':>10}  {'MAE':>10}  "
      f"{'MAPE%':>10}  {'sMAPE%':>10}  {'MASE':>10}  {'YoY%':>10}")
print("  " + "-" * 85)

metric_rows = []
for h in horizons:
    sub = df_roll[df_roll["horizon"] == h]
    y   = sub["actual"].values
    for model_name, fc_col in [("SARIMA", "fc_sarima"), ("Naive-S4", "fc_naive")]:
        fc    = sub[fc_col].values
        valid = ~np.isnan(fc) & ~np.isnan(y)
        if valid.sum() < 3:
            continue
        yv, fv = y[valid], fc[valid]
        _mase  = mase(yv, fv, series[:TRAIN_MIN])
        _yoy   = yoy_accuracy(yv, fv) * 100 if len(yv) > 4 else np.nan
        row = {
            "horizon": f"{h}Q", "model": model_name,
            "RMSE": rmse(yv, fv), "MAE": mae(yv, fv),
            "MAPE": mape(yv, fv), "sMAPE": smape(yv, fv),
            "MASE": _mase, "YoY_acc": _yoy, "n": int(valid.sum()),
        }
        metric_rows.append(row)
        _yoy_str = f"{_yoy:.1f}" if not np.isnan(_yoy) else "N/A"
        print(f"  {f'{h}Q':>8}  {model_name:>12}  {row['RMSE']:>10.2f}  {row['MAE']:>10.2f}  "
              f"{row['MAPE']:>10.2f}  {row['sMAPE']:>10.2f}  {row['MASE']:>10.3f}  "
              f"{_yoy_str:>10}")

# ── 3. Diebold-Mariano tests ──────────────────────────────────────────────────
print("\n[3] Diebold-Mariano tests  (H0: equal accuracy)")
print(f"  {'Horizon':>8}  {'Loss':>10}  {'DM-stat':>10}  {'p-value':>10}  {'Better':>18}  {'Significant?':>15}")
print("  " + "-" * 78)
dm_rows = []
for h in horizons:
    sub = df_roll[df_roll["horizon"] == h].dropna(subset=["fc_sarima", "fc_naive"])
    if len(sub) < 5:
        continue
    y  = sub["actual"].values
    e1 = y - sub["fc_sarima"].values
    e2 = y - sub["fc_naive"].values
    for loss_name in ["squared", "absolute"]:
        dm_stat, p_val, better = diebold_mariano(e1, e2, h=h, loss=loss_name)
        sig = "YES (**)" if p_val < 0.05 else ("marginal (*)" if p_val < 0.10 else "NO")
        dm_rows.append({
            "horizon": f"{h}Q", "loss": loss_name, "dm_stat": dm_stat,
            "p_value": p_val, "better": better, "significant": sig,
        })
        print(f"  {f'{h}Q':>8}  {loss_name:>10}  {dm_stat:>10.3f}  {p_val:>10.4f}  "
              f"{better:>18}  {sig:>15}")

# ── 4. Training-window sensitivity ───────────────────────────────────────────
print("\n[4] Training-window sensitivity analysis")
print("  (Tests whether results change with training window size)")
if N >= 16:
    train_fracs = [0.5, 0.6, 0.7, 0.8]
    print(f"  {'Train%':>8}  {'Train N':>8}  {'Test N':>8}  "
          f"{'SARIMA RMSE':>14}  {'Naive RMSE':>12}")
    print("  " + "-" * 58)
    for frac in train_fracs:
        cutoff = max(8, int(N * frac))
        y_tr   = series[:cutoff]
        y_te   = series[cutoff:]
        if len(y_te) < 2:
            continue
        try:
            m1 = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,0,1,4),
                         enforce_stationarity=False, enforce_invertibility=False)
            r1 = m1.fit(disp=False, maxiter=200)
            fc_sarima = r1.forecast(steps=len(y_te))
        except Exception:
            fc_sarima = np.full(len(y_te), np.nan)
        fc_naive = np.array([
            series[cutoff + i - 4] if cutoff + i - 4 >= 0 else np.nan
            for i in range(len(y_te))
        ])
        valid = ~np.isnan(fc_sarima) & ~np.isnan(fc_naive)
        if valid.sum() < 2:
            continue
        print(f"  {frac*100:.0f}%{'':>5}  {cutoff:>8}  {len(y_te):>8}  "
              f"{rmse(y_te[valid], fc_sarima[valid]):>14.2f}  "
              f"{rmse(y_te[valid], fc_naive[valid]):>12.2f}")

# ── 5. Probabilistic forecast intervals ──────────────────────────────────────
print("\n[5] Probabilistic forecast intervals (coverage check)")
try:
    full_train = series[:-4]
    full_test  = series[-4:]
    m_full = SARIMAX(full_train, order=(1,1,1), seasonal_order=(1,0,1,4),
                     enforce_stationarity=False, enforce_invertibility=False)
    r_full = m_full.fit(disp=False, maxiter=300)
    fc_summary = r_full.get_forecast(steps=4).summary_frame(alpha=0.10)
    print(f"  Last-4-quarters forecast with 90% prediction interval:")
    print(f"  {'Q':>4}  {'Actual':>12}  {'Point fc':>12}  {'Lower 90%':>12}  {'Upper 90%':>12}  {'Covered?':>10}")
    print("  " + "-" * 65)
    covered = 0
    for i in range(4):
        actual = full_test[i]
        point  = fc_summary["mean"].iloc[i]
        lo     = fc_summary["mean_ci_lower"].iloc[i]
        hi     = fc_summary["mean_ci_upper"].iloc[i]
        in_ci  = lo <= actual <= hi
        covered += int(in_ci)
        print(f"  {i+1:>4}  {actual:>12.2f}  {point:>12.2f}  {lo:>12.2f}  {hi:>12.2f}  "
              f"{'YES ✓' if in_ci else 'NO ✗':>10}")
    print(f"\n  Empirical 90% PI coverage: {covered}/4 = {covered/4*100:.0f}%  (expected ≈ 90%)")
except Exception as e:
    print(f"  Could not compute PI: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
pd.DataFrame(metric_rows).to_csv(out / "t1_forecast_metrics.csv",  index=False)
pd.DataFrame(dm_rows).to_csv(    out / "t1_diebold_mariano.csv",   index=False)

print(f"\n  Results saved to:")
print(f"    {out / 't1_forecast_metrics.csv'}")
print(f"    {out / 't1_diebold_mariano.csv'}")

print("\n" + sep)
print("T1 COMPLETE")
print(sep)
