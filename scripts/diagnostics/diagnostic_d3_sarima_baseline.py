"""
DIAGNOSTIC D3 — SARIMA Baseline Justification
==============================================
Reviewer concern: SARIMA baseline chosen without justification.
Required tests: Grid exploration, AIC/BIC table, Ljung-Box on residuals.

HOW TO USE ON KAGGLE:
  • Paste this entire cell into a new Kaggle code cell and run it.
  • Requires india_remittance_processed.csv (or india_remittance_final.csv)
    in /kaggle/working/ or /kaggle/input/.

Output: printed table + d3_sarima_selection.csv
"""

import subprocess, sys
try:
    import statsmodels
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])

import pandas as pd
import numpy as np
import warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import shapiro, jarque_bera

sep = "=" * 72

SEARCH_DIRS = [
    "/kaggle/working/",
    "/kaggle/input/",
    "./",
    "../",
]

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
        "D3 could not find a quarterly remittance CSV.\n"
        "Expected: inward_quarterly_seasonal.csv or features_combined.csv\n"
        "in /kaggle/working/ or /kaggle/input/."
    )

print(sep)
print("D3 — SARIMA BASELINE JUSTIFICATION")
print("     Mode: REAL DATA")
print(sep)

df = pd.read_csv(combined_path)
print(f"  Loaded: {combined_path}  (columns: {list(df.columns)})")

# inward_flow is confirmed present in inward_quarterly_seasonal.csv and features_combined.csv
if "inward_flow" not in df.columns:
    print(f"❌  'inward_flow' column not found. Available: {list(df.columns)}")
    raise SystemExit(1)

sort_col = next((c for c in ["quarter", "date", "year"] if c in df.columns), df.columns[0])
df = df.sort_values(sort_col)

series = df["inward_flow"].dropna().values.astype(float)
N = len(series)
print(f"\n  Series length : {N} observations")
print(f"  Series mean   : {series.mean():,.2f}")
print(f"  Series std    : {series.std():,.2f}")

# ── 1. Stationarity tests ────────────────────────────────────────────────────
print("\n[1] Stationarity tests")
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag="AIC")
print(f"  ADF test:  stat={adf_stat:.4f}, p={adf_p:.4f}  "
      f"→ {'stationary (p<0.05)' if adf_p < 0.05 else 'NON-stationary (p≥0.05)'}")

try:
    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")
    print(f"  KPSS test: stat={kpss_stat:.4f}, p={kpss_p:.4f}  "
          f"→ {'non-stationary (p<0.05)' if kpss_p < 0.05 else 'stationary (p≥0.05)'}")
except Exception as e:
    print(f"  KPSS: skipped ({e})")

d_series = np.diff(series)
adf2_stat, adf2_p, *_ = adfuller(d_series, autolag="AIC")
print(f"  ADF (d=1): stat={adf2_stat:.4f}, p={adf2_p:.4f}  "
      f"→ {'stationary' if adf2_p < 0.05 else 'still non-stationary'}")

best_d = 0 if adf_p < 0.05 else 1
print(f"  ➜ Selected d = {best_d}")

# ── 2. Grid search ───────────────────────────────────────────────────────────
print("\n[2] SARIMA grid search  (p∈[0..2], q∈[0..2], P∈[0..1], Q∈[0..1], m=4)")

m = 4  # quarterly seasonality

# ── Shortcut: use pre-existing grid results if available ──────────────────────
existing_grid = find("sarima_grid_search.csv")
if existing_grid:
    print(f"  ✓ Loading pre-computed grid from: {existing_grid}")
    df_res = pd.read_csv(existing_grid)
    df_res.columns = [c.strip() for c in df_res.columns]
    print(f"  Grid columns: {list(df_res.columns)}")

    # Normalise AIC/BIC/HQIC column names (case-insensitive)
    col_map = {c.lower(): c for c in df_res.columns}
    for target, variants in [("AIC",["aic","Aic"]), ("BIC",["bic","Bic"]),
                              ("HQIC",["hqic","Hqic","HQIC"])]:
        if target not in df_res.columns:
            for v in variants:
                if v in col_map:
                    df_res = df_res.rename(columns={col_map[v]: target})
                    break
    # Add HQIC column if still missing
    if "HQIC" not in df_res.columns:
        df_res["HQIC"] = np.nan

    df_res = df_res.sort_values("AIC")
    if "converged" not in df_res.columns:
        df_res["converged"] = True

    # Extract p, d, q from an 'order' string column if individual cols are absent
    import re
    for col_p, col_d, col_q in [("p","d","q")]:
        if col_p not in df_res.columns:
            if "order" in df_res.columns:
                def parse_pdq(o):
                    nums = list(map(int, re.findall(r"\d+", str(o))))
                    return nums[:3] if len(nums) >= 3 else [0, best_d, 0]
                pdq = pd.DataFrame(df_res["order"].apply(parse_pdq).tolist(),
                                   index=df_res.index, columns=["p","d","q"])
                df_res = pd.concat([df_res, pdq], axis=1)
            else:
                df_res["p"] = 1
                df_res["d"] = best_d
                df_res["q"] = 1

    # Extract P, Q from seasonal order string if absent
    for col in ["P", "Q"]:
        if col not in df_res.columns:
            if "seasonal_order" in df_res.columns:
                idx = 0 if col == "P" else 2
                df_res[col] = df_res["seasonal_order"].apply(
                    lambda o: list(map(int, re.findall(r"\d+", str(o))))[idx]
                    if len(re.findall(r"\d+", str(o))) > idx else 0)
            else:
                df_res[col] = 0
    if "D" not in df_res.columns:
        df_res["D"] = 0
else:
    max_p = 2 if N >= 20 else 1
    max_q = 2 if N >= 20 else 1
    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    P_range = range(0, 2)
    Q_range = range(0, 2)

    results = []
    total_combinations = len(p_range) * len(q_range) * len(P_range) * len(Q_range)
    print(f"  Testing {total_combinations} combinations...")

    for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
        try:
            mod = SARIMAX(
                series,
                order=(p, best_d, q),
                seasonal_order=(P, 0, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=200)
            results.append({
                "p": p, "d": best_d, "q": q,
                "P": P, "D": 0, "Q": Q, "m": m,
                "AIC":  round(res.aic,  2),
                "BIC":  round(res.bic,  2),
                "HQIC": round(res.hqic, 2),
                "converged": (res.mle_retvals.get("warnflag", 0) == 0
                              if hasattr(res.mle_retvals, "get") else True),
            })
        except Exception:
            pass

    df_res = pd.DataFrame(results).sort_values("AIC")

print(f"\n  Top 10 models by AIC:")
has_hqic = "HQIC" in df_res.columns and df_res["HQIC"].notna().any()
if has_hqic:
    print(f"  {'Order':<18} {'Seasonal':<16} {'AIC':>10}  {'BIC':>10}  {'HQIC':>10}  Conv")
else:
    print(f"  {'Order':<18} {'Seasonal':<16} {'AIC':>10}  {'BIC':>10}  Conv")
print("  " + "-" * 70)
for _, row in df_res.head(10).iterrows():
    # safely cast p/d/q/P/Q — handle float or string values
    def _int(v):
        try: return int(float(v))
        except: return 0
    order    = f"({_int(row.get('p',0))},{_int(row.get('d',1))},{_int(row.get('q',0))})"
    seasonal = f"({_int(row.get('P',0))},0,{_int(row.get('Q',0))},{m})"
    conv     = "✓" if row.get("converged", True) else "✗"
    if has_hqic:
        hqic_val = row.get("HQIC", np.nan)
        hqic_str = f"{hqic_val:>10.2f}" if pd.notna(hqic_val) else f"{'N/A':>10}"
        print(f"  {order:<18} {seasonal:<16} {row['AIC']:>10.2f}  {row['BIC']:>10.2f}  "
              f"{hqic_str}  {conv}")
    else:
        print(f"  {order:<18} {seasonal:<16} {row['AIC']:>10.2f}  {row['BIC']:>10.2f}  {conv}")

best = df_res.iloc[0]
def _int(v):
    try: return int(float(v))
    except: return 0
print(f"\n  ★ Best SARIMA: ({_int(best.get('p',0))},{_int(best.get('d',1))},{_int(best.get('q',0))})"
      f"({_int(best.get('P',0))},0,{_int(best.get('Q',0))},{m})")
print(f"    AIC={best['AIC']:.2f}  BIC={best['BIC']:.2f}")

# ── 3. Residual diagnostics on best model ───────────────────────────────────
print("\n[3] Residual diagnostics (best model)")
try:
    best_p = _int(best.get("p", 0))
    best_d_val = _int(best.get("d", 1))
    best_q = _int(best.get("q", 0))
    best_P = _int(best.get("P", 0))
    best_Q = _int(best.get("Q", 0))
    best_mod = SARIMAX(
        series,
        order=(best_p, best_d_val, best_q),
        seasonal_order=(best_P, 0, best_Q, m),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    best_fit = best_mod.fit(disp=False, maxiter=300)
    resids   = best_fit.resid

    # Ljung-Box
    lags_to_test = [4, 8, 12]
    lb_results = acorr_ljungbox(resids, lags=lags_to_test, return_df=True)
    print(f"  Ljung-Box test on residuals:")
    print(f"  {'Lag':>6}  {'Statistic':>12}  {'p-value':>12}  {'H0: No autocorr.':>25}")
    print("  " + "-" * 62)
    for lag in lags_to_test:
        if lag in lb_results.index:
            stat = lb_results.loc[lag, "lb_stat"]
            pval = lb_results.loc[lag, "lb_pvalue"]
            result = "FAIL-REJECT (autocorr. present)" if pval < 0.05 else "Pass (no autocorr.)"
            print(f"  {lag:>6}  {stat:>12.4f}  {pval:>12.4f}  {result}")

    # Normality of residuals
    _, sw_p = shapiro(resids[:min(len(resids), 200)])
    jb_stat, jb_p = jarque_bera(resids)
    print(f"\n  Shapiro-Wilk p-value : {sw_p:.4f}  "
          f"→ {'Normal' if sw_p >= 0.05 else 'Non-normal'} residuals")
    print(f"  Jarque-Bera p-value  : {jb_p:.4f}  "
          f"→ {'Normal' if jb_p >= 0.05 else 'Non-normal'} residuals")

    print(f"\n  AIC of best model    : {best_fit.aic:.2f}")
    print(f"  BIC of best model    : {best_fit.bic:.2f}")
except Exception as e:
    print(f"  ❌  Could not fit best model: {e}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
df_res.to_csv(out / "d3_sarima_selection.csv", index=False)
print(f"\n  Full grid table saved → {out / 'd3_sarima_selection.csv'}")

print("\n" + sep)
print("D3 COMPLETE — Use the table above to justify SARIMA order choice.")
print(sep)
