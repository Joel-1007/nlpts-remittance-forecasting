"""
DIAGNOSTIC D5 — Annual Forecasting Validity & YoY Justification
================================================================
Addresses reviewer concern:
  "The paper uses disaggregated annual data as if it were genuine quarterly
   observations. Quarterly MAPE is mathematically derived from the annual
   result, not an independent measure."

This diagnostic provides four proofs:
  [1] Annual aggregation & N=17 training set summary
  [2] IGARCH verification  (α + β ≈ 1.000 — permanent shock propagation)
  [3] YoY Accuracy audit   (directional hit-rate on annual step changes)
  [4] Broadcasting transparency — quantifies how quarterly MAPE relates
      to annual MAPE, and confirms it should be reported as DERIVED.

HOW TO USE ON KAGGLE:
  Requires (in /kaggle/working/):
    inward_quarterly_seasonal.csv   — raw quarterly series
    phase8_predictions.csv          — final model vs actual
    phase8_garch_params.json        — stored GARCH α, β, ω
    cell9_predictions.csv           — optional (multi-model comparison)
    baseline_forecasts.csv          — SARIMA baseline for DM comparison

Output: d5_annual_yoy_justification.csv
"""

import subprocess, sys
try:
    import statsmodels
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])

import pandas as pd
import numpy as np
import json, warnings
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

sep  = "=" * 72
sep2 = "-" * 72

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

# ── Required files ─────────────────────────────────────────────────────────────
quarterly_path  = find("inward_quarterly_seasonal.csv") or find("features_combined.csv")
phase8_pred     = find("phase8_predictions.csv")
garch_path      = find("phase8_garch_params.json")
cell9_path      = find("cell9_predictions.csv")
baseline_path   = find("baseline_forecasts.csv")

if quarterly_path is None:
    raise FileNotFoundError(
        "D5 requires inward_quarterly_seasonal.csv (or features_combined.csv).\n"
        "Upload your pipeline output to /kaggle/working/ and rerun."
    )

print(sep)
print("D5 — ANNUAL FORECASTING VALIDITY & YOY JUSTIFICATION")
print("     Mode: REAL DATA")
print(sep)

# ── Load quarterly series ──────────────────────────────────────────────────────
df_q = pd.read_csv(quarterly_path)
df_q["date"] = pd.to_datetime(df_q.get("date", df_q.get("quarter", None)), errors="coerce")
df_q = df_q.sort_values("date").dropna(subset=["inward_flow"])
df_q["year"] = df_q["date"].dt.year
print(f"\n  Quarterly series loaded: {len(df_q)} rows, {df_q['year'].min()}–{df_q['year'].max()}")

# ── Aggregate to annual ────────────────────────────────────────────────────────
df_a = df_q.groupby("year").agg(
    inward_flow_annual=("inward_flow", "sum"),
    n_quarters=("inward_flow", "count")
).reset_index()
# Only keep years with 4 complete quarters
df_a = df_a[df_a["n_quarters"] == 4].copy()
df_a["yoy_change"]  = df_a["inward_flow_annual"].diff()
df_a["yoy_pct_chg"] = df_a["inward_flow_annual"].pct_change() * 100
df_a = df_a.dropna(subset=["yoy_change"])  # drop first year (no prior)

N_annual = len(df_a)

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("[1] Annual Training Set Summary  (addresses N=17 claim)")
print(sep)

print(f"\n  Complete annual observations (4 full quarters): {N_annual}")
print(f"  Year range  : {int(df_a['year'].min())} – {int(df_a['year'].max())}")
print(f"\n  YoY step-change distribution:")
print(f"    Mean   : ${df_a['yoy_change'].mean():>12,.2f} M")
print(f"    Std    : ${df_a['yoy_change'].std():>12,.2f} M")
print(f"    Min    : ${df_a['yoy_change'].min():>12,.2f} M")
print(f"    Max    : ${df_a['yoy_change'].max():>12,.2f} M")
print(f"    Median : ${df_a['yoy_change'].median():>12,.2f} M")

# Positive vs negative transitions
n_pos = (df_a["yoy_change"] > 0).sum()
n_neg = (df_a["yoy_change"] < 0).sum()
print(f"\n  Directional split:")
print(f"    Growth years    : {n_pos}  ({n_pos/N_annual*100:.1f}%)")
print(f"    Contraction yrs : {n_neg}  ({n_neg/N_annual*100:.1f}%)")
print(f"\n  Annual series table:")
print(f"  {'Year':>6}  {'Annual Flow (M)':>18}  {'YoY Change (M)':>16}  {'YoY %':>8}")
print(f"  {sep2}")
for _, row in df_a.iterrows():
    print(f"  {int(row.year):>6}  {row.inward_flow_annual:>18,.2f}  "
          f"{row.yoy_change:>+16,.2f}  {row.yoy_pct_chg:>+7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("[2] GARCH Parameter Verification  (α + β ≈ 1.000 = IGARCH)")
print(sep)

if garch_path:
    with open(garch_path, encoding="utf-8") as fh:
        garch_params = json.load(fh)
    print(f"\n  Source: {garch_path}")
    print(f"  Raw parameters: {json.dumps(garch_params, indent=4)}")

    # Try common key names
    alpha = (garch_params.get("alpha") or garch_params.get("alpha[1]") or
             garch_params.get("ARCH") or garch_params.get("arch"))
    beta  = (garch_params.get("beta")  or garch_params.get("beta[1]")  or
             garch_params.get("GARCH") or garch_params.get("garch"))
    omega = (garch_params.get("omega") or garch_params.get("Omega") or
             garch_params.get("const") or 0.0)

    if alpha is not None and beta is not None:
        alpha, beta, omega = float(alpha), float(beta), float(omega)
        ab_sum = alpha + beta
        print(f"\n  ω (intercept) : {omega:.6f}")
        print(f"  α (ARCH term) : {alpha:.6f}")
        print(f"  β (GARCH term): {beta:.6f}")
        print(f"  α + β         : {ab_sum:.6f}")

        if abs(ab_sum - 1.0) < 0.02:
            print(f"\n  ✓ IGARCH confirmed (α + β = {ab_sum:.4f} ≈ 1.000)")
            print(f"    Economic interpretation:")
            print(f"      Shocks to EPU volatility are PERMANENT — they do not decay.")
            print(f"      A volatility spike in Q1 (e.g., COVID onset) remains 'active'")
            print(f"      through Q4, effectively anchoring the gate at annual resolution.")
            print(f"      This justifies annual-level interpretation of the GARCH gate.")
        elif ab_sum < 1.0:
            halflife = -np.log(2) / np.log(ab_sum)
            print(f"\n  ⚠  Stationary GARCH (α + β = {ab_sum:.4f} < 1.000)")
            print(f"     Shock half-life ≈ {halflife:.1f} periods")
            print(f"     Shocks decay — not IGARCH. This weakens the annual persistence claim.")
        else:
            print(f"\n  ⚠  α + β = {ab_sum:.4f} > 1.000 — explosive process. Check parameters.")
    else:
        print(f"  ⚠  Could not extract alpha/beta from JSON. Keys found: {list(garch_params.keys())}")
        print(f"     Manually verify: α + β should ≈ 1.000 for IGARCH property.")
else:
    print("\n  phase8_garch_params.json not found in /kaggle/working/")
    print("  ⚠  Cannot verify IGARCH property without stored parameters.")

    # Re-estimate from EPU series as fallback
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from arch import arch_model
        if "EPU_Index" in df_q.columns:
            epu = df_q["EPU_Index"].dropna().values
            gm  = arch_model(epu, vol="GARCH", p=1, q=1, dist="normal")
            gf  = gm.fit(disp="off")
            params = gf.params
            alpha_re = params.get("alpha[1]", params.get("alpha", np.nan))
            beta_re  = params.get("beta[1]",  params.get("beta",  np.nan))
            print(f"\n  Re-estimated from EPU_Index series:")
            print(f"    α = {alpha_re:.6f}   β = {beta_re:.6f}   α+β = {alpha_re+beta_re:.6f}")
    except Exception as e:
        print(f"  (arch re-estimation skipped: {e})")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("[3] YoY Accuracy Audit")
print(sep)

def yoy_accuracy_report(y_actual, y_pred, model_name, y_annual_actual=None):
    """
    Computes YoY accuracy: fraction of annual transitions where the
    model correctly predicted growth vs. contraction direction.
    y_actual, y_pred: arrays of annual values (already aggregated).
    """
    if len(y_actual) < 2:
        print(f"  {model_name}: too few points for YoY accuracy.")
        return np.nan
    delta_actual = np.diff(y_actual)
    delta_pred   = np.diff(y_pred)
    hits  = (np.sign(delta_actual) == np.sign(delta_pred))
    acc   = hits.mean()
    rmse  = np.sqrt(np.mean((y_actual - y_pred)**2))
    mae_  = np.mean(np.abs(y_actual - y_pred))
    mape_ = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-9))) * 100
    ss_res = np.sum((y_actual - y_pred)**2)
    ss_tot = np.sum((y_actual - y_actual.mean())**2)
    r2    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"\n  {model_name}")
    print(f"    N (annual)      : {len(y_actual)}")
    print(f"    YoY Accuracy    : {acc*100:.1f}%  ({hits.sum()}/{len(hits)} correct directions)")
    print(f"    Annual RMSE     : {rmse:>12,.2f} M")
    print(f"    Annual MAE      : {mae_:>12,.2f} M")
    print(f"    Annual MAPE     : {mape_:>8.2f}%")
    print(f"    R²              : {r2:.4f}")
    return acc

results_summary = []

# ── Load phase8 predictions ────────────────────────────────────────────────────
if phase8_pred:
    df_p8 = pd.read_csv(phase8_pred)
    df_p8["date"] = pd.to_datetime(df_p8.get("date", df_p8.columns[0]), errors="coerce")
    df_p8 = df_p8.sort_values("date").dropna(subset=["actual"])
    df_p8["year"] = df_p8["date"].dt.year

    # Aggregate predictions to annual
    agg_cols = {"actual": "sum"}
    model_cols = [c for c in df_p8.columns if c not in ["date","quarter","year","covid_period","actual"]]
    for c in model_cols:
        try:
            df_p8[c] = pd.to_numeric(df_p8[c], errors="coerce")
            agg_cols[c] = "sum"
        except Exception:
            pass

    df_p8_a = df_p8.groupby("year").agg(agg_cols).reset_index()
    df_p8_a = df_p8_a[df_p8_a.groupby("year")["actual"].transform("count") > 0]

    y_act = df_p8_a["actual"].values
    for col in model_cols[:6]:  # limit to first 6 models to keep output manageable
        if col in df_p8_a.columns and df_p8_a[col].notna().sum() >= 2:
            y_pred = df_p8_a[col].values
            acc = yoy_accuracy_report(y_act, y_pred, f"phase8 → {col}")
            results_summary.append({"model": col, "yoy_accuracy": acc,
                                     "source": "phase8_predictions.csv"})
else:
    print("\n  phase8_predictions.csv not found — skipping phase8 YoY audit.")

# ── Load cell9 predictions ─────────────────────────────────────────────────────
if cell9_path:
    df_c9 = pd.read_csv(cell9_path)
    df_c9["date"] = pd.to_datetime(df_c9.get("date", df_c9.columns[0]), errors="coerce")
    df_c9 = df_c9.sort_values("date").dropna(subset=["actual"])
    df_c9["year"] = df_c9["date"].dt.year

    agg9 = {"actual": "sum"}
    model9 = [c for c in df_c9.columns
              if c not in ["date","quarter","year","covid_period","actual","EPU_garch_vol"]]
    for c in model9:
        try:
            df_c9[c] = pd.to_numeric(df_c9[c], errors="coerce")
            agg9[c] = "sum"
        except Exception:
            pass

    df_c9_a = df_c9.groupby("year").agg(agg9).reset_index()
    y_act9  = df_c9_a["actual"].values
    for col in model9[:8]:
        if col in df_c9_a.columns and df_c9_a[col].notna().sum() >= 2:
            y_pred9 = df_c9_a[col].values
            acc9 = yoy_accuracy_report(y_act9, y_pred9, f"cell9 → {col}")
            results_summary.append({"model": col, "yoy_accuracy": acc9,
                                     "source": "cell9_predictions.csv"})

# ── SARIMA baseline YoY ────────────────────────────────────────────────────────
if baseline_path:
    df_bl = pd.read_csv(baseline_path)
    df_bl["date"] = pd.to_datetime(df_bl.get("date", df_bl.columns[0]), errors="coerce")
    df_bl = df_bl.sort_values("date").dropna(subset=["actual"])
    df_bl["year"] = df_bl["date"].dt.year
    for col in ["sarima", "SARIMA_baseline", "sarima_baseline"]:
        if col in df_bl.columns:
            df_bl_a = df_bl.groupby("year").agg(
                actual=(   "actual", "sum"),
                sarima_fc=(col,      "sum"),
            ).reset_index()
            acc_bl = yoy_accuracy_report(
                df_bl_a["actual"].values, df_bl_a["sarima_fc"].values,
                "SARIMA Baseline")
            results_summary.append({"model": "SARIMA Baseline", "yoy_accuracy": acc_bl,
                                     "source": "baseline_forecasts.csv"})
            break

# ── Summary table ──────────────────────────────────────────────────────────────
if results_summary:
    df_summ = pd.DataFrame(results_summary).dropna(subset=["yoy_accuracy"])
    df_summ = df_summ.sort_values("yoy_accuracy", ascending=False)
    print(f"\n  {'Model':<35} {'YoY Acc':>9}  {'Source'}")
    print(f"  {sep2}")
    for _, r in df_summ.iterrows():
        print(f"  {r['model']:<35} {r['yoy_accuracy']*100:>8.1f}%  {r['source']}")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("[4] Broadcasting Transparency Analysis")
print(sep)

print("""
  ── What "anchor-based broadcasting" actually means ────────────────────
  When the model produces one annual forecast Ŷ_year, broadcasting
  distributes it into four quarterly estimates using prior-year
  seasonal proportions (s_q):

      ŷ_q = Ŷ_year × s_q / Σ s_q    where s_q = y_{q,year-1} / Y_{year-1}

  The four quarterly predictions are MATHEMATICALLY DERIVED from the
  single annual prediction — they are not four independent forecasts.
""")

# Compute seasonal proportions from historical data
df_q["q_label"] = pd.to_datetime(df_q["date"].dt.to_period("Q").astype(str)).dt.quarter
seasonal_props = (
    df_q.groupby(["year","q_label"])["inward_flow"]
    .sum()
    .unstack("q_label")
)
seasonal_props = seasonal_props.div(seasonal_props.sum(axis=1), axis=0)

print(f"  Historical seasonal proportions (fraction of annual total per quarter):")
mean_props = seasonal_props.mean()
std_props  = seasonal_props.std()
print(f"  {'Quarter':>8}  {'Mean share':>12}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'-'*52}")
for q in [1, 2, 3, 4]:
    if q in mean_props.index:
        print(f"  {'Q'+str(q):>8}  {mean_props[q]:>11.1%}  {std_props[q]:>7.1%}  "
              f"{seasonal_props[q].min():>7.1%}  {seasonal_props[q].max():>7.1%}")

print(f"""
  ── What this means for MAPE reporting ─────────────────────────────────
  If Annual MAPE  = E_a  then
  Quarterly MAPE ≈ E_a + seasonal_std  (not an independent number).

  The quarterly MAPE will ALWAYS be close to the annual MAPE because
  the seasonality is derived from prior-year actuals (which are known).

  ── Reviewer-Safe Recommendation ───────────────────────────────────────
  ✓ REPORT  : Annual RMSE, Annual MAE, Annual MAPE, YoY Accuracy
  ✓ DISCLOSE: "Quarterly figures are disaggregated from annual forecasts
               using prior-year seasonal anchors; they are not independent
               quarterly predictions."
  ✗ AVOID   : Claiming "Quarterly MAPE = X%" as if independently achieved.
""")

# Quantify: what is the quarterly MAPE vs annual MAPE if we use broadcasting?
if phase8_pred and "df_p8" in dir():
    try:
        # Pick first numeric model column
        mc = [c for c in model_cols if df_p8[c].notna().sum() > 4][0]
        # Annual MAPE
        a_act  = df_p8_a["actual"].values
        a_pred = df_p8_a[mc].values
        annual_mape = np.mean(np.abs((a_act - a_pred) / (a_act + 1e-9))) * 100

        # Quarterly MAPE (raw, before broadcasting)
        q_act  = df_p8["actual"].values
        q_pred = df_p8[mc].values
        valid  = ~np.isnan(q_pred)
        quarterly_mape = np.mean(np.abs((q_act[valid] - q_pred[valid]) / (q_act[valid] + 1e-9))) * 100

        print(f"  Quantified comparison (model: {mc}):")
        print(f"    Annual  MAPE : {annual_mape:.2f}%")
        print(f"    Quarterly MAPE: {quarterly_mape:.2f}%")
        print(f"    Difference    : {abs(annual_mape - quarterly_mape):.2f} pp")
        if abs(annual_mape - quarterly_mape) < 3:
            print(f"    → Quarterly MAPE closely mirrors Annual MAPE (< 3pp gap).")
            print(f"      This CONFIRMS they are not independent metrics.")
    except Exception as e:
        print(f"  (MAPE comparison skipped: {e})")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

# Annual series with YoY info
annual_out = df_a[["year","inward_flow_annual","yoy_change","yoy_pct_chg"]].copy()
annual_out.to_csv(out / "d5_annual_yoy_justification.csv", index=False)
print(f"\n  Annual series saved → {out / 'd5_annual_yoy_justification.csv'}")

# Model summary
if results_summary:
    pd.DataFrame(results_summary).to_csv(out / "d5_model_yoy_accuracy.csv", index=False)
    print(f"  Model YoY accuracy → {out / 'd5_model_yoy_accuracy.csv'}")

print(f"\n{sep}")
print("D5 COMPLETE")
print(sep)
print("""
  ── Repositioning Narrative for Reviewer ────────────────────────────────
  This study is an ANNUAL forecasting study that leverages high-frequency
  (monthly/quarterly) sentiment as a leading indicator. The 17 genuine
  annual observations are the primary unit of analysis. Quarterly figures
  in the paper are presented for granularity only and are explicitly
  derived from annual forecasts via prior-year seasonal decomposition.
  The YoY Accuracy metric is the study's primary claim of success.
""")
