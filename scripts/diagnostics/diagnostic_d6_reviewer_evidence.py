"""
DIAGNOSTIC D6 — Four Reviewer Evidence Proofs
==============================================
[1] GARCH Conditional Volatility & Gate Persistence
[2] SHAP Feature Attribution (Annual Level)
[3] Residual Whitening: SARIMA vs Best Pipeline Model
[4] Historical Seasonal Anchors (Q1-Q4 Proof)

Requires in /kaggle/working/:
  inward_quarterly_seasonal.csv
  phase8_epu_vol_train.csv + phase8_epu_vol_test.csv
  phase8_features_train.csv
  phase8_predictions.csv  (or cell9_predictions.csv)
  baseline_forecasts.csv
"""

import subprocess, sys
for pkg in ["statsmodels", "shap", "xgboost"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import pandas as pd
import numpy as np
import json, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
warnings.filterwarnings("ignore")

sep  = "=" * 72
sep2 = "-" * 72
out  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

SEARCH_DIRS = ["/kaggle/working/", "/kaggle/input/", "./", "../"]

def find(f):
    for d in SEARCH_DIRS:
        p = Path(d) / f
        if p.exists():
            return str(p)
    inp = Path("/kaggle/input")
    if inp.exists():
        for sub in inp.iterdir():
            p = sub / f
            if p.exists():
                return str(p)
    return None

qseries_path = find("inward_quarterly_seasonal.csv") or find("features_combined.csv")
feat_train   = find("phase8_features_train.csv")
pred_path    = find("phase8_predictions.csv") or find("cell9_predictions.csv")

print(sep)
print("D6 — FOUR REVIEWER EVIDENCE PROOFS")
print(sep)

# =============================================================================
# [1] GARCH CONDITIONAL VOLATILITY & GATE PERSISTENCE
# =============================================================================
print(f"\n{sep}")
print("[1] GARCH Conditional Volatility & Gate Persistence")
print(sep)

dfs_vol = []
for fname in ["phase8_epu_vol_train.csv", "phase8_epu_vol_test.csv"]:
    p = find(fname)
    if p:
        dfs_vol.append(pd.read_csv(p))
if not dfs_vol and feat_train:
    dfs_vol.append(pd.read_csv(feat_train))

resid_sarima = None   # initialise here so section 3 can reference it

if dfs_vol:
    df_vol  = pd.concat(dfs_vol, ignore_index=True)
    vol_col = next((c for c in ["EPU_garch_vol","garch_vol","cond_vol","h_t"]
                    if c in df_vol.columns), None)
    if vol_col is None:
        print(f"  ⚠  No volatility column found. Available: {list(df_vol.columns)}")
    else:
        sigma = df_vol[vol_col].dropna().values
        gate  = 1 / (1 + np.exp(-sigma))
        ac1   = pd.Series(sigma).autocorr(lag=1)
        ac4   = pd.Series(sigma).autocorr(lag=4)

        print(f"\n  EPU conditional volatility ({vol_col}), N={len(sigma)}")
        print(f"    Mean σ_t : {sigma.mean():.4f}")
        print(f"    Max  σ_t : {sigma.max():.4f}  (crisis peak)")
        print(f"    AC(1)    : {ac1:.4f}")
        print(f"    AC(4)    : {ac4:.4f}  {'✓ IGARCH — persists >1 year' if ac4 > 0.7 else '⚠ decays < 4 quarters'}")
        print(f"\n  Gate g_t = sigmoid(σ_t):")
        print(f"    Mean gate : {gate.mean():.4f}")
        print(f"    Gate > 0.7 (sentiment amplified): {(gate > 0.7).sum()} periods ({(gate > 0.7).mean()*100:.1f}%)")
        print(f"    Gate < 0.4 (SARIMA dominates)  : {(gate < 0.4).sum()} periods ({(gate < 0.4).mean()*100:.1f}%)")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(sigma, color="#e74c3c", lw=1.5, label="EPU Conditional Volatility σ_t")
        axes[0].axhline(sigma.mean(), color="gray", ls="--", alpha=0.6, label="Mean")
        axes[0].set_title("EPU Conditional Volatility (IGARCH property)", fontsize=13)
        axes[0].set_ylabel("σ_t"); axes[0].legend()
        axes[1].plot(gate, color="#2ecc71", lw=1.5, label="Gate g_t = sigmoid(σ_t)")
        axes[1].axhline(0.5, color="gray", ls="--", alpha=0.5)
        axes[1].set_title("Gate Activation  [0 = SARIMA dominates, 1 = Sentiment dominates]")
        axes[1].set_ylabel("g_t"); axes[1].legend()
        plt.tight_layout()
        fig.savefig(out / "d6_garch_gate_persistence.png", dpi=150)
        plt.close()
        print(f"\n  Saved → {out / 'd6_garch_gate_persistence.png'}")
else:
    print("  EPU volatility files not found — skipping section 1.")

# =============================================================================
# [2] SHAP FEATURE ATTRIBUTION (ANNUAL LEVEL)
# =============================================================================
print(f"\n{sep}")
print("[2] SHAP Feature Attribution (Annual Level)")
print(sep)

if qseries_path and feat_train:
    import xgboost as xgb
    import shap

    df_q = pd.read_csv(qseries_path)
    df_q["date"] = pd.to_datetime(df_q.get("date", df_q.get("quarter")), errors="coerce")
    df_q = df_q.sort_values("date")
    df_q["year"] = df_q["date"].dt.year
    df_ann = (df_q.groupby("year")["inward_flow"].sum()
              .reset_index().rename(columns={"inward_flow": "annual_flow"}))
    df_ann["yoy_change"] = df_ann["annual_flow"].diff()
    df_ann = df_ann.dropna(subset=["yoy_change"])

    df_ft = pd.read_csv(feat_train)
    n_q   = len(df_ft)
    start_year = int(df_q["year"].min())
    df_ft["year"] = [start_year + i // 4 for i in range(n_q)]

    num_cols = df_ft.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "year"]
    df_ft_a  = df_ft.groupby("year")[num_cols].mean().reset_index()
    df_shap  = df_ft_a.merge(df_ann[["year","yoy_change"]], on="year", how="inner").dropna()

    if len(df_shap) >= 5:
        FEAT_COLS = [c for c in num_cols if c in df_shap.columns]
        X = df_shap[FEAT_COLS].values
        y = df_shap["yoy_change"].values

        model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.1, random_state=42)
        model_xgb.fit(X, y)
        explainer = shap.TreeExplainer(model_xgb)
        shap_vals = explainer.shap_values(X)
        mean_abs  = np.abs(shap_vals).mean(axis=0)

        shap_df = (pd.DataFrame({"feature": FEAT_COLS, "mean_abs_shap": mean_abs})
                   .sort_values("mean_abs_shap", ascending=False).head(20))

        print(f"\n  N annual obs for SHAP: {len(df_shap)}")
        print(f"\n  Top-20 features by mean |SHAP|  (YoY Change target):")
        print(f"  {'Rank':>4}  {'Feature':<40}  {'Mean |SHAP|':>12}")
        print(f"  {sep2}")
        SENT_KEYS = ["sentiment","crisis","positive","prop","crisis_index"]
        for i, (_, row) in enumerate(shap_df.iterrows(), 1):
            tag = "  ← SENTIMENT" if any(k in row.feature for k in SENT_KEYS) else ""
            print(f"  {i:>4}  {row.feature:<40}  {row.mean_abs_shap:>12.4f}{tag}")

        top5 = shap_df.head(5)["feature"].tolist()
        sent_top5 = [f for f in top5 if any(k in f for k in SENT_KEYS)]
        if sent_top5:
            print(f"\n  ✓ Sentiment features in top-5: {sent_top5}")
            print(f"    → NLP pipeline adds measurable alpha over structural lags.")
        else:
            print(f"\n  ⚠  No sentiment features in top-5. Top-5: {top5}")

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#e74c3c" if any(k in f for k in SENT_KEYS) else "#3498db"
                  for f in shap_df["feature"]]
        ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1],
                color=colors[::-1])
        ax.set_xlabel("Mean |SHAP| Value")
        ax.set_title("Feature Attribution for Annual YoY Remittance Change\n"
                     "(Red = Sentiment/Crisis, Blue = Structural)", fontsize=12)
        plt.tight_layout()
        fig.savefig(out / "d6_shap_annual.png", dpi=150)
        plt.close()
        shap_df.to_csv(out / "d6_shap_annual.csv", index=False)
        print(f"\n  Saved → {out / 'd6_shap_annual.png'}  |  {out / 'd6_shap_annual.csv'}")
    else:
        print(f"  ⚠  Only {len(df_shap)} annual observations — too few for SHAP.")
else:
    print("  Required files not found — skipping SHAP section.")

# =============================================================================
# [3] RESIDUAL WHITENING: SARIMA vs BEST PIPELINE MODEL
# =============================================================================
print(f"\n{sep}")
print("[3] Residual Whitening: SARIMA vs Best Pipeline Model")
print(sep)

if qseries_path and pred_path:
    # ── SARIMA residuals on full series ──────────────────────────────────────
    df_qs = pd.read_csv(qseries_path)
    df_qs["date"] = pd.to_datetime(df_qs.get("date", df_qs.get("quarter")), errors="coerce")
    df_qs = df_qs.sort_values("date")
    series_full = df_qs["inward_flow"].dropna().values

    print("\n  Fitting SARIMA(1,1,1)(1,0,1,4) on full series...")
    try:
        m_sar = SARIMAX(series_full, order=(1,1,1), seasonal_order=(1,0,1,4),
                        enforce_stationarity=False, enforce_invertibility=False)
        r_sar = m_sar.fit(disp=False, maxiter=300)
        resid_sarima = r_sar.resid
        MAX_LAGS_SAR = min(16, len(resid_sarima) // 2 - 1)
        lb_sar = acorr_ljungbox(resid_sarima, lags=[l for l in [4,8,12] if l <= MAX_LAGS_SAR],
                                return_df=True)
        print(f"\n  SARIMA Residuals (N={len(resid_sarima)}) — Ljung-Box:")
        print(f"  {'Lag':>6}  {'LB Stat':>10}  {'p-value':>10}  {'Autocorr?':>15}")
        print(f"  {'-'*46}")
        for lag, row_lb in lb_sar.iterrows():
            flag = "YES ← bad" if row_lb["lb_pvalue"] < 0.05 else "No (good)"
            print(f"  {lag:>6}  {row_lb['lb_stat']:>10.4f}  {row_lb['lb_pvalue']:>10.4f}  {flag:>15}")
    except Exception as e:
        print(f"  SARIMA fit failed: {e}")
        resid_sarima = None

    # ── Best pipeline model (lowest residual std on test window) ─────────────
    df_pred = pd.read_csv(pred_path)
    df_pred["date"] = pd.to_datetime(df_pred.get("date", df_pred.columns[0]), errors="coerce")
    df_pred = df_pred.sort_values("date").dropna(subset=["actual"])
    SKIP = {"date","quarter","year","covid_period","actual","EPU_garch_vol"}
    model_cols_pred = [c for c in df_pred.columns if c not in SKIP]

    best_std, best_col = np.inf, None
    for c in model_cols_pred:
        try:
            r = (df_pred["actual"] - pd.to_numeric(df_pred[c], errors="coerce")).dropna()
            if len(r) >= 8 and r.std() < best_std:
                best_std, best_col = r.std(), c
        except Exception:
            pass

    print(f"\n  Available models: {model_cols_pred}")
    print(f"  Best (lowest residual std): {best_col}  (std={best_std:.2f} M)")

    if best_col:
        resid_best = (df_pred["actual"] - pd.to_numeric(
            df_pred[best_col], errors="coerce")).dropna().values
        N_best = len(resid_best)
        MAX_LAGS_BEST = min(16, N_best // 2 - 1)
        LB_LAGS_BEST  = [l for l in [4, 8, 12] if l <= N_best - 2 and l <= MAX_LAGS_BEST]

        lb_best = acorr_ljungbox(resid_best, lags=LB_LAGS_BEST, return_df=True)
        print(f"\n  {best_col} Residuals (N={N_best}) — Ljung-Box:")
        print(f"  {'Lag':>6}  {'LB Stat':>10}  {'p-value':>10}  {'Autocorr?':>15}")
        print(f"  {'-'*46}")
        for lag, row_lb in lb_best.iterrows():
            flag = "YES ← bad" if row_lb["lb_pvalue"] < 0.05 else "No (good) ✓"
            print(f"  {lag:>6}  {row_lb['lb_stat']:>10.4f}  {row_lb['lb_pvalue']:>10.4f}  {flag:>15}")

        print(f"\n  Residual std comparison:")
        if resid_sarima is not None:
            print(f"    SARIMA full series (N={len(resid_sarima)}): {resid_sarima.std():>10.2f} M")
        print(f"    {best_col} test window (N={N_best}): {resid_best.std():>10.2f} M")
        print(f"    (Note: SARIMA is full-sample in-sample; pipeline model is out-of-sample.)")

        # ACF/PACF plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        if resid_sarima is not None:
            plot_acf( resid_sarima, lags=MAX_LAGS_SAR, ax=axes[0][0],
                      title=f"SARIMA ACF  (lags≤{MAX_LAGS_SAR})")
            plot_pacf(resid_sarima, lags=MAX_LAGS_SAR, ax=axes[0][1],
                      title=f"SARIMA PACF (lags≤{MAX_LAGS_SAR})")
        else:
            for ax in axes[0]:
                ax.text(0.5, 0.5, "SARIMA not available", ha="center", va="center")

        plot_acf( resid_best, lags=MAX_LAGS_BEST, ax=axes[1][0],
                  title=f"{best_col} ACF  (lags≤{MAX_LAGS_BEST}, N={N_best})")
        plot_pacf(resid_best, lags=MAX_LAGS_BEST, ax=axes[1][1],
                  title=f"{best_col} PACF (lags≤{MAX_LAGS_BEST}, N={N_best})")

        plt.suptitle("Residual Whitening: SARIMA (top) vs Best Pipeline Model (bottom)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out / "d6_residual_acf_pacf.png", dpi=150)
        plt.close()
        print(f"\n  Saved → {out / 'd6_residual_acf_pacf.png'}")
    else:
        print(f"  ⚠  No usable model column found in {pred_path}.")
else:
    print("  Required files not found — skipping residual section.")

# =============================================================================
# [4] HISTORICAL SEASONAL ANCHORS (Q1-Q4 PROOF)
# =============================================================================
print(f"\n{sep}")
print("[4] Historical Seasonal Anchors (Quarterly Disaggregation Proof)")
print(sep)

if qseries_path:
    df_qs2 = pd.read_csv(qseries_path)
    df_qs2["date"]    = pd.to_datetime(df_qs2.get("date", df_qs2.get("quarter")), errors="coerce")
    df_qs2            = df_qs2.sort_values("date")
    df_qs2["year"]    = df_qs2["date"].dt.year
    df_qs2["quarter"] = df_qs2["date"].dt.quarter

    ann_tot = df_qs2.groupby("year")["inward_flow"].sum().rename("annual_total")
    df_qs2  = df_qs2.merge(ann_tot, on="year")
    df_qs2["q_share"] = df_qs2["inward_flow"] / df_qs2["annual_total"]

    full_years = df_qs2.groupby("year")["quarter"].count()
    full_years = full_years[full_years == 4].index
    df_qs2     = df_qs2[df_qs2["year"].isin(full_years)]

    seasonal = df_qs2.groupby("quarter")["q_share"].agg(
        mean="mean", std="std", min="min", max="max", n="count"
    )

    print(f"\n  Years with 4 complete quarters: {len(full_years)}"
          f"  ({int(df_qs2['year'].min())}–{int(df_qs2['year'].max())})\n")
    print(f"  {'Quarter':>8}  {'Mean%':>8}  {'Std%':>7}  {'Min%':>7}  {'Max%':>7}  {'CoV%':>7}")
    print(f"  {sep2}")
    for q in [1, 2, 3, 4]:
        if q in seasonal.index:
            r   = seasonal.loc[q]
            cov = r["std"] / r["mean"] * 100
            print(f"  {'Q'+str(q):>8}  {r['mean']*100:>7.1f}%  {r['std']*100:>6.1f}%  "
                  f"{r['min']*100:>6.1f}%  {r['max']*100:>6.1f}%  {cov:>6.1f}%")

    stable = all(seasonal.loc[q,"std"] / seasonal.loc[q,"mean"] < 0.10
                 for q in [1,2,3,4] if q in seasonal.index)
    if stable:
        print(f"\n  ✓ CoV < 10% for all quarters — seasonal shares are STABLE.")
        print(f"    Broadcasting is a deterministic accounting identity, not simulation.")
    else:
        unstable = [q for q in [1,2,3,4]
                    if q in seasonal.index and
                    seasonal.loc[q,"std"] / seasonal.loc[q,"mean"] >= 0.10]
        print(f"\n  ⚠  Q{unstable} have CoV ≥ 10% — disclose this variability.")

    # Year-by-year table
    pivot = (df_qs2.pivot_table(index="year", columns="quarter",
                                values="q_share", aggfunc="mean"))
    pivot.columns = [f"Q{int(c)}_share" for c in pivot.columns]
    print(f"\n  Year-by-year shares:")
    print(f"  {'Year':>6}  {'Q1%':>8}  {'Q2%':>8}  {'Q3%':>8}  {'Q4%':>8}  {'Sum%':>8}")
    print(f"  {sep2}")
    for year, row in pivot.iterrows():
        vals = [row.get(f"Q{q}_share", np.nan) for q in [1,2,3,4]]
        total = sum(v for v in vals if not np.isnan(v))
        vs = "  ".join(f"{v*100:>7.1f}%" if not np.isnan(v) else f"{'—':>8}" for v in vals)
        print(f"  {int(year):>6}  {vs}  {total*100:>7.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db","#2ecc71","#e67e22","#e74c3c"]
    for q, c in zip([1,2,3,4], colors):
        sub = df_qs2[df_qs2["quarter"]==q].sort_values("year")
        ax.plot(sub["year"], sub["q_share"]*100, marker="o", color=c,
                lw=1.5, label=f"Q{q}", alpha=0.8)
    ax.axhline(25, color="gray", ls="--", alpha=0.4, label="Equal share")
    ax.set_xlabel("Year"); ax.set_ylabel("Share of Annual Total (%)")
    ax.set_title("Historical Q1-Q4 Seasonal Shares  (CoV < 10% = deterministic anchor)",
                 fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "d6_seasonal_anchors.png", dpi=150)
    plt.close()
    pivot.reset_index().to_csv(out / "d6_seasonal_anchors.csv", index=False)
    print(f"\n  Saved → {out / 'd6_seasonal_anchors.png'}  |  {out / 'd6_seasonal_anchors.csv'}")
else:
    print("  inward_quarterly_seasonal.csv not found — skipping section 4.")

# =============================================================================
print(f"\n{sep}")
print("D6 COMPLETE")
print(sep)
print("""
  Outputs:
    d6_garch_gate_persistence.png  — σ_t + gate activation time series
    d6_shap_annual.png / .csv      — SHAP attribution for annual YoY
    d6_residual_acf_pacf.png       — Pre/post-sentiment residual whitening
    d6_seasonal_anchors.png / .csv — Q1-Q4 CoV proof

  Copy-paste reviewer responses in D5 output — rerun D5 for the text.
""")
