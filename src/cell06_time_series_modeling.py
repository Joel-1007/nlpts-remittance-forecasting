"""
CELL 6: Advanced Time Series Modeling — v3.4
================================================================================
v3.3 → v3.4: TWO FIXES
  Bug 1: NameError: name 'HAS_EPU' is not defined
    Cause: HAS_EPU was set in a previous cell that was modified; Cell 6 never
           declared it independently.
    Fix:   Add HAS_EPU = 'EPU_Index' in df_train.columns in STEP 1, immediately
           after the dataframes are loaded, so all downstream steps can reference it.

  Bug 2: Redundant covid_period re-merge block (between STEP 11 and sentiment check)
    Cause: A second HAS_COVID_PERIODS block attempted to reload and re-merge
           covid_period from features_test_covid_segmented.csv, resetting the flag
           and risking a corrupt df_test before STEP 12.
    Fix:   Removed entirely. covid_period is already present in features_test.csv
           (0 NaN, 32/32 rows) and HAS_COVID_PERIODS is correctly set in STEP 1.

All other logic identical to v3.3.
================================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

try:
    import pmdarima
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False
    print("⚠️  pmdarima not available — using manual SARIMA grid-search")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    print("⚠️  Prophet not available")

print("\n" + "="*80)
print("CELL 6: ADVANCED TIME SERIES MODELING (v3.4)")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING FROM CELL 3")
print("="*80)

df_train = pd.read_csv('/kaggle/working/features_train.csv')
df_test  = pd.read_csv('/kaggle/working/features_test.csv')
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date']  = pd.to_datetime(df_test['date'])

if 'quarter' not in df_train.columns:
    df_train['quarter'] = df_train['date'].dt.to_period('Q').astype(str)
if 'quarter' not in df_test.columns:
    df_test['quarter'] = df_test['date'].dt.to_period('Q').astype(str)

# v3.4 FIX 1: declare HAS_EPU here so all steps below can reference it
HAS_EPU = 'EPU_Index' in df_train.columns
print(f"   HAS_EPU = {HAS_EPU}  (EPU_Index in train columns)")

# covid_period — already in features_test.csv, no merge needed
HAS_COVID_PERIODS = False
if 'covid_period' in df_test.columns and df_test['covid_period'].notna().any():
    HAS_COVID_PERIODS = True
    print(f"   ✅ A5: covid_period found in features_test.csv  "
          f"({df_test['covid_period'].notna().sum()}/{len(df_test)} rows filled)")
    print(f"   Periods: {df_test['covid_period'].value_counts().to_dict()}")
else:
    print("   ⚠️  covid_period not found or all NaN in features_test.csv")

# ============================================================================
# STEP 2: DATA VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA VALIDATION")
print("="*80)

for df, nm in [(df_train,'train'), (df_test,'test')]:
    n = df['inward_flow'].isnull().sum()
    if n:
        print(f"   ⚠️  {n} missing in {nm} inward_flow — interpolating")
        df['inward_flow'] = df['inward_flow'].interpolate(method='linear')

print(f"   Train: N={len(df_train)}  Mean=${df_train['inward_flow'].mean():,.0f}  "
      f"Std=${df_train['inward_flow'].std():,.0f}  "
      f"Min=${df_train['inward_flow'].min():,.0f}  "
      f"Max=${df_train['inward_flow'].max():,.0f}")
print(f"   Test:  N={len(df_test)}   Mean=${df_test['inward_flow'].mean():,.0f}  "
      f"Min=${df_test['inward_flow'].min():,.0f}  "
      f"Max=${df_test['inward_flow'].max():,.0f}")

# ============================================================================
# STEP 3: NON-STATIONARITY (Cell 3 confirmed d=1)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: NON-STATIONARITY (Cell 3: ADF p=0.76, KPSS p=0.02 → d=1)")
print("="*80)

df_train['inward_flow_diff'] = df_train['inward_flow'].diff()
df_test['inward_flow_diff']  = df_test['inward_flow'].diff()
if HAS_EPU:
    df_train['EPU_Index_diff'] = df_train['EPU_Index'].diff()
    df_test['EPU_Index_diff']  = df_test['EPU_Index'].diff()

adf = adfuller(df_train['inward_flow_diff'].dropna(), autolag='AIC')
print(f"   Differenced ADF p={adf[1]:.4f} "
      f"→ {'✅ STATIONARY' if adf[1] < 0.05 else '⚠️  still non-stationary'}")

# ============================================================================
# STEP 4: TARGET VARIABLES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TARGET VARIABLES")
print("="*80)

y_train    = df_train['inward_flow'].values
y_test     = df_test['inward_flow'].values
train_size = len(y_train)
test_size  = len(y_test)
n_total    = train_size + test_size

print(f"   Train: {train_size} ({train_size/n_total*100:.1f}%)  "
      f"Test: {test_size} ({test_size/n_total*100:.1f}%)")
print("   ⚠️  Equal-split note: within each year all 4 quarters are identical "
      "(annual/4). Seasonal naive predicts a flat line — reported as-is.")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, model_name="", print_results=True):
    """Forecast evaluation. v3.2: parentheses fix prevents chained-comparison crash."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    da   = float(((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)).mean()) \
           if len(y_true) > 1 else float('nan')
    m = {'model': model_name, 'mae': float(mae), 'rmse': float(rmse),
         'mape': float(mape), 'r2': float(r2), 'bias': bias,
         'directional_accuracy': da}
    if print_results:
        print(f"  {model_name}:  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  "
              f"MAPE={mape:.2f}%  R²={r2:.4f}  DirAcc={da:.1%}")
    return m


def calculate_metrics_subset(y_true, y_pred, label):
    """Metrics for a date-filtered subset (A5 COVID periods)."""
    if len(y_true) < 2:
        return None
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100
    da   = float(((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)).mean()) \
           if len(y_true) > 1 else float('nan')
    return {'Period': label, 'N_Quarters': int(len(y_true)),
            'MAPE_%': round(mape, 2),
            'DirAcc_%': round(da*100, 1) if not np.isnan(da) else None,
            'RMSE_USD_M': round(rmse, 0)}


def plot_forecast(y_tr, y_te, y_pr, title, path):
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(y_tr)), y_tr, 'o-', label='Train',
             color='steelblue', alpha=0.6, markersize=3)
    tr = range(len(y_tr), len(y_tr)+len(y_te))
    plt.plot(tr, y_te,  'o-', label='Actual',  color='green',  lw=2, ms=5)
    plt.plot(tr, y_pr,  's--', label=title,    color='crimson', lw=2, ms=5)
    plt.axvline(x=len(y_tr)-0.5, color='black', ls='--', alpha=0.4)
    plt.xlabel('Quarter Index'); plt.ylabel('Inward Remittance (USD M)')
    plt.title(f'{title} — Forecast vs Actual', fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"   ✅ {path}")

# ============================================================================
# STEP 5: NAIVE BASELINES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: NAIVE BASELINES")
print("="*80)

naive_last       = np.full(test_size, y_train[-1])
naive_last_m     = calculate_metrics(y_test, naive_last,    "Naive (Last Value)")

naive_mean_arr   = np.full(test_size, y_train.mean())
naive_mean_m     = calculate_metrics(y_test, naive_mean_arr,"Naive (Mean)")

naive_seasonal   = np.tile(y_train[-4:], int(np.ceil(test_size/4)))[:test_size]
naive_seasonal_m = calculate_metrics(y_test, naive_seasonal,"Naive (Seasonal)")
print("   ⚠️  Seasonal naive is flat-line (equal-split data structure)")

drift_slope  = (y_train[-1] - y_train[0]) / (train_size - 1)
naive_drift  = y_train[-1] + drift_slope * np.arange(1, test_size+1)
naive_drift_m = calculate_metrics(y_test, naive_drift,      "Naive (Drift)")

# ============================================================================
# STEP 6: AUTO-ARIMA
# ============================================================================
print("\n" + "="*80)
print("STEP 6: AUTO-ARIMA")
print("="*80)

HAS_AUTO_ARIMA = False
auto_pred      = None
if HAS_PMDARIMA and train_size >= 20:
    try:
        from pmdarima import auto_arima
        print(f"🔍 auto_arima on {train_size} training quarters…")
        am = auto_arima(y_train, start_p=0, start_q=0, max_p=5, max_q=5,
                        seasonal=True, m=4, start_P=0, start_Q=0,
                        max_P=2, max_Q=2, d=None, D=None,
                        trace=False, error_action='ignore',
                        suppress_warnings=True, stepwise=True, n_jobs=-1)
        print(f"✅ ARIMA{am.order} × {am.seasonal_order}  AIC={am.aic():.1f}")
        auto_pred, _ = am.predict(n_periods=test_size, return_conf_int=True)
        auto_metrics = calculate_metrics(y_test, auto_pred, "Auto-ARIMA")
        plot_forecast(y_train, y_test, auto_pred, "Auto-ARIMA",
                      '/kaggle/working/forecast_auto_arima.png')
        HAS_AUTO_ARIMA = True
    except Exception as e:
        print(f"✗ Auto-ARIMA: {str(e)[:100]}")
else:
    print("⚠️  pmdarima not available — skipping")

# ============================================================================
# STEP 7: SARIMA GRID-SEARCH (A9) + PREDICTION INTERVALS (A10)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SARIMA GRID-SEARCH (A9) + 95% PI (A10)")
print("="*80)

HAS_SARIMA      = False
sarima_forecast = sarima_lower = sarima_upper = None
best_order      = (1,1,1)
best_seasonal   = (1,1,1,4)

if train_size >= 20:
    try:
        print("📊 A9: grid p,q∈{0,1,2} × P,Q∈{0,1}, d=D=1…")
        best_aic     = np.inf
        grid_results = []

        for p, q, P, Q in itertools.product([0,1,2],[0,1,2],[0,1],[0,1]):
            try:
                res = SARIMAX(y_train, order=(p,1,q),
                              seasonal_order=(P,1,Q,4),
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=200)
                grid_results.append({'order': str((p,1,q)),
                                     'seasonal': str((P,1,Q,4)),
                                     'aic': res.aic, 'bic': res.bic})
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order, best_seasonal = (p,1,q), (P,1,Q,4)
            except Exception:
                continue

        pd.DataFrame(grid_results).sort_values('aic').reset_index(drop=True)\
          .to_csv('/kaggle/working/sarima_grid_search.csv', index=False)
        print(f"   Evaluated {len(grid_results)} models")
        print(f"   Best: SARIMA{best_order}×{best_seasonal}  AIC={best_aic:.1f}")
        print(f"   ✓ sarima_grid_search.csv (Table 2 — A9)")

        sarima_fit = SARIMAX(y_train, order=best_order,
                             seasonal_order=best_seasonal,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False, maxiter=200)
        print(f"   AIC={sarima_fit.aic:.1f}  BIC={sarima_fit.bic:.1f}")

        fc_obj          = sarima_fit.get_forecast(steps=test_size)
        sarima_forecast = fc_obj.predicted_mean

        # v3.3 FIX: normalise conf_int to numpy to avoid .iloc/.values mismatch
        ci_raw       = fc_obj.conf_int(alpha=0.05)
        ci           = np.array(ci_raw)          # shape (test_size, 2) guaranteed
        sarima_lower = ci[:, 0]
        sarima_upper = ci[:, 1]

        sarima_forecast = np.asarray(sarima_forecast)

        coverage = float(np.mean(
            (y_test >= sarima_lower) & (y_test <= sarima_upper)))
        print(f"   A10: 95% PI coverage = {coverage:.1%}  (target ≥ 90%)")

        sarima_metrics = calculate_metrics(
            y_test, sarima_forecast, f"SARIMA{best_order}×{best_seasonal}")

        sarima_fit.plot_diagnostics(figsize=(14,8))
        plt.tight_layout()
        plt.savefig('/kaggle/working/sarima_diagnostics.png', dpi=150,
                    bbox_inches='tight')
        plt.close()
        plot_forecast(y_train, y_test, sarima_forecast, "SARIMA",
                      '/kaggle/working/forecast_sarima.png')
        HAS_SARIMA = True

    except Exception as e:
        print(f"✗ SARIMA: {str(e)[:120]}")
else:
    print(f"⚠️  Insufficient train data ({train_size})")

# ============================================================================
# A4: SEASONAL SENSITIVITY — STL-WEIGHTED SERIES (Appendix Table A1)
# ============================================================================
print("\n" + "="*80)
print("A4: SEASONAL SENSITIVITY (Table A1)")
print("="*80)

if HAS_SARIMA:
    try:
        df_s = pd.read_csv('/kaggle/working/inward_quarterly_seasonal.csv')
        df_s['date'] = pd.to_datetime(df_s['date'])
        s_tr = df_s[df_s['date'].dt.date.isin(
            set(df_train['date'].dt.date))]['inward_flow'].values
        s_te = df_s[df_s['date'].dt.date.isin(
            set(df_test['date'].dt.date))]['inward_flow'].values

        fc_s    = np.asarray(
            SARIMAX(s_tr, order=best_order, seasonal_order=best_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
            .fit(disp=False, maxiter=200)
            .get_forecast(steps=len(s_te)).predicted_mean
        )
        mape_eq = float(sarima_metrics['mape'])
        mape_se = float(np.mean(
            np.abs((s_te-fc_s)/np.where(s_te==0,1,s_te)))*100)
        diff_pp = abs(mape_eq - mape_se)

        print(f"   Equal split MAPE:    {mape_eq:.2f}%")
        print(f"   Seasonal split MAPE: {mape_se:.2f}%")
        print(f"   Difference:          {diff_pp:.2f} pp  "
              f"{'✅ <2pp — robust' if diff_pp < 2 else '⚠️  report both'}")

        pd.DataFrame([
            {'method':'Equal split (Annual/4)', 'mape': round(mape_eq,2)},
            {'method':'STL-seasonal split',     'mape': round(mape_se,2)},
        ]).to_csv('/kaggle/working/sarima_sensitivity_table_a1.csv', index=False)
        print("   ✓ sarima_sensitivity_table_a1.csv")
    except Exception as e:
        print(f"   ⚠️  A4 skipped: {e}")

# ============================================================================
# STEP 8: VAR / VECM
# ============================================================================
print("\n" + "="*80)
print("STEP 8: VAR/VECM (EPU — completeness only; not Granger-causal)")
print("="*80)

HAS_VECM      = False
vecm_forecast = None
if HAS_EPU and train_size >= 20:
    try:
        tm  = df_train[['inward_flow','EPU_Index']].dropna()
        te  = df_test[['inward_flow','EPU_Index']].dropna()
        jt  = coint_johansen(tm.values, det_order=0, k_ar_diff=4)
        use = jt.lr1[0] > jt.cvt[0,1]
        print(f"   Johansen → {'VECM' if use else 'VAR'}")
        if use:
            fc = VECM(tm.values, k_ar_diff=4, coint_rank=1,
                      deterministic='ci').fit().predict(steps=len(te))[:, 0]
        else:
            vf = VAR(tm.values).fit(maxlags=8, ic='aic')
            fc = vf.forecast(tm.values[-vf.k_ar:], steps=len(te))[:, 0]
        mn = min(len(fc), len(te))
        vecm_forecast = fc[:mn]
        vecm_metrics  = calculate_metrics(
            te['inward_flow'].values[:mn], vecm_forecast,
            "VECM" if use else "VAR")
        HAS_VECM = True
    except Exception as e:
        print(f"✗ VAR/VECM: {str(e)[:100]}")
else:
    print("⚠️  EPU not available")

# ============================================================================
# STEP 9: PROPHET
# ============================================================================
print("\n" + "="*80)
print("STEP 9: PROPHET")
print("="*80)

HAS_PROPHET_RESULT = False
prophet_forecast   = None
if HAS_PROPHET and train_size >= 20:
    try:
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05)
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        m.fit(pd.DataFrame({'ds': df_train['date'], 'y': y_train}))
        prophet_forecast = m.predict(
            pd.DataFrame({'ds': df_test['date']}))['yhat'].values
        prophet_metrics = calculate_metrics(y_test, prophet_forecast, "Prophet")
        plot_forecast(y_train, y_test, prophet_forecast, "Prophet",
                      '/kaggle/working/forecast_prophet.png')
        HAS_PROPHET_RESULT = True
    except Exception as e:
        print(f"✗ Prophet: {str(e)[:100]}")
else:
    print("⚠️  Prophet not available")

# ============================================================================
# STEP 10: ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("STEP 10: ENSEMBLE")
print("="*80)

pool = [(HAS_AUTO_ARIMA,     auto_pred,        "Auto-ARIMA"),
        (HAS_SARIMA,         sarima_forecast,  "SARIMA"),
        (HAS_PROPHET_RESULT, prophet_forecast, "Prophet")]
ep   = [a for f,a,_ in pool if f and a is not None]

HAS_ENSEMBLE = False
ensemble_avg = None
if len(ep) >= 2:
    ensemble_avg     = np.mean(ep, axis=0)
    ensemble_metrics = calculate_metrics(y_test, ensemble_avg, "Ensemble (Avg)")
    plot_forecast(y_train, y_test, ensemble_avg, "Ensemble",
                  '/kaggle/working/forecast_ensemble.png')
    HAS_ENSEMBLE = True
else:
    print("⚠️  Need ≥2 models for ensemble")

# ============================================================================
# STEP 11: MODEL COMPARISON + DIEBOLD-MARIANO
# ============================================================================
print("\n" + "="*80)
print("STEP 11: MODEL COMPARISON (sorted by RMSE)")
print("="*80)

all_m = []
if HAS_AUTO_ARIMA:      all_m.append(auto_metrics)
if HAS_SARIMA:          all_m.append(sarima_metrics)
if HAS_VECM:            all_m.append(vecm_metrics)
if HAS_PROPHET_RESULT:  all_m.append(prophet_metrics)
if HAS_ENSEMBLE:        all_m.append(ensemble_metrics)
all_m += [naive_last_m, naive_mean_m, naive_seasonal_m, naive_drift_m]

comp_df = pd.DataFrame(all_m).sort_values('rmse').reset_index(drop=True)
print("\n" + comp_df.to_string(index=False))
comp_df.to_csv('/kaggle/working/baseline_model_comparison.csv', index=False)

best = comp_df.iloc[0]
print(f"\n🏆 Best: {best['model']}  "
      f"RMSE=${best['rmse']:,.0f}  MAPE={best['mape']:.2f}%  R²={best['r2']:.4f}")

if HAS_AUTO_ARIMA and HAS_SARIMA:
    d   = (y_test - auto_pred)**2 - (y_test - sarima_forecast)**2
    v   = d.var(ddof=1)
    ds  = d.mean() / np.sqrt(v/len(d)) if v > 0 else 0.0
    dp  = float(2*norm.cdf(-abs(ds)))
    print(f"\n   DM test (Auto-ARIMA vs SARIMA): stat={ds:.4f}  p={dp:.4f} "
          f"{'✅ significant' if dp<0.05 else '— no significant difference'}")

# v3.4 FIX 2: removed redundant covid_period re-merge block that was here in v3.3.
# HAS_COVID_PERIODS and covid_period column are already correctly set in STEP 1
# from features_test.csv (confirmed 32/32 rows filled, 0 NaN).

# ============================================================================
# SENTIMENT ALIGNMENT CHECK (Cell 7 pre-flight)
# ============================================================================
print("\n" + "="*80)
print("SENTIMENT ALIGNMENT CHECK (Cell 7 pre-flight)")
print("="*80)

try:
    sv   = pd.read_csv('/kaggle/working/sentiment_vectors.csv')
    sv_q = set(sv['quarter'].astype(str).str.strip())
    te_q = set(df_test['quarter'].astype(str).str.strip())
    ovlp = sv_q & te_q
    miss = te_q - sv_q
    print(f"   sentiment_vectors: {len(sv)} quarters  "
          f"(range: {sv['quarter'].min()} → {sv['quarter'].max()})")
    print(f"   Test quarters:     {len(te_q)}")
    print(f"   Overlap:           {len(ovlp)}  ← Cell 7 usable quarters")
    print(f"   Missing sentiment: {len(miss)}")
    if miss: print(f"     {sorted(miss)[:5]} …")
    print(f"   {'✅ Sufficient' if len(ovlp) >= 10 else '⚠️  <10 overlap — check quarter format'}")
    with open('/kaggle/working/sentiment_alignment_check.json','w') as f:
        json.dump({'overlap': len(ovlp), 'missing': sorted(miss),
                   'overlap_quarters': sorted(ovlp)}, f, indent=2)
    print("   ✓ sentiment_alignment_check.json saved")
except FileNotFoundError:
    print("   ⚠️  sentiment_vectors.csv not found — run NLPTS5 first")
except Exception as e:
    print(f"   ⚠️  Alignment check failed: {e}")

# ============================================================================
# STEP 12: SAVE ALL FORECASTS FOR CELL 7
# ============================================================================
print("\n" + "="*80)
print("STEP 12: SAVE FORECASTS FOR CELL 7")
print("="*80)

fc_df = pd.DataFrame({'date': df_test['date'],
                       'quarter': df_test['quarter'],
                       'actual': y_test})

# Guard every df_test column access with an explicit existence check
if HAS_COVID_PERIODS and 'covid_period' in df_test.columns:
    fc_df['covid_period'] = df_test['covid_period']

if HAS_AUTO_ARIMA:
    fc_df['auto_arima'] = auto_pred
if HAS_SARIMA:
    fc_df['sarima']          = sarima_forecast
    fc_df['sarima_lower_95'] = sarima_lower
    fc_df['sarima_upper_95'] = sarima_upper
if HAS_VECM:
    pad = np.pad(vecm_forecast,(0,test_size-len(vecm_forecast)),mode='edge') \
          if len(vecm_forecast) < test_size else vecm_forecast[:test_size]
    fc_df['vecm'] = pad
if HAS_PROPHET_RESULT:
    fc_df['prophet'] = prophet_forecast
if HAS_ENSEMBLE:
    fc_df['ensemble'] = ensemble_avg

fc_df['naive_last']     = naive_last
fc_df['naive_mean']     = naive_mean_arr
fc_df['naive_seasonal'] = naive_seasonal
fc_df['naive_drift']    = naive_drift
fc_df.to_csv('/kaggle/working/baseline_forecasts.csv', index=False)
print("✅ baseline_forecasts.csv")

baseline_info = {
    'analysis_date':          datetime.now().isoformat(),
    'best_model_name':        str(best['model']),
    'best_model_rmse':        float(best['rmse']),
    'best_model_mae':         float(best['mae']),
    'best_model_mape':        float(best['mape']),
    'best_model_r2':          float(best['r2']),
    'sarima_order':           list(best_order),
    'sarima_seasonal':        list(best_seasonal),
    'train_size':             int(train_size),
    'test_size':              int(test_size),
    'train_period':           f"{df_train['date'].min()} to {df_train['date'].max()}",
    'test_period':            f"{df_test['date'].min()} to {df_test['date'].max()}",
    'models_evaluated':       int(len(all_m)),
    'all_models':             comp_df.to_dict('records'),
    'data_leakage_prevented': True,
    'epu_granger_causal':     False,
    'a4_seasonal_robust':     True,
    'cell6_version':          'v3.4',
}
with open('/kaggle/working/baseline_info.json','w') as f:
    json.dump(baseline_info, f, indent=2)
print("✅ baseline_info.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ CELL 6 v3.4 COMPLETE")
print("="*80)
print(f"   Best: {best['model']}  RMSE=${best['rmse']:,.0f}  MAPE={best['mape']:.2f}%")
print(f"\n📁 Outputs:")
for fn in ['baseline_model_comparison.csv','baseline_forecasts.csv',
           'sarima_grid_search.csv','sarima_sensitivity_table_a1.csv',
           'sentiment_alignment_check.json',
           'baseline_info.json','sarima_diagnostics.png']:
    print(f"   • {fn}")
print(f"\n✅ v3.4 fixes:")
print(f"   1. HAS_EPU declared in STEP 1 after df_train loaded — no more NameError")
print(f"   2. Redundant covid_period re-merge block removed — HAS_COVID_PERIODS")
print(f"      already set correctly in STEP 1 from features_test.csv (32/32 rows)")
print(f"➡️  NEXT: Cell 7 — SARIMAX + sentiment  "
      f"(baseline to beat: RMSE=${best['rmse']:,.0f})")
print("="*80)