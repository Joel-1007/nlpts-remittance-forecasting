# Pipeline Overview — Methodology Notes

> **NLPTS Remittance Forecasting** — Full technical methodology reference for reviewers and reproducers.

---

## Architecture Diagram

```
Raw Data (Excel: Inward/Outward + EPU)
    │
    ▼ [Cell 2] Data Loading
    │   • EPU Index (monthly, 2003–2025) — 272 rows
    │   • Inward remittance flows (annual, 201 countries) — World Bank/RBI
    │   • Outward remittance flows (annual, 201 countries) — World Bank/RBI
    │   • India-specific rows filtered (country code "IND")
    │
    ▼ [Cell 3] Preprocessing & Feature Engineering (ZERO DATA LEAKAGE)
    │   • Annual → quarterly conversion (equal split: Annual ÷ 4)
    │   • Conservation check: error = 0.000000%
    │   • Temporal split FIRST: 70% train (2000–2016) / 30% test (2017–2024)
    │   • STL decomposition on TRAINING data only
    │   • Rolling features with expanding window for test
    │   • Stationarity tests: ADF + KPSS (training data only)
    │   • Granger causality tests (training data only)
    │
    ▼ [Cell 4] News Collection (37,978+ articles)
    │   • GDELT GKG bulk download (2017–2025, daily, theme: ECON_REMITTANCE)
    │   • Google News RSS (8 Indian languages)
    │   • Relevance: English ≥5%, Multilingual ≥0.05%
    │
    ▼ [Cell 5] Sentiment Analysis
    │   • English: VADER labeling → mBERT fine-tuning (F1=0.999, n=34,357)
    │   • Indian languages: XLM-RoBERTa shared embeddings
    │   • Quarterly aggregation: mean sentiment + std + article count
    │   • Ablation: per-language F1 scores validated
    │
    ▼ [Cell 6] Time Series Baseline Modeling
    │   • SARIMA grid search (7 configurations)
    │   • Best SARIMA: (0,1,2)×(1,1,1,4) → RMSE = 6,429 USD M, MAPE = 17.03%
    │   • XGBoost: Diff_XGB_diff → RMSE = 2,783 USD M, MAPE = 8.78% (↓56.7%)
    │   • Prediction intervals: 80/90/95% coverage validated
    │
    ▼ [Cell 7] Sentiment-Integrated Forecasting
    │   • Sentiment gate (learned weighting) on 33 sentiment features
    │   • 34 model configurations evaluated
    │   • Best model: Diff_XGB_diff, R² = 0.767, YoY Dir Acc = 71.4%
    │   • Ensemble tested: SARIMA + XGB + Sentiment
    │
    ▼ [Cell 8] GARCH Volatility Features
    │   • GARCH(1,1) fitted on EPU residuals
    │   • Conditional volatility used as additional predictor
    │   • GARCH-augmented GBM: RMSE = 2,827 USD M, YoY Dir Acc = 83.3%
    │   • 53 total features: 6 EPU, 42 sentiment, 5 GARCH
    │
    ▼ [Cell 9] Deep Learning (Annual Resolution, 17 training observations)
    │   • GARCH-Gated LSTM/GRU: RMSE = 2,124 USD M, MAPE = 6.76%, R² = 0.864
    │   • DL+C7+C8 Thirds Ensemble: RMSE = 2,347 USD M
    │   • Training: StandardScaler on differenced annual targets
    │   • Data leakage prevented: scaler fit on train only
    │
    ▼ [Cell 10] Visualization Suite
        • Figure 1: Quarterly article heatmap by language
        • Figure 2: SARIMA diagnostics (ACF, PACF, residuals)
        • Figure 3: Forecast comparison (all models vs SARIMA)
        • Figure 4: Ensemble forecasts
        • Table A1: Sensitivity (equal vs STL seasonal weights)
        • Table A2: COVID-period performance breakdown
```

---

## Key Design Decisions

### 1. Annual → Quarterly Conversion

Annual World Bank data converted to quarterly using **equal distribution** (Annual / 4).

- **Conservation check**: 0.000000% error across all years
- **Sensitivity check** vs STL-derived seasonal weights: mean deviation = 3.0%
- **Conclusion**: Equal split is robust (Appendix Table A1)

### 2. Temporal Split Before Feature Engineering

All STL decomposition, rolling features, and normalization computed on **training data only**.

- Train: 2000 Q1 – 2016 Q4 (70%)
- Test: 2017 Q1 – 2024 Q4 (30%)
- Test set uses seasonal patterns extrapolated from the last 4 training quarters
- This prevents any form of look-ahead bias

### 3. GDELT Data Collection Strategy

- **Method**: GKG (Global Knowledge Graph) table, NOT the Mentions table
- **Reason**: GKG contains GDELT theme codes (`ECON_REMITTANCE`, `WB_2396_REMITTANCES`) enabling precise article matching even without full text
- **Coverage**: Every day sampled (no skipping), 2017–2025

### 4. Multilingual Sentiment Architecture

- **English** (n=34,357): VADER labels → mBERT fine-tuning, F1 = 0.999
- **Indian languages**: mBERT shared embeddings (104-language model) with XLM-RoBERTa labels
- **Relevance threshold**: Very lenient (0.05%) for non-English to maximise recall
- **VADER→mBERT agreement**: 99.9% on English articles

### 5. QoQ Directional Accuracy — Measurement Note

QoQ directional accuracy is a **measurement artifact** in this dataset: source data is annual values divided equally into 4 identical quarters, so within-year difference = 0 always. **YoY directional accuracy is the correct metric** (71.4% for best models).

---

## Validation & Statistical Tests

| Test | Description | Result |
|------|-------------|--------|
| **ADF + KPSS** | Stationarity on training EPU series | Stationary after 1st differencing |
| **Granger Causality** | EPU → Remittances (lags 1–4) | Significant at p<0.05 (lag 1) |
| **Ljung-Box** | SARIMA residual autocorrelation | Residuals are white noise |
| **Diebold-Mariano (T1)** | Diff_XGB_diff vs SARIMA | Statistically superior |
| **Cohen's κ (T2)** | Manual annotation agreement | κ > 0.80 (substantial) |
| **Zero-shot classifier (T3)** | XLM-RoBERTa vs mBERT | mBERT competitive on multilingual |

---

## Reviewer Response Map

| Reviewer Concern | Response Script |
|-----------------|-----------------|
| Data disaggregation transparency | `scripts/diagnostics/diagnostic_d1_data_integrity.py` |
| NLP corpus transparency | `scripts/diagnostics/diagnostic_d2_nlp_corpus.py` |
| SARIMA baseline justification | `scripts/diagnostics/diagnostic_d3_sarima_baseline.py` |
| Black-box pipeline evidence | `scripts/diagnostics/diagnostic_d4_pipeline_workflow.py` |
| Annual forecasting validity | `scripts/diagnostics/diagnostic_d5_annual_forecasting.py` |
| GARCH/SHAP/residual evidence | `scripts/diagnostics/diagnostic_d6_reviewer_evidence.py` |
| Forecast superiority (DM test) | `scripts/tests/test_t1_diebold_mariano.py` |
| Manual annotation agreement | `scripts/tests/test_t2_manual_annotation.py` |
| Zero-shot vs mBERT comparison | `scripts/tests/test_t3_zero_shot_classifier.py` |

---

## Output Files Reference

| File | Description |
|------|-------------|
| `cell7_model_comparison.csv` | All 34 model RMSE/MAPE/R²/DirAcc scores |
| `cell7_summary.json` | Cell 7 best model summary |
| `cell7_forecast_comparison.png` | Main forecast figure |
| `phase8_results.json` | GARCH-augmented model summary |
| `cell9_summary.json` | Deep learning results summary |
| `cell9_best_model.keras` | Saved GARCH-Gated model weights |
| `ablation_results.json` | Per-language NLP F1 ablation |
| `sarima_diagnostics.png` | SARIMA diagnostic plots |
| `figure1_quarterly_article_heatmap.png` | News coverage by language |
| `sentiment_stability_analysis.csv` | Sentiment vector stability |
| `stationarity_tests.csv` | ADF/KPSS results |
| `granger_causality_tests.csv` | Granger test p-values |
