# Pipeline Overview — Methodology Notes

## Architecture

```
Raw Data (Excel)
    │
    ▼
[Cell 2] Data Loading
    │   • EPU Index (monthly, 2003–2025)
    │   • Inward remittance flows (annual, 201 countries)
    │   • Outward remittance flows (annual, 201 countries)
    │
    ▼
[Cell 3] Preprocessing & Feature Engineering (NO DATA LEAKAGE)
    │   • Filter India-specific rows
    │   • Annual → quarterly conversion (equal split, conservation validated)
    │   • Temporal split FIRST (70% train / 30% test)
    │   • STL decomposition on TRAINING data only
    │   • Rolling features with expanding window for test
    │   • Stationarity tests (ADF + KPSS) on training data
    │   • Granger causality on training data
    │
    ▼
[Cell 4] News Collection
    │   • GDELT GKG bulk download (2017–2025, every day)
    │   • Google News RSS (8 Indian languages)
    │   • Relevance scoring: English ≥5%, Multilingual ≥0.05%
    │
    ▼
[Cell 5] Sentiment Analysis (mBERT / XLM-RoBERTa)
    │   • VADER for English labeling
    │   • mBERT fine-tuning for 8 languages
    │   • Sentiment vectors aggregated quarterly
    │   • Ablation study: per-language contribution
    │
    ▼
[Cell 6] Time Series Modeling
    │   • SARIMA baseline (grid search, AIC/BIC selection)
    │   • XGBoost with differenced features (best: Diff_XGB_diff)
    │   • SARIMA prediction intervals (80/90/95%)
    │
    ▼
[Cell 7] Sentiment-Integrated Forecasting
    │   • Sentiment gate (learned weighting)
    │   • Ensemble: SARIMA + XGB + Sentiment
    │
    ▼
[Cell 8] GARCH Volatility Features
    │   • GARCH(1,1) on EPU residuals
    │   • Conditional volatility as additional predictor
    │
    ▼
[Cell 9] Deep Learning (Annual Resolution)
    │   • LSTM / GRU with annual-level inputs
    │   • Compared against Diff_XGB_diff baseline
    │
    ▼
[Cell 10] Visualization Suite
        • Publication-quality figures (300 DPI)
        • Main paper figures + supplementary
        • COVID period analysis (Table A2)
```

## Key Design Decisions

### 1. Annual → Quarterly Conversion
Annual World Bank data converted to quarterly using **equal distribution** (Annual / 4).
- Conservation check: 0.000000% error
- Sensitivity check vs STL-derived seasonal weights: mean deviation = 3.0% → equal split is robust (Appendix Table A1)

### 2. Temporal Split Before Feature Engineering
All STL decomposition, rolling features, and normalization computed on **training data only**.
Test set uses seasonal patterns extrapolated from the last 4 training quarters.

### 3. GDELT Data Collection Strategy
- **Method**: GKG (Global Knowledge Graph) table, NOT the Mentions table
- **Reason**: GKG contains GDELT theme codes (ECON_REMITTANCE, WB_2396_REMITTANCES) enabling precise article matching even without full text
- **Coverage**: Every day sampled (no skipping), 2017–2025

### 4. Multilingual Sentiment
- English: VADER labels → mBERT fine-tuning
- Indian languages: mBERT shared embeddings (104-language model)
- Relevance threshold: very lenient (0.05%) for non-English to maximise recall

## Reviewer Responses

| Reviewer Concern | Response Cell |
|-----------------|---------------|
| Data disaggregation | Diagnostic D1 |
| NLP corpus transparency | Diagnostic D2 |
| SARIMA baseline justification | Diagnostic D3 |
| Black-box pipeline | Diagnostic D4 |
| Annual forecasting validity | Diagnostic D5 |
| GARCH/SHAP evidence | Diagnostic D6 |
| Forecast superiority (DM test) | Test T1 |
| Manual annotation agreement | Test T2 |
| Zero-shot classifier comparison | Test T3 |
