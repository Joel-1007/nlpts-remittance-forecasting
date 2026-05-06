# 🌐 Multi-Lingual Sentiment-Driven Remittance Flow Forecasting

> **Q1 Journal Publication Standard** — Forecasting India's inward/outward remittance flows using multilingual NLP sentiment signals, econometric decomposition, and hybrid ML/DL models.

---

## 📋 Project Overview

This project builds a **sentiment-augmented forecasting pipeline** for India's remittance flows (2000–2025), incorporating:

- **Multilingual news sentiment** (8 Indian languages via mBERT / XLM-RoBERTa) sourced from GDELT + Google News RSS
- **Economic Policy Uncertainty (EPU) Index** as a macro feature
- **STL decomposition**, Granger causality, and stationarity testing (ADF + KPSS)
- **Time-series models**: SARIMA, XGBoost, GARCH, and deep learning (LSTM/GRU)
- **No-leakage temporal split** (70% train / 30% test) applied *before* all feature engineering

---

## 🗂️ Repository Structure

```
nlpts-remittance-forecasting/
│
├── notebooks/
│   └── nlpts_remittance_forecasting.ipynb   # Original Kaggle notebook (full pipeline)
│
├── src/                                      # Core pipeline scripts (run in order)
│   ├── cell01_environment_setup.py           # Install dependencies, check GPU
│   ├── cell02_data_loading.py                # Load EPU + Inward/Outward Excel files
│   ├── cell03_preprocessing_feature_engineering.py  # STL, rolling features, temporal split
│   ├── cell04_news_collector_gdelt.py        # GDELT GKG + Google News multilingual collector
│   ├── cell05_sentiment_analysis_mbert.py    # mBERT sentiment labeling + ablation
│   ├── cell06_time_series_modeling.py        # SARIMA baseline + XGBoost (Diff_XGB_diff)
│   ├── cell07_sentiment_integrated_forecasting.py  # Sentiment gate + ensemble forecasts
│   ├── cell08_garch_volatility.py            # GARCH EPU volatility features
│   ├── cell09_deep_learning.py               # LSTM/GRU annual-resolution training
│   └── cell10_visualization_suite.py         # Publication-quality figures (Q1 standard)
│
├── scripts/
│   ├── diagnostics/                          # Reviewer-response diagnostics
│   │   ├── diagnostic_d1_data_integrity.py   # D1: Disaggregation check
│   │   ├── diagnostic_d2_nlp_corpus.py       # D2: NLP corpus transparency
│   │   ├── diagnostic_d3_sarima_baseline.py  # D3: SARIMA grid + AIC/BIC + Ljung-Box
│   │   ├── diagnostic_d4_pipeline_workflow.py # D4: VADER→mBERT labeling evidence
│   │   ├── diagnostic_d5_annual_forecasting.py # D5: YoY validity & justification
│   │   ├── diagnostic_d6_reviewer_evidence.py  # D6: GARCH, SHAP, residuals, CI
│   │   └── cell06b_diagnostic_feature_check.py # Quick feature sanity check
│   │
│   ├── tests/                                # Statistical tests for reviewers
│   │   ├── test_t1_diebold_mariano.py        # T1: DM test + forecast superiority
│   │   ├── test_t2_manual_annotation.py      # T2: Inter-rater agreement (Cohen's κ)
│   │   └── test_t3_zero_shot_classifier.py   # T3: Zero-shot vs mBERT comparison
│   │
│   └── utils/                                # Helper/utility scripts
│       ├── install_statsmodels.sh
│       ├── install_feedparser.sh
│       ├── install_arch.sh
│       ├── util_list_files.py                # List Kaggle working/input files
│       ├── util_check_files.py               # Verify required output CSVs exist
│       ├── util_map_output_files.py          # Map all pipeline output files
│       ├── util_deep_read_files.py           # Deep-read key output files
│       ├── util_granger_causality_check.py   # Quick Granger causality recheck
│       ├── util_prediction_interval_check.py # PI coverage validation
│       ├── util_epu_stationarity_check.py    # EPU stationarity quick check
│       ├── util_yoy_directional_accuracy.py  # YoY directional accuracy + Wilson CI
│       ├── util_dm_test_check.py             # DM test manual recheck
│       ├── util_pi_coverage_check.py         # 95% PI coverage check
│       ├── util_sentiment_ablation_check.py  # Per-language F1 ablation summary
│       ├── util_zip_plots.py                 # Zip plots/ for download
│       ├── util_zip_outputs.py               # Zip all outputs for download
│       └── util_ljungbox_sarima.py           # Ljung-Box test on SARIMA residuals
│
├── data/                                     # Place your Excel input files here
│   └── .gitkeep
│
├── docs/
│   └── pipeline_overview.md                  # Methodology notes
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Kaggle)

1. **Upload Excel data files** to a Kaggle dataset:
   - `Inward_remittance_flows_*.xlsx`
   - `Outward_remittance_flows_*.xlsx`
   - `India_Policy_Uncertainty_Data*.xlsx`

2. **Run cells in order** inside the Kaggle notebook:
   ```
   Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 →
   Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10
   ```

3. **Download outputs**: Run `util_zip_outputs.py` to download all results.

---

## 📦 Data Sources

| Dataset | Source | Period | Frequency |
|---------|--------|--------|-----------|
| Inward Remittance Flows | World Bank / RBI | 2000–2024 | Annual |
| Outward Remittance Flows | World Bank / RBI | 2000–2024 | Annual |
| India EPU Index | PolicyUncertainty.com | 2003–2025 | Monthly |
| News Articles (English) | GDELT GKG | 2017–2025 | Daily |
| News Articles (Multilingual) | Google News RSS | 2017–2025 | Live |

---

## 🧠 Models

| Model | RMSE (USD M) | DirAcc (%) |
|-------|-------------|------------|
| SARIMA baseline | — | — |
| Diff_XGB_diff (best) | 2,783 | — |
| GARCH-augmented XGB | — | — |
| LSTM/GRU (annual) | — | — |

---

## 📊 Key Features

- ✅ **8 Indian languages**: Hindi, Tamil, Telugu, Malayalam, Bengali, Punjabi, Gujarati, English
- ✅ **No data leakage**: Temporal split applied before STL decomposition and rolling features
- ✅ **Conservation-checked**: Annual → quarterly conversion error = 0.000000%
- ✅ **COVID segmentation**: Pre / During / Post COVID test set analysis (Table A2)
- ✅ **Reviewer diagnostics**: D1–D6 + Tests T1–T3 fully implemented
- ✅ **Publication figures**: 300 DPI, journal-style, zero synthetic placeholders

---

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full dependency list.

---

## 📝 Citation

If you use this codebase, please cite the associated paper (forthcoming).

---

## 📄 License

MIT License. See `LICENSE` for details.
