# DIAGNOSTIC 2: Deep-read remaining key files

import pandas as pd
import json
import numpy as np

print("="*70)
print("DIAGNOSTIC 2: Content of remaining key files")
print("="*70)

files_to_read = {
    'ablation_results.json':              'json',
    'baseline_info.json':                 'json',
    'sentiment_correlation_analysis.json':'json',
    'sentiment_value_add.json':           'json',
    'collection_metadata.json':           'json',
    'phase8_garch_params.json':           'json',
    'sentiment_alignment_check.json':     'json',
    'language_f1_weights.csv':            'csv',
    'sentiment_vectors.csv':              'csv',
    'sentiment_stability_analysis.csv':   'csv',
    'stationarity_tests.csv':             'csv',
    'cell7_model_comparison.csv':         'csv',
    'cell8_model_comparison.csv':         'csv',
    'phase8_feature_importance.csv':      'csv',
    'remittances_quarterly.csv':          'csv',
    'remittances_quarterly_by_language.csv': 'csv',
    'epu_data.csv':                       'csv',
    'cell7_annual_yoy_breakdown.csv':     'csv',
    'covid_period_table_a2_cell7.csv':    'csv',
    'granger_causality_tests.csv':        'csv',
    'baseline_model_comparison.csv':      'csv',
    'sentiment_model_comparison.csv':     'csv',
    'inward_quarterly_seasonal.csv':      'csv',
    'crisis_index_train.csv':             'csv',
}

base = '/kaggle/working/'
for fname, ftype in files_to_read.items():
    full = base + fname
    try:
        if ftype == 'json':
            with open(full) as f:
                d = json.load(f)
            print(f"\n📊 {fname}:")
            for k, v in d.items():
                print(f"   {k}: {v}")
        else:
            df = pd.read_csv(full)
            print(f"\n📋 {fname}  shape={df.shape}")
            print(f"   cols: {df.columns.tolist()}")
            print(df.to_string(index=False))
    except Exception as e:
        print(f"\n❌ {fname}: {e}")

# Special: mbert_sentiment directory
import os
mb_dir = '/kaggle/working/mbert_sentiment'
if os.path.isdir(mb_dir):
    print(f"\n📁 mbert_sentiment/ contents:")
    for f in os.listdir(mb_dir):
        full = os.path.join(mb_dir, f)
        size = os.path.getsize(full)/1024
        print(f"   {f}  ({size:.1f} KB)")
        if f.endswith('.json'):
            with open(full) as fp:
                d = json.load(fp)
            for k, v in list(d.items())[:20]:
                print(f"      {k}: {v}")
        elif f.endswith('.csv'):
            df = pd.read_csv(full)
            print(f"   shape={df.shape}  cols={df.columns.tolist()}")
            print(df.head(3).to_string())

# Also check inward_flows.csv structure (first few rows only - it's large)
print("\n📋 inward_flows.csv (first 3 rows):")
df_if = pd.read_csv('/kaggle/working/inward_flows.csv', nrows=3)
print(f"   cols: {df_if.columns.tolist()}")
print(df_if.to_string())

print("\n📋 outward_flows.csv (first 3 rows):")
df_of = pd.read_csv('/kaggle/working/outward_flows.csv', nrows=3)
print(f"   cols: {df_of.columns.tolist()}")
print(df_of.to_string())

# Sentiment vectors full print
print("\n📋 sentiment_vectors.csv (all rows):")
df_sv = pd.read_csv('/kaggle/working/sentiment_vectors.csv')
print(f"   shape={df_sv.shape}")
print(df_sv.to_string())