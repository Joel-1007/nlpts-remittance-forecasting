# DIAGNOSTIC: Map all available output files from Cells 6-9
import os
import json
import pandas as pd

print("="*70)
print("DIAGNOSTIC: Available pipeline output files")
print("="*70)

search_dirs = ['/kaggle/working/', '/kaggle/input/']

for search_dir in search_dirs:
    if not os.path.exists(search_dir):
        print(f"\n{search_dir} — NOT FOUND")
        continue
    print(f"\n📁 {search_dir}")
    all_files = sorted(os.listdir(search_dir))
    for f in all_files:
        full = os.path.join(search_dir, f)
        size = os.path.getsize(full) / 1024
        print(f"   {f:<55} ({size:.1f} KB)")

print("\n" + "="*70)
print("KEY FILE CONTENTS")
print("="*70)

# Check all JSON summaries
json_files = [
    '/kaggle/working/cell7_summary.json',
    '/kaggle/working/phase8_results.json',
    '/kaggle/working/cell9_summary.json',
]
for jf in json_files:
    if os.path.exists(jf):
        with open(jf) as f:
            data = json.load(f)
        print(f"\n📊 {jf}:")
        for k, v in data.items():
            print(f"   {k}: {v}")
    else:
        print(f"\n❌ MISSING: {jf}")

# Check all CSV shapes and columns
csv_files = [
    '/kaggle/working/features_train.csv',
    '/kaggle/working/features_test.csv',
    '/kaggle/working/cell7_forecasts.csv',
    '/kaggle/working/phase8_predictions.csv',
    '/kaggle/working/cell9_predictions.csv',
    '/kaggle/working/cell9_model_comparison.csv',
    '/kaggle/working/phase8_features_train.csv',
    '/kaggle/working/phase8_epu_vol_train.csv',
    '/kaggle/working/phase8_epu_vol_test.csv',
]
for cf in csv_files:
    if os.path.exists(cf):
        df = pd.read_csv(cf)
        print(f"\n📋 {cf}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Head(2):\n{df.head(2).to_string()}")
    else:
        print(f"\n❌ MISSING: {cf}")

# Check for any sentiment/EPU/language files
print("\n" + "="*70)
print("SEARCHING FOR SENTIMENT / EPU / LANGUAGE FILES")
print("="*70)
for f in sorted(os.listdir('/kaggle/working/')):
    if any(kw in f.lower() for kw in ['sent', 'epu', 'lang', 'mbert', 'nlp', 'phase', 'cell']):
        full = os.path.join('/kaggle/working/', f)
        size = os.path.getsize(full)/1024
        if f.endswith('.csv'):
            df = pd.read_csv(full)
            print(f"   {f:<50} {df.shape}  cols: {df.columns.tolist()[:8]}")
        elif f.endswith('.json'):
            with open(full) as fp:
                d = json.load(fp)
            print(f"   {f:<50} keys: {list(d.keys())[:10]}")
        else:
            print(f"   {f:<50} ({size:.1f} KB)")