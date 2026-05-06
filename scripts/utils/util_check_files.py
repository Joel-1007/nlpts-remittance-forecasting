import pandas as pd
from pathlib import Path

files_to_check = [
    "inward_quarterly_seasonal.csv",
    "features_combined.csv",
    "features_train.csv",
    "baseline_forecasts.csv",
    "remittances_quarterly_by_language.csv",
    "inward_flows.csv",
    "cell7_forecasts.csv",
    "cell9_predictions.csv",
    "phase8_features_train.csv",
]

for f in files_to_check:
    p = Path("/kaggle/working") / f
    if p.exists():
        df = pd.read_csv(p, nrows=3)
        print(f"\n{'='*60}")
        print(f"{f}  ({p.stat().st_size//1024} KB,  shape will be larger)")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample row: {df.iloc[0].to_dict()}")
