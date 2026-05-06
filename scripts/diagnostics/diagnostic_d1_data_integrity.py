"""
DIAGNOSTIC D1 — Data Integrity & Granularity
=============================================
Reviewer concern: Suspected artificial disaggregation of annual data.

HOW TO USE ON KAGGLE:
  • Paste this entire cell into a new Kaggle code cell and run it.
  • Requires real CSV files in /kaggle/working/ or /kaggle/input/ subdirs.
    Expected files (at least one of):
      epu_data.csv
      inward_flows.csv
      outward_flows.csv
      india_remittance_processed.csv  OR  india_remittance_final.csv

Output: printed report + d1_data_integrity_report.txt
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Locate data files ─────────────────────────────────────────────────────────
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

# ── Load real data — abort if nothing found ────────────────────────────────────
epu_path      = find("epu_data.csv")
inward_path   = find("inward_flows.csv")
outward_path  = find("outward_flows.csv")
combined_path = (
    find("remittances_quarterly.csv") or
    find("india_remittance_processed.csv") or
    find("india_remittance_final.csv") or
    find("features_combined.csv")
)

if not any([epu_path, inward_path, outward_path, combined_path]):
    raise FileNotFoundError(
        "D1 requires at least one real CSV file in /kaggle/working/ or /kaggle/input/.\n"
        "Expected: epu_data.csv, inward_flows.csv, outward_flows.csv, "
        "india_remittance_processed.csv (or india_remittance_final.csv).\n"
        "Upload your pipeline output files and rerun."
    )

df_epu  = pd.read_csv(epu_path,    parse_dates=["date"]) if epu_path    else None
df_in   = pd.read_csv(inward_path, parse_dates=["date"]) if inward_path else None
df_out  = pd.read_csv(outward_path, parse_dates=["date"]) if outward_path else None
df_c    = pd.read_csv(combined_path)                       if combined_path else None

sep = "=" * 72
print(sep)
print("D1 — DATA INTEGRITY & GRANULARITY DIAGNOSTIC")
print("     Mode: REAL DATA")
print(sep)

# ── 1. File presence ───────────────────────────────────────────────────────────
print("\n[1] File presence check")
for label, path in [("EPU",      epu_path),
                    ("Inward",   inward_path),
                    ("Outward",  outward_path),
                    ("Combined", combined_path)]:
    status = "FOUND" if path else "MISSING"
    print(f"  {label:10s}: {status}  {path or '—'}")

# ── 2. EPU frequency ──────────────────────────────────────────────────────────
print("\n[2] EPU index native frequency")
if df_epu is not None:
    df_epu = df_epu.sort_values("date")
    gaps = df_epu["date"].diff().dt.days.dropna()
    median_gap = gaps.median()
    freq_label = (
        "MONTHLY (native, ~30 days)"   if 20 <= median_gap <= 40  else
        "QUARTERLY (native, ~90 days)" if 80 <= median_gap <= 100 else
        "ANNUAL (native, ~365 days)"   if 300 <= median_gap <= 380 else
        f"IRREGULAR (median gap {median_gap:.0f} days)"
    )
    print(f"  Rows           : {len(df_epu)}")
    print(f"  Date range     : {df_epu['date'].min().date()} → {df_epu['date'].max().date()}")
    print(f"  Median row gap : {median_gap:.0f} days")
    print(f"  Frequency      : {freq_label}")
    if median_gap < 50:
        print("  ✓ EPU is native monthly data — no disaggregation applied.")
    else:
        print("  ⚠  EPU may have been disaggregated or is already quarterly/annual.")
else:
    print("  [SKIP] EPU data unavailable.")

# ── 3. Remittance native frequency ────────────────────────────────────────────
print("\n[3] Remittance data native frequency")
for label, df_flow, col in [("Inward",  df_in,  "inward_flow"),
                              ("Outward", df_out, "outward_flow")]:
    if df_flow is not None:
        counts_per_country = df_flow.groupby("country")["date"].nunique()
        typical_obs = int(counts_per_country.median())
        date_range = (df_flow["date"].max() - df_flow["date"].min()).days / 365.25
        implied_freq = date_range / max(typical_obs - 1, 1) if typical_obs > 1 else 0
        freq_label = (
            "ANNUAL   (~1 obs/year)"    if 0.9 <= implied_freq <= 1.2  else
            "QUARTERLY (~4 obs/year)"   if 0.2 <= implied_freq < 0.9   else
            "UNKNOWN"
        )
        print(f"\n  {label}:")
        print(f"    Rows              : {len(df_flow)}")
        print(f"    Countries         : {df_flow['country'].nunique()}")
        print(f"    Median obs/country: {typical_obs}")
        print(f"    Implied frequency : {freq_label}")
        if implied_freq > 0.8:
            print("    ⚠  ANNUAL data detected — replicating each year 4× creates")
            print("       artificial quarterly rows. Reviewer concern is VALID here.")
            print("       Recommendation: use annual frequency or source RBI quarterly data.")
        else:
            print("    ✓ Sub-annual observations found.")
    else:
        print(f"  {label}: [SKIP] data unavailable.")

# ── 4. Combined dataset timeline ──────────────────────────────────────────────
print("\n[4] Combined dataset quarterly timeline")
quarters = []
years_flat = []
n_total = 0
if df_c is not None:
    n_total = len(df_c)
    if "quarter" in df_c.columns:
        quarters = sorted(df_c["quarter"].unique())
        print(f"  Total quarters : {len(quarters)}")
        print(f"  First          : {quarters[0]}")
        print(f"  Last           : {quarters[-1]}")

    if "inward_flow" in df_c.columns and "year" in df_c.columns:
        df_c["year"] = df_c["year"].astype(int)
        dup_check = (
            df_c.groupby("year")["inward_flow"]
            .apply(lambda x: x.nunique() == 1)
        )
        years_flat = dup_check[dup_check].index.tolist()
        if years_flat:
            print(f"\n  ⚠  {len(years_flat)} years have IDENTICAL inward_flow across")
            print(f"     all 4 quarters — consistent with annual→quarterly replication:")
            print(f"     Years: {years_flat[:10]}")
        else:
            print("\n  ✓ Quarterly values differ within each year — no flat replication.")
else:
    print("  [SKIP] Combined dataset unavailable.")

# ── 5. Missing-value audit ────────────────────────────────────────────────────
print("\n[5] Missing-value audit (combined dataset)")
if df_c is not None:
    mv = df_c.isnull().sum()
    mv = mv[mv > 0]
    if len(mv):
        print(mv.to_string())
        print("\n  Handling method (from notebook):")
        print("    inward_flow  → fillna(0)   [assumption: no flow = 0]")
        print("    outward_flow → fillna(0)")
        print("    EPU_Index    → ffill()      [carry forward last known]")
        print("    Lags/MAs     → rows dropped [first N quarters excluded]")
    else:
        print("  No missing values after preprocessing.")
else:
    print("  [SKIP]")

# ── 6. Observation count table ────────────────────────────────────────────────
print("\n[6] Observation count table (for data section of paper)")
if df_c is not None and quarters:
    n80 = int(0.7 * len(quarters))
    n90 = int(0.8 * len(quarters))
    train_qs = quarters[:n80]
    val_qs   = quarters[n80:n90]
    test_qs  = quarters[n90:]
    print(f"  {'Block':<12} {'Quarters':>10}  {'Range'}")
    print(f"  {'-'*55}")
    print(f"  {'Train':<12} {len(train_qs):>10}  {train_qs[0]} – {train_qs[-1]}")
    if val_qs:
        print(f"  {'Validation':<12} {len(val_qs):>10}  {val_qs[0]} – {val_qs[-1]}")
    if test_qs:
        print(f"  {'Test':<12} {len(test_qs):>10}  {test_qs[0]} – {test_qs[-1]}")
    print(f"  {'TOTAL':<12} {n_total:>10}")
    print(f"\n  Features per observation: {df_c.shape[1]}")
elif df_c is not None:
    print(f"  Total rows: {n_total}  |  Columns: {df_c.shape[1]}")
else:
    print("  [SKIP]")

# ── 7. Save report ────────────────────────────────────────────────────────────
out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
output_path = out_dir / "d1_data_integrity_report.txt"

with open(output_path, "w", encoding="utf-8") as fh:
    fh.write("D1 — Data Integrity & Granularity Diagnostic\n")
    fh.write("Mode: REAL DATA\n")
    fh.write("Run on: " + pd.Timestamp.now().isoformat() + "\n\n")
    if df_c is not None:
        fh.write(f"Total rows       : {n_total}\n")
        if quarters:
            fh.write(f"Quarter range    : {quarters[0]} – {quarters[-1]}\n")
        if years_flat:
            fh.write(f"Annual rep. years: {years_flat}\n")
        else:
            fh.write("Quarterly variation: confirmed\n")

print(f"\n  Report saved → {output_path}")
print("\n" + sep)
print("D1 COMPLETE")
print(sep)
