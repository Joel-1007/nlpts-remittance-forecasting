# KAGGLE CELL 2: Data Loading from Kaggle Input Directory
# Week 1: Data Sourcing (5 hours estimated)
# ENHANCED FOR Q1 JOURNAL PUBLICATION STANDARDS

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import logging

print("=== LOADING DATA FROM KAGGLE INPUT ===")
print("Scanning /kaggle/input/ directory...\n")

# Resume logging from Cell 1
try:
    log_filename = config['log_filename']
    logging.info("="*70)
    logging.info("CELL 2: DATA LOADING STARTED")
    logging.info("="*70)
except:
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting data loading (logging config from Cell 1 not found)")

# List all available datasets
input_path = "/kaggle/input/"
if os.path.exists(input_path):
    datasets = os.listdir(input_path)
    print(f"Available datasets: {len(datasets)}")
    logging.info(f"Found {len(datasets)} datasets in {input_path}")
    for dataset in datasets[:10]:
        print(f"  - {dataset}")
        logging.info(f"  Dataset: {dataset}")
    if len(datasets) > 10:
        print(f"  ... and {len(datasets) - 10} more")
        logging.info(f"  ... and {len(datasets) - 10} more datasets")
else:
    print("⚠ No input directory found. Please add datasets to your Kaggle notebook.")
    logging.warning("No input directory found at /kaggle/input/")
    datasets = []

print("\n=== SEARCHING FOR EXCEL FILES ===")

# Search for Excel files in all input directories
excel_files = []
for dataset in datasets:
    dataset_path = os.path.join(input_path, dataset)
    if os.path.isdir(dataset_path):
        files = glob.glob(os.path.join(dataset_path, "*.xlsx")) + \
                glob.glob(os.path.join(dataset_path, "*.xls"))
        excel_files.extend(files)

print(f"Found {len(excel_files)} Excel file(s):")
logging.info(f"Found {len(excel_files)} Excel files")
for f in excel_files:
    print(f"  • {os.path.basename(f)}")
    logging.info(f"  File: {os.path.basename(f)} at {f}")

# ============================================================================
# DATA QUALITY METRICS FUNCTION (Q1 JOURNAL REQUIREMENT)
# ============================================================================
def compute_data_quality_metrics(df, name):
    """
    Compute comprehensive data quality metrics for publication.
    Required for methodology transparency in Q1 journals.
    """
    logging.info(f"Computing data quality metrics for: {name}")
    
    metrics = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'completeness': round(1 - df.isnull().sum().sum() / (len(df) * len(df.columns)), 4),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_pct': round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    
    # For numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Count outliers using IQR method
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count += ((df[col] < (Q1 - 1.5 * IQR)) | 
                             (df[col] > (Q3 + 1.5 * IQR))).sum()
        
        metrics['outliers_iqr'] = outlier_count
        metrics['outlier_pct'] = round(outlier_count / (len(df) * len(numeric_cols)) * 100, 2)
        metrics['zero_variance_cols'] = sum([df[col].std() == 0 for col in numeric_cols])
        metrics['negative_values'] = sum([(df[col] < 0).sum() for col in numeric_cols])
    else:
        metrics['outliers_iqr'] = 0
        metrics['outlier_pct'] = 0
        metrics['zero_variance_cols'] = 0
        metrics['negative_values'] = 0
    
    # Missing data pattern
    missing_per_col = df.isnull().sum()
    metrics['cols_with_missing'] = (missing_per_col > 0).sum()
    metrics['max_missing_pct'] = round(missing_per_col.max() / len(df) * 100, 2) if len(df) > 0 else 0
    
    logging.info(f"  Completeness: {metrics['completeness']*100:.2f}%")
    logging.info(f"  Duplicates: {metrics['duplicate_rows']} ({metrics['duplicate_pct']:.2f}%)")
    logging.info(f"  Outliers: {metrics['outliers_iqr']} ({metrics['outlier_pct']:.2f}%)")
    
    return metrics

# ============================================================================
# TEMPORAL COVERAGE ANALYSIS (Q1 JOURNAL REQUIREMENT)
# ============================================================================
def find_temporal_gaps(df, date_col, freq='Q', name='Dataset'):
    """
    Find missing periods in time series.
    Essential for explaining interpolation/imputation choices.
    """
    logging.info(f"Analyzing temporal coverage for: {name}")
    
    df_sorted = df.sort_values(date_col).copy()
    min_date = df_sorted[date_col].min()
    max_date = df_sorted[date_col].max()

    # Map old aliases to new pandas 2.x aliases
    freq_map = {'M': 'ME', 'Q': 'QE', 'Y': 'YE'}
    period_freq_map = {'M': 'M', 'Q': 'Q', 'Y': 'Y'}  # Period freq aliases unchanged
    range_freq = freq_map.get(freq, freq)
    period_freq = period_freq_map.get(freq, freq)

    full_range = pd.date_range(min_date, max_date, freq=range_freq)
    existing_periods = df_sorted[date_col].dt.to_period(period_freq).unique()
    full_periods = pd.PeriodIndex(full_range, freq=period_freq)

    missing = sorted(set(full_periods) - set(existing_periods))
    
    coverage_stats = {
        'name': name,
        'frequency': freq,
        'start_date': str(min_date),
        'end_date': str(max_date),
        'total_periods': len(full_periods),
        'observed_periods': len(existing_periods),
        'missing_periods': len(missing),
        'coverage_pct': round(len(existing_periods) / len(full_periods) * 100, 2) if len(full_periods) > 0 else 0
    }
    
    logging.info(f"  Date range: {min_date} to {max_date}")
    logging.info(f"  Coverage: {coverage_stats['coverage_pct']:.2f}% ({coverage_stats['observed_periods']}/{coverage_stats['total_periods']} periods)")
    logging.info(f"  Missing: {len(missing)} periods")
    
    return missing, coverage_stats

# Function to find file by partial name
def find_file(partial_name):
    """Find file in Kaggle input by partial name match"""
    for f in excel_files:
        if partial_name.lower() in os.path.basename(f).lower():
            return f
    return None

print("\n=== LOADING DATASET 3: INDIA EPU DATA ===")
logging.info("Loading India EPU (Economic Policy Uncertainty) data...")

# Try to find EPU file
epu_file = find_file("policy") or find_file("uncertainty") or find_file("epu")

if epu_file:
    try:
        print(f"Loading: {os.path.basename(epu_file)}")
        logging.info(f"Reading file: {epu_file}")
        
        df_epu = pd.read_excel(epu_file)
        print(f"✓ Loaded: {df_epu.shape[0]} rows, {df_epu.shape[1]} columns")
        logging.info(f"Raw EPU data: {df_epu.shape[0]} rows × {df_epu.shape[1]} columns")
        print(f"  Columns: {list(df_epu.columns)[:5]}")
        
        # Process date columns - handle both string and integer column names
        if 'Year' in df_epu.columns and 'Month' in df_epu.columns:
            # Clean and convert Year and Month to integers
            df_epu['Year'] = pd.to_numeric(df_epu['Year'], errors='coerce')
            df_epu['Month'] = pd.to_numeric(df_epu['Month'], errors='coerce')
            
            # Remove rows where Year or Month are missing
            rows_before = len(df_epu)
            df_epu = df_epu.dropna(subset=['Year', 'Month'])
            rows_dropped = rows_before - len(df_epu)
            if rows_dropped > 0:
                logging.warning(f"Dropped {rows_dropped} rows with missing Year/Month")
            
            # Convert to integers
            df_epu['Year'] = df_epu['Year'].astype(int)
            df_epu['Month'] = df_epu['Month'].astype(int)
            
            # Create date column
            df_epu['date'] = pd.to_datetime(
                df_epu['Year'].astype(str) + '-' + 
                df_epu['Month'].astype(str).str.zfill(2) + '-01',
                format='%Y-%m-%d'
            )
            df_epu['quarter'] = df_epu['date'].dt.to_period('Q')
            print(f"  Date range: {df_epu['date'].min()} to {df_epu['date'].max()}")
            logging.info(f"Date range: {df_epu['date'].min()} to {df_epu['date'].max()}")
        
        print(f"✓ EPU data ready: {len(df_epu)} rows")
        
    except Exception as e:
        print(f"✗ Error loading EPU data: {e}")
        logging.error(f"Error loading EPU data: {e}")
        import traceback
        traceback.print_exc()
        logging.error(traceback.format_exc())
        df_epu = None
else:
    print("⚠ EPU file not found in input")
    logging.warning("EPU file not found in input directory")
    df_epu = None

print("\n=== LOADING INWARD REMITTANCE FLOWS ===")
logging.info("Loading inward remittance flows...")

inward_file = find_file("inward") or find_file("remittance")

if inward_file:
    try:
        print(f"Loading: {os.path.basename(inward_file)}")
        logging.info(f"Reading file: {inward_file}")
        
        df_inward_raw = pd.read_excel(inward_file)
        print(f"✓ Raw data loaded: {df_inward_raw.shape[0]} rows, {df_inward_raw.shape[1]} columns")
        logging.info(f"Raw inward data: {df_inward_raw.shape[0]} rows × {df_inward_raw.shape[1]} columns")
        
        # Display structure
        print(f"\nData structure:")
        print(f"  First column: {df_inward_raw.columns[0]}")
        print(f"  Year columns (sample): {list(df_inward_raw.columns[1:6])}")
        
        # Identify year columns (they should be integers or year strings)
        year_columns = [col for col in df_inward_raw.columns[1:] 
                       if isinstance(col, (int, float)) or 
                       (isinstance(col, str) and col.isdigit())]
        
        print(f"  Detected {len(year_columns)} year columns from {year_columns[0]} to {year_columns[-1]}")
        logging.info(f"Detected {len(year_columns)} year columns: {year_columns[0]} to {year_columns[-1]}")
        
        # Reshape from wide to long format
        id_col = df_inward_raw.columns[0]  # Country/region column
        
        # Melt the dataframe
        df_inward = pd.melt(
            df_inward_raw,
            id_vars=[id_col],
            value_vars=year_columns,
            var_name='year',
            value_name='inward_flow'
        )
        
        # Clean the data
        df_inward['year'] = pd.to_numeric(df_inward['year'], errors='coerce')
        df_inward = df_inward.dropna(subset=['year'])
        df_inward['year'] = df_inward['year'].astype(int)
        
        # Create date and quarter
        df_inward['date'] = pd.to_datetime(df_inward['year'].astype(str) + '-01-01')
        df_inward['quarter'] = df_inward['date'].dt.to_period('Q')
        
        # Rename country column
        df_inward = df_inward.rename(columns={id_col: 'country'})
        
        # Convert flow values to numeric
        df_inward['inward_flow'] = pd.to_numeric(df_inward['inward_flow'], errors='coerce')
        
        # Remove rows with missing flow values
        rows_before = len(df_inward)
        df_inward = df_inward.dropna(subset=['inward_flow'])
        rows_dropped = rows_before - len(df_inward)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows with missing inward_flow values")
        
        print(f"\n✓ Reshaped inward data: {len(df_inward)} rows")
        print(f"  Date range: {df_inward['date'].min()} to {df_inward['date'].max()}")
        print(f"  Countries/regions: {df_inward['country'].nunique()}")
        print(f"  Sample countries: {df_inward['country'].head(3).tolist()}")
        
        logging.info(f"Reshaped inward data: {len(df_inward)} rows")
        logging.info(f"Countries/regions: {df_inward['country'].nunique()}")
        
    except Exception as e:
        print(f"✗ Error loading inward flows: {e}")
        logging.error(f"Error loading inward flows: {e}")
        import traceback
        traceback.print_exc()
        logging.error(traceback.format_exc())
        df_inward = None
else:
    print("⚠ Inward remittance file not found")
    logging.warning("Inward remittance file not found in input directory")
    df_inward = None

print("\n=== LOADING OUTWARD REMITTANCE FLOWS ===")
logging.info("Loading outward remittance flows...")

outward_file = find_file("outward")

if outward_file:
    try:
        print(f"Loading: {os.path.basename(outward_file)}")
        logging.info(f"Reading file: {outward_file}")
        
        df_outward_raw = pd.read_excel(outward_file)
        print(f"✓ Raw data loaded: {df_outward_raw.shape[0]} rows, {df_outward_raw.shape[1]} columns")
        logging.info(f"Raw outward data: {df_outward_raw.shape[0]} rows × {df_outward_raw.shape[1]} columns")
        
        # Display structure
        print(f"\nData structure:")
        print(f"  First column: {df_outward_raw.columns[0]}")
        print(f"  Year columns (sample): {list(df_outward_raw.columns[1:6])}")
        
        # Identify year columns - handle both int and string years
        year_columns = []
        for col in df_outward_raw.columns[1:]:
            # Try to convert to year
            if isinstance(col, (int, float)):
                year_columns.append(col)
            elif isinstance(col, str):
                # Remove any spaces and check if it's a number
                col_clean = col.strip()
                if col_clean.isdigit():
                    year_columns.append(col)
        
        print(f"  Detected {len(year_columns)} year columns")
        logging.info(f"Detected {len(year_columns)} year columns")
        
        # Reshape from wide to long format
        id_col = df_outward_raw.columns[0]
        
        # Melt the dataframe
        df_outward = pd.melt(
            df_outward_raw,
            id_vars=[id_col],
            value_vars=year_columns,
            var_name='year',
            value_name='outward_flow'
        )
        
        # Clean the data
        df_outward['year'] = pd.to_numeric(df_outward['year'], errors='coerce')
        df_outward = df_outward.dropna(subset=['year'])
        df_outward['year'] = df_outward['year'].astype(int)
        
        # Create date and quarter
        df_outward['date'] = pd.to_datetime(df_outward['year'].astype(str) + '-01-01')
        df_outward['quarter'] = df_outward['date'].dt.to_period('Q')
        
        # Rename country column
        df_outward = df_outward.rename(columns={id_col: 'country'})
        
        # Convert flow values to numeric
        df_outward['outward_flow'] = pd.to_numeric(df_outward['outward_flow'], errors='coerce')
        
        # Remove rows with missing flow values
        rows_before = len(df_outward)
        df_outward = df_outward.dropna(subset=['outward_flow'])
        rows_dropped = rows_before - len(df_outward)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows with missing outward_flow values")
        
        print(f"\n✓ Reshaped outward data: {len(df_outward)} rows")
        print(f"  Date range: {df_outward['date'].min()} to {df_outward['date'].max()}")
        print(f"  Countries/regions: {df_outward['country'].nunique()}")
        print(f"  Sample countries: {df_outward['country'].head(3).tolist()}")
        
        logging.info(f"Reshaped outward data: {len(df_outward)} rows")
        logging.info(f"Countries/regions: {df_outward['country'].nunique()}")
        
    except Exception as e:
        print(f"✗ Error loading outward flows: {e}")
        logging.error(f"Error loading outward flows: {e}")
        import traceback
        traceback.print_exc()
        logging.error(traceback.format_exc())
        df_outward = None
else:
    print("⚠ Outward remittance file not found")
    logging.warning("Outward remittance file not found in input directory")
    df_outward = None

# ============================================================================
# DATA QUALITY ASSESSMENT (Q1 JOURNAL REQUIREMENT)
# ============================================================================
print("\n=== DATA QUALITY ASSESSMENT ===")
logging.info("="*70)
logging.info("DATA QUALITY ASSESSMENT")
logging.info("="*70)

quality_metrics = []

if df_epu is not None:
    epu_quality = compute_data_quality_metrics(df_epu, "EPU Index")
    quality_metrics.append(epu_quality)
    print(f"\nEPU Data Quality:")
    print(f"  Completeness: {epu_quality['completeness']*100:.2f}%")
    print(f"  Duplicates: {epu_quality['duplicate_rows']} ({epu_quality['duplicate_pct']:.2f}%)")
    print(f"  Outliers: {epu_quality['outliers_iqr']} ({epu_quality['outlier_pct']:.2f}%)")

if df_inward is not None:
    inward_quality = compute_data_quality_metrics(df_inward, "Inward Remittances")
    quality_metrics.append(inward_quality)
    print(f"\nInward Remittance Quality:")
    print(f"  Completeness: {inward_quality['completeness']*100:.2f}%")
    print(f"  Duplicates: {inward_quality['duplicate_rows']} ({inward_quality['duplicate_pct']:.2f}%)")
    print(f"  Outliers: {inward_quality['outliers_iqr']} ({inward_quality['outlier_pct']:.2f}%)")

if df_outward is not None:
    outward_quality = compute_data_quality_metrics(df_outward, "Outward Remittances")
    quality_metrics.append(outward_quality)
    print(f"\nOutward Remittance Quality:")
    print(f"  Completeness: {outward_quality['completeness']*100:.2f}%")
    print(f"  Duplicates: {outward_quality['duplicate_rows']} ({outward_quality['duplicate_pct']:.2f}%)")
    print(f"  Outliers: {outward_quality['outliers_iqr']} ({outward_quality['outlier_pct']:.2f}%)")

# Save quality report
if quality_metrics:
    quality_report = pd.DataFrame(quality_metrics)
    quality_report.to_csv('/kaggle/working/data_quality_report.csv', index=False)
    print(f"\n✓ Data quality report saved to: /kaggle/working/data_quality_report.csv")
    logging.info("Data quality report saved")

# ============================================================================
# TEMPORAL COVERAGE ANALYSIS (Q1 JOURNAL REQUIREMENT)
# ============================================================================
print("\n=== TEMPORAL COVERAGE ANALYSIS ===")
logging.info("="*70)
logging.info("TEMPORAL COVERAGE ANALYSIS")
logging.info("="*70)

coverage_stats_list = []

if df_epu is not None and 'date' in df_epu.columns:
    epu_gaps, epu_coverage = find_temporal_gaps(df_epu, 'date', freq='M', name='EPU Index')
    coverage_stats_list.append(epu_coverage)
    print(f"\nEPU Index Coverage:")
    print(f"  Date range: {epu_coverage['start_date']} to {epu_coverage['end_date']}")
    print(f"  Coverage: {epu_coverage['coverage_pct']:.2f}% ({epu_coverage['observed_periods']}/{epu_coverage['total_periods']} months)")
    print(f"  Missing: {len(epu_gaps)} months")
    if len(epu_gaps) > 0:
        print(f"  First missing: {epu_gaps[:5]}" + (" ..." if len(epu_gaps) > 5 else ""))

if df_inward is not None and 'date' in df_inward.columns:
    inward_gaps, inward_coverage = find_temporal_gaps(df_inward, 'date', freq='Y', name='Inward Remittances')
    coverage_stats_list.append(inward_coverage)
    print(f"\nInward Remittance Coverage:")
    print(f"  Date range: {inward_coverage['start_date']} to {inward_coverage['end_date']}")
    print(f"  Coverage: {inward_coverage['coverage_pct']:.2f}% ({inward_coverage['observed_periods']}/{inward_coverage['total_periods']} years)")
    print(f"  Missing: {len(inward_gaps)} years")
    if len(inward_gaps) > 0:
        print(f"  Missing years: {inward_gaps[:10]}" + (" ..." if len(inward_gaps) > 10 else ""))

if df_outward is not None and 'date' in df_outward.columns:
    outward_gaps, outward_coverage = find_temporal_gaps(df_outward, 'date', freq='Y', name='Outward Remittances')
    coverage_stats_list.append(outward_coverage)
    print(f"\nOutward Remittance Coverage:")
    print(f"  Date range: {outward_coverage['start_date']} to {outward_coverage['end_date']}")
    print(f"  Coverage: {outward_coverage['coverage_pct']:.2f}% ({outward_coverage['observed_periods']}/{outward_coverage['total_periods']} years)")
    print(f"  Missing: {len(outward_gaps)} years")
    if len(outward_gaps) > 0:
        print(f"  Missing years: {outward_gaps[:10]}" + (" ..." if len(outward_gaps) > 10 else ""))

# Save coverage report
if coverage_stats_list:
    coverage_report = pd.DataFrame(coverage_stats_list)
    coverage_report.to_csv('/kaggle/working/temporal_coverage_report.csv', index=False)
    print(f"\n✓ Temporal coverage report saved to: /kaggle/working/temporal_coverage_report.csv")
    logging.info("Temporal coverage report saved")

# Data validation
print("\n=== DATA VALIDATION ===")

all_loaded = df_epu is not None and df_inward is not None and df_outward is not None

if all_loaded:
    print("✅ ALL DATA FILES LOADED SUCCESSFULLY!")
    logging.info("All data files loaded successfully")
    
    print("\n=== DATASET SUMMARY ===")
    print(f"EPU Data: {len(df_epu)} rows from {df_epu['date'].min()} to {df_epu['date'].max()}")
    print(f"Inward Flows: {len(df_inward)} rows, {df_inward['country'].nunique()} countries")
    print(f"Outward Flows: {len(df_outward)} rows, {df_outward['country'].nunique()} countries")
    
    # Save to Kaggle working directory
    df_epu.to_csv('/kaggle/working/epu_data.csv', index=False)
    print("\n✓ Saved: /kaggle/working/epu_data.csv")
    logging.info("Saved: /kaggle/working/epu_data.csv")
    
    df_inward.to_csv('/kaggle/working/inward_flows.csv', index=False)
    print("✓ Saved: /kaggle/working/inward_flows.csv")
    logging.info("Saved: /kaggle/working/inward_flows.csv")
    
    df_outward.to_csv('/kaggle/working/outward_flows.csv', index=False)
    print("✓ Saved: /kaggle/working/outward_flows.csv")
    logging.info("Saved: /kaggle/working/outward_flows.csv")
    
    # Display sample data
    print("\n=== SAMPLE DATA PREVIEW ===")
    print("\nEPU Data (first 3 rows):")
    print(df_epu.head(3))
    
    print("\nInward Flows (first 3 rows):")
    print(df_inward.head(3))
    
    print("\nOutward Flows (first 3 rows):")
    print(df_outward.head(3))
    
else:
    print("❌ SOME DATA FILES FAILED TO LOAD")
    logging.error("Some data files failed to load")
    print("\nPlease check the error messages above and verify:")
    print("  1. Excel files are properly formatted")
    print("  2. Files contain expected columns")
    print("  3. Date/year information is present")

print("\n" + "="*70)
print("DATA LOADING COMPLETE")
print("="*70)

logging.info("="*70)
logging.info("CELL 2: DATA LOADING COMPLETED")
logging.info(f"EPU loaded: {df_epu is not None}")
logging.info(f"Inward flows loaded: {df_inward is not None}")
logging.info(f"Outward flows loaded: {df_outward is not None}")
logging.info("="*70)

print("\n=== Q1 JOURNAL DOCUMENTATION GENERATED ===")
print("The following reports are ready for Methods section:")
print("  • data_quality_report.csv - Completeness, duplicates, outliers")
print("  • temporal_coverage_report.csv - Date ranges, gaps, coverage %")
print("\nThese reports provide transparency required by reviewers.")

print("\n✓ Ready for Cell 3: Data Preprocessing and Feature Engineering")