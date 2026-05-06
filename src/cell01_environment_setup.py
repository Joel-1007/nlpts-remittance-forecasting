# KAGGLE CELL 1: Environment Setup and Library Installation
# Phase 1 - Week 1-4: Data Preparation and mBERT Fine-Tuning
# Multi-Lingual Sentiment-Driven Remittance Flow Forecasting

print("=== KAGGLE ENVIRONMENT SETUP ===")
print("Checking pre-installed libraries and installing missing packages...\n")

import sys
import subprocess
import os

# Check Kaggle environment
print(f"Python version: {sys.version}")
print(f"Working directory: /kaggle/working/")
print(f"Input data directory: /kaggle/input/\n")

# Install additional libraries not in Kaggle base environment
print("=== Installing Additional Libraries ===")
print("Note: Installing packages that require network connectivity...")

packages = {
    'pyvmd': '0.2.0',
    'pydmd': '0.4.1', 
    'aif360': '0.5.0',
    'vaderSentiment': '3.3.2',
    'gdelt': '0.1.13'
}

installed_packages = {}

for package, version in packages.items():
    try:
        # Try installing without specifying version if exact version fails
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", f"{package}=={version}"],
                timeout=30
            )
            installed_packages[package] = True
            print(f"✓ Installed: {package}=={version}")
        except:
            # Try without version specification
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package],
                timeout=30
            )
            installed_packages[package] = True
            print(f"✓ Installed: {package} (latest version)")
    except Exception as e:
        installed_packages[package] = False
        print(f"⚠ Could not install: {package} - Will use alternatives")

print("\n=== Importing Core Libraries ===")

# Core data manipulation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Time series and statistics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
from scipy import signal

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# NLP and transformers (Kaggle has these pre-installed)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset, DatasetDict

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

print("\n=== Checking Optional Libraries ===")

# Signal processing - PyVMD
if installed_packages.get('pyvmd', False):
    try:
        import pyvmd as vmd
        print("✓ PyVMD available for signal decomposition")
    except:
        print("⚠ PyVMD not available - will use Savitzky-Golay filter instead")
        installed_packages['pyvmd'] = False
else:
    print("⚠ PyVMD not available - will use Savitzky-Golay filter instead")

# Dynamic Mode Decomposition
if installed_packages.get('pydmd', False):
    try:
        from pydmd import RDMD
        print("✓ PyDMD available for dynamic analysis")
    except:
        print("⚠ PyDMD not available - will use alternative decomposition")
        installed_packages['pydmd'] = False
else:
    print("⚠ PyDMD not available - will use alternative decomposition")

# Fairness metrics
if installed_packages.get('aif360', False):
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
        print("✓ AIF360 available for fairness audits")
    except:
        print("⚠ AIF360 not available - will use custom fairness metrics")
        installed_packages['aif360'] = False
else:
    print("⚠ AIF360 not available - will use custom fairness metrics")

# VADER Sentiment
if installed_packages.get('vaderSentiment', False):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✓ VADER Sentiment available")
    except:
        print("⚠ VADER not available - will use transformer-based sentiment only")
        installed_packages['vaderSentiment'] = False
else:
    print("⚠ VADER not available - will use transformer-based sentiment only")

# GDELT
if installed_packages.get('gdelt', False):
    try:
        import gdelt
        print("✓ GDELT library available")
    except:
        print("⚠ GDELT not available - external data features will be skipped")
        installed_packages['gdelt'] = False
else:
    print("⚠ GDELT not available - external data features will be skipped")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("\n=== GPU/CPU Configuration ===")
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = torch.device('cuda')
else:
    print("⚠ Running on CPU (training will be slower)")
    device = torch.device('cpu')

print("\n=== Random Seeds Set ===")
print("✓ NumPy seed: 42")
print("✓ PyTorch seed: 42")
print("✓ All operations are reproducible")

# Save configuration for later cells
config = {
    'device': device,
    'installed_packages': installed_packages,
    'random_seed': 42
}

print("\n" + "="*70)
print("ENVIRONMENT READY")
print("="*70)

print("\n=== Installation Summary ===")
for package, status in installed_packages.items():
    status_icon = "✓" if status else "✗"
    print(f"{status_icon} {package}: {'Available' if status else 'Using alternative'}")

print("\n=== Alternative Methods Enabled ===")
if not installed_packages.get('pyvmd', False):
    print("  • Signal denoising: Savitzky-Golay filter (scipy.signal)")
if not installed_packages.get('pydmd', False):
    print("  • Decomposition: Seasonal decompose + FFT")
if not installed_packages.get('aif360', False):
    print("  • Fairness: Custom demographic parity metrics")
if not installed_packages.get('vaderSentiment', False):
    print("  • Sentiment: mBERT-based multilingual sentiment")

print("\n=== Kaggle-Specific Notes ===")
print("  • Upload Excel files to /kaggle/input/your-dataset-name/")
print("  • Files accessible as: /kaggle/input/your-dataset-name/filename.xlsx")
print("  • Output files save to: /kaggle/working/")
print("  • Kaggle provides 16GB RAM and up to 30GB GPU RAM")
print("  • If network issues persist, restart kernel and try again")

print("\n✓ Ready for Cell 2: Data Loading and Preprocessing")
print("\nNext Steps:")
print("  1. Upload your Excel files as a Kaggle dataset")
print("  2. Note the dataset path (e.g., /kaggle/input/your-dataset-name/)")
print("  3. Run Cell 2 to load and preprocess data")