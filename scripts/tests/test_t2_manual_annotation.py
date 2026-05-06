"""
TEST T2 — Manual Annotation Simulation & Inter-Rater Agreement
==============================================================
Reviewer requirement:
  • Construct a manually annotated sample (expert bilingual annotation).
  • Report Accuracy, Precision, Recall, and Cohen's Kappa (inter-rater).
  • Compare automated (VADER / mBERT) labels against the manual standard.

HOW TO USE ON KAGGLE:
  • Requires remittances_news_final.csv in /kaggle/working/ or /kaggle/input/.

MODES:
  Set ANNOTATED = False  → generates annotation template CSV (default)
  Set ANNOTATED = True   → computes Kappa/Accuracy after you fill the CSV

Output: t2_annotation_sample.csv  (MODE A)
        Accuracy / Kappa report   (MODE B)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score,
)

# ─────────────────────── CONFIG ───────────────────────────────────────────────
ANNOTATED        = False    # ← flip to True after filling manual_label column
SAMPLE_SIZE      = 150
STRATIFY_LANGS   = True
RANDOM_SEED      = 42
ANNOTATION_FILE  = "t2_annotation_sample.csv"
LABEL_MAP        = {0: "Positive", 1: "Negative", 2: "Neutral"}
# ─────────────────────────────────────────────────────────────────────────────

sep = "=" * 72
SEARCH_DIRS = ["/kaggle/working/", "/kaggle/input/", "./", "../"]

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

LANG_NAMES = {"en": "English", "hi": "Hindi",  "ta": "Tamil",  "te": "Telugu",
              "ml": "Malayalam","bn": "Bengali","pa": "Punjabi","gu": "Gujarati"}

# ── Load real data — abort if not found ───────────────────────────────────────
news_path = find("remittances_news_final.csv")
out       = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
ann_path  = out / ANNOTATION_FILE

if news_path is None:
    raise FileNotFoundError(
        "T2 requires remittances_news_final.csv in /kaggle/working/ or /kaggle/input/.\n"
        "Upload the news corpus CSV from your pipeline and rerun."
    )

print(sep)
print("T2 — MANUAL ANNOTATION & INTER-RATER AGREEMENT")
print("     Mode: REAL DATA")
print(sep)

df_all = pd.read_csv(news_path)
print(f"\n  Total corpus size: {len(df_all):,} articles")

# ── MODE A: Generate annotation template ─────────────────────────────────────
if not ANNOTATED:
    print(f"\n[MODE A] Generating annotation template ({SAMPLE_SIZE} articles)")

    if STRATIFY_LANGS:
        langs   = df_all["language"].unique()
        per_lang = max(5, SAMPLE_SIZE // len(langs))
        frames  = []
        for lang in langs:
            sub = df_all[df_all["language"] == lang]
            n   = min(per_lang, len(sub))
            frames.append(sub.sample(n=n, random_state=RANDOM_SEED))
        sample = pd.concat(frames).sample(frac=1, random_state=RANDOM_SEED).head(SAMPLE_SIZE)
    else:
        sample = df_all.sample(n=min(SAMPLE_SIZE, len(df_all)), random_state=RANDOM_SEED)

    print(f"  Sampled {len(sample)} articles:")
    for lang, cnt in sample["language"].value_counts().items():
        print(f"    {LANG_NAMES.get(lang, lang):<14} ({lang}): {cnt}")

    sample = sample.copy()

    # VADER predictions
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        sample["vader_label"] = sample["title"].apply(
            lambda t: 0 if vader.polarity_scores(str(t))["compound"] > 0.05
                      else (1 if vader.polarity_scores(str(t))["compound"] < -0.05
                            else 2))
        sample["vader_label_text"] = sample["vader_label"].map(LABEL_MAP)
        print("\n  ✓ VADER labels computed")
    except ImportError:
        pos_kw = {"record","high","surge","boost","growth","rise","increase","digital",
                   "benefit","opportunity","cuts","reduces","overtakes"}
        neg_kw = {"crisis","drop","fall","slash","bleak","tighten","deportation","lockdown",
                   "slowdown","decline","decrease","tension"}
        def kw_label(t):
            words = set(str(t).lower().split())
            if words & pos_kw:  return 0
            if words & neg_kw:  return 1
            return 2
        sample["vader_label"]      = sample["title"].apply(kw_label)
        sample["vader_label_text"] = sample["vader_label"].map(LABEL_MAP)
        print("\n  ⚠  vaderSentiment not installed — using keyword heuristic instead")

    # mBERT predictions — look for the real checkpoint
    # Try mbert_sentiment subdirs first (checkpoint-9384 is the final epoch)
    def find_mbert_dir(base_out):
        for name in ["mbert_sentiment/checkpoint-9384",
                     "mbert_sentiment/checkpoint-6256",
                     "mbert_sentiment/checkpoint-3128",
                     "mbert_results"]:
            p = base_out / name
            if p.exists():
                return p
        return None

    mbert_dir = find_mbert_dir(out)
    if mbert_dir is not None and mbert_dir.exists():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(str(mbert_dir))
            model     = AutoModelForSequenceClassification.from_pretrained(str(mbert_dir))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            mbert_labels, mbert_confs = [], []
            for title in sample["title"].fillna("").tolist():
                inputs = tokenizer(title, return_tensors="pt", truncation=True,
                                   max_length=128, padding=True).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred   = int(np.argmax(probs))
                mbert_labels.append(pred)
                mbert_confs.append(float(probs[pred]))
            sample["mbert_label"]      = mbert_labels
            sample["mbert_label_text"] = [LABEL_MAP[l] for l in mbert_labels]
            sample["mbert_confidence"] = mbert_confs
            print("  ✓ mBERT labels computed from checkpoint")
        except Exception as e:
            sample["mbert_label"]      = np.nan
            sample["mbert_label_text"] = ""
            sample["mbert_confidence"] = np.nan
            print(f"  ⚠  mBERT prediction failed: {e}")
    else:
        sample["mbert_label"]      = np.nan
        sample["mbert_label_text"] = ""
        sample["mbert_confidence"] = np.nan
        print("  ⚠  No mBERT checkpoint found — mBERT labels left blank")

    sample["manual_label"]      = ""
    sample["manual_label_text"] = ""

    cols_out = ["title", "language", "seendate", "url", "source_api",
                "relevance_score",
                "vader_label", "vader_label_text",
                "mbert_label", "mbert_label_text", "mbert_confidence",
                "manual_label", "manual_label_text"]
    cols_out = [c for c in cols_out if c in sample.columns]
    sample[cols_out].to_csv(ann_path, index=False, encoding="utf-8-sig")

    print(f"\n  ✓ Annotation template saved → {ann_path}")
    print("""
  ── NEXT STEPS ──────────────────────────────────────────────────────────
  1. Download  t2_annotation_sample.csv  (UTF-8-BOM for Excel compatibility)
  2. Have ≥ 2 bilingual annotators independently fill the 'manual_label'
     column with:   0 = Positive   1 = Negative   2 = Neutral
     Save each annotator's file separately (annotator1.csv, annotator2.csv)
  3. Upload the annotated file back to /kaggle/working/
  4. Set  ANNOTATED = True  and rerun this cell to get Accuracy/Kappa.
  ────────────────────────────────────────────────────────────────────────
""")

# ── MODE B: Compute metrics against manual labels ─────────────────────────────
else:
    print(f"\n[MODE B] Computing inter-rater agreement against manual labels")

    if not ann_path.exists():
        print(f"  ❌  {ann_path} not found. Run MODE A first.")
        raise SystemExit(1)

    df_ann = pd.read_csv(ann_path)
    df_ann = df_ann[df_ann["manual_label"].notna() & (df_ann["manual_label"] != "")]
    df_ann["manual_label"] = df_ann["manual_label"].astype(int)
    n = len(df_ann)
    print(f"  Annotated rows: {n}")

    if n < 10:
        print("  ⚠  Too few annotations (< 10).  Please fill more rows.")
        raise SystemExit(1)

    y_true = df_ann["manual_label"].values

    if "vader_label" in df_ann.columns and df_ann["vader_label"].notna().sum() > 5:
        df_v = df_ann.dropna(subset=["vader_label"])
        y_v  = df_v["vader_label"].astype(int).values
        y_m  = df_v["manual_label"].values
        print("\n  VADER vs Manual Annotation")
        print(f"    Accuracy  : {accuracy_score(y_m, y_v):.4f}")
        print(f"    Precision : {precision_score(y_m, y_v, average='weighted', zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(y_m, y_v,    average='weighted', zero_division=0):.4f}")
        print(f"    F1        : {f1_score(y_m, y_v,        average='weighted', zero_division=0):.4f}")
        print(f"    Cohen κ   : {cohen_kappa_score(y_m, y_v):.4f}")
        print(f"\n    Classification Report (VADER):")
        print(classification_report(y_m, y_v,
                                    target_names=["Positive","Negative","Neutral"],
                                    zero_division=0))

    if "mbert_label" in df_ann.columns and df_ann["mbert_label"].notna().sum() > 5:
        df_m  = df_ann.dropna(subset=["mbert_label"])
        y_b   = df_m["mbert_label"].astype(int).values
        y_m2  = df_m["manual_label"].values
        print("\n  mBERT vs Manual Annotation")
        print(f"    Accuracy  : {accuracy_score(y_m2, y_b):.4f}")
        print(f"    Precision : {precision_score(y_m2, y_b, average='weighted', zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(y_m2, y_b,    average='weighted', zero_division=0):.4f}")
        print(f"    F1        : {f1_score(y_m2, y_b,        average='weighted', zero_division=0):.4f}")
        print(f"    Cohen κ   : {cohen_kappa_score(y_m2, y_b):.4f}")
        print(f"\n    Classification Report (mBERT):")
        print(classification_report(y_m2, y_b,
                                    target_names=["Positive","Negative","Neutral"],
                                    zero_division=0))

        df_both = df_ann.dropna(subset=["vader_label","mbert_label"])
        if len(df_both) > 5:
            y_vv = df_both["vader_label"].astype(int).values
            y_bb = df_both["mbert_label"].astype(int).values
            print(f"\n  VADER vs mBERT (inter-system agreement):")
            print(f"    Cohen κ   : {cohen_kappa_score(y_vv, y_bb):.4f}")
            print(f"    Agreement : {(y_vv == y_bb).mean()*100:.1f}%")

    print("""
  ── KAPPA INTERPRETATION ──
  κ < 0.00   : Less than chance
  κ 0.00–0.20: Slight
  κ 0.21–0.40: Fair
  κ 0.41–0.60: Moderate
  κ 0.61–0.80: Substantial
  κ 0.81–1.00: Almost perfect
""")

print("\n" + sep)
print("T2 COMPLETE")
print(sep)
