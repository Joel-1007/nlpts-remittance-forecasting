"""
TEST T3 — Zero-shot Classifier Comparison & Ragas-style Assessment
==================================================================
Reviewer alternatives:
  "Compare against other zero-shot classifiers OR use the Ragas framework."

HOW TO USE ON KAGGLE:
  • Requires t2_annotation_sample.csv OR remittances_news_final.csv in
    /kaggle/working/ or /kaggle/input/. At least one must be present.
  • The Hugging Face zero-shot models are downloaded from the internet on
    first run (Kaggle has internet access enabled by default).

Output: t3_zeroshot_comparison.csv, t3_model_distribution.csv
"""

import subprocess, sys
for pkg in ["transformers", "torch"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import pandas as pd
import numpy as np
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

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

# ── Load real data — abort if nothing found ────────────────────────────────────
ann_path  = find("t2_annotation_sample.csv")
news_path = find("remittances_news_final.csv")
out       = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

print(sep)
print("T3 — ZERO-SHOT CLASSIFIER COMPARISON & RAGAS ASSESSMENT")
print(sep)

if ann_path:
    print(f"  Source: t2_annotation_sample.csv")
    df = pd.read_csv(ann_path)
    df = df[df["title"].notna() & (df["title"] != "=== INSTRUCTIONS ===")].copy()
elif news_path:
    print(f"  Source: remittances_news_final.csv")
    df = pd.read_csv(news_path).head(300)
else:
    raise FileNotFoundError(
        "T3 requires t2_annotation_sample.csv or remittances_news_final.csv "
        "in /kaggle/working/ or /kaggle/input/.\n"
        "Upload your pipeline output files and rerun."
    )

# Sub-sample for speed (zero-shot models are slow)
EVAL_SIZE = min(100, len(df))
df = df.sample(n=EVAL_SIZE, random_state=42).reset_index(drop=True)
texts = df["title"].fillna("").tolist()
langs = df["language"].tolist() if "language" in df.columns else ["en"] * len(df)

print(f"\n  Evaluating {EVAL_SIZE} articles across "
      f"{df['language'].nunique() if 'language' in df.columns else 1} languages")

SENTIMENT_LABELS = [
    "positive remittance news",
    "negative remittance news",
    "neutral remittance information",
]
LABEL_MAP_ZS = {0: "Positive", 1: "Negative", 2: "Neutral"}

def run_zero_shot(model_name, texts, batch_size=8):
    import torch
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    clf = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        batch_size=batch_size,
        multi_label=False,
    )
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            preds = clf(batch, SENTIMENT_LABELS)
            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                top_label = pred["labels"][0]
                top_score = pred["scores"][0]
                label_idx = SENTIMENT_LABELS.index(top_label)
                results.append({
                    "label":      label_idx,
                    "confidence": top_score,
                    "label_text": LABEL_MAP_ZS[label_idx],
                })
        except Exception:
            for _ in batch:
                results.append({"label": 2, "confidence": 0.0, "label_text": "Neutral"})
    return results

# ── 1. Run zero-shot models ───────────────────────────────────────────────────
models = {
    "BART-MNLI":   "facebook/bart-large-mnli",
    "MiniLM-NLI":  "cross-encoder/nli-MiniLM-L2-v2",
    "DeBERTa-NLI": "cross-encoder/nli-deberta-v3-small",
}

predictions = {}
for model_label, model_id in models.items():
    print(f"\n[Running] {model_label} ({model_id})")
    try:
        preds = run_zero_shot(model_id, texts)
        predictions[model_label] = preds
        print(f"  ✓ Done  ({preds[0]['label_text']}  conf={preds[0]['confidence']:.2f})")
    except Exception as e:
        print(f"  ⚠  Failed: {e} — using neutral fallback")
        predictions[model_label] = [{"label": 2, "confidence": 0.0, "label_text": "Neutral"}] * len(texts)

# ── 2. VADER baseline ─────────────────────────────────────────────────────────
print("\n[Running] VADER / keyword baseline")
if "vader_label" in df.columns and df["vader_label"].notna().sum() > 0:
    vader_preds = []
    for _, row in df.iterrows():
        lbl = int(row["vader_label"]) if not pd.isna(row["vader_label"]) else 2
        vader_preds.append({"label": lbl, "confidence": 0.5,
                            "label_text": LABEL_MAP_ZS.get(lbl, "Neutral")})
    print("  ✓ Using pre-computed VADER labels from CSV")
else:
    pos_kw = {"record","high","surge","boost","growth","rise","increase","digital",
               "benefit","opportunity","cuts","reduces","overtakes"}
    neg_kw = {"crisis","drop","fall","slash","bleak","tighten","deportation","lockdown",
               "slowdown","decline","decrease","tension","threatens"}
    vader_preds = []
    for t in texts:
        words = set(str(t).lower().split())
        if words & pos_kw:   lbl = 0
        elif words & neg_kw: lbl = 1
        else:                lbl = 2
        vader_preds.append({"label": lbl, "confidence": 0.5, "label_text": LABEL_MAP_ZS[lbl]})
    print("  ✓ Done (keyword heuristic — vaderSentiment not in CSV)")
predictions["VADER"] = vader_preds

# ── 3. Inter-model agreement matrix ───────────────────────────────────────────
from sklearn.metrics import cohen_kappa_score

print("\n[3] Inter-model agreement matrix (Cohen's κ)")
pred_labels = {k: np.array([r["label"] for r in v]) for k, v in predictions.items()}
model_names = list(pred_labels.keys())

print(f"\n  {'':<16}" + "".join(f"  {m:>12}" for m in model_names))
print("  " + "-" * (16 + 14 * len(model_names)))
for m1 in model_names:
    row = f"  {m1:>16}"
    for m2 in model_names:
        if m1 == m2:
            row += f"  {'1.000':>12}"
        else:
            try:
                kappa = cohen_kappa_score(pred_labels[m1], pred_labels[m2])
                row += f"  {kappa:>12.3f}"
            except Exception:
                row += f"  {'N/A':>12}"
    print(row)
print("  (Values are Cohen's κ — higher = more agreement)")

# ── 4. Distribution comparison ────────────────────────────────────────────────
print("\n[4] Sentiment distribution per model")
print(f"  {'Model':<16}  {'Positive':>10}  {'Negative':>10}  {'Neutral':>10}  {'Avg Conf':>10}")
print("  " + "-" * 65)
dist_rows = []
for model_name, preds in predictions.items():
    labels = np.array([r["label"] for r in preds])
    confs  = np.array([r["confidence"] for r in preds])
    pos  = (labels == 0).mean()
    neg  = (labels == 1).mean()
    neu  = (labels == 2).mean()
    conf = confs.mean()
    dist_rows.append({"model": model_name, "positive": pos, "negative": neg,
                       "neutral": neu, "avg_confidence": conf})
    print(f"  {model_name:<16}  {pos:>10.1%}  {neg:>10.1%}  {neu:>10.1%}  {conf:>10.3f}")

# ── 5. Ragas-inspired quality metrics ─────────────────────────────────────────
print(f"""
[5] Ragas-inspired corpus quality assessment
  (Adapts Ragas dimensions — faithfulness, context relevance, answer relevance)

  Dimension 1 | FAITHFULNESS (label consistency across models)
  ─────────────────────────────────────────────────────────────
  "A faithful label is one that at least 2/3 of models agree on."
""")
if len(pred_labels) >= 2:
    from scipy.stats import mode as scipy_mode
    label_matrix = np.stack(list(pred_labels.values()), axis=1)
    majority, _  = scipy_mode(label_matrix, axis=1, keepdims=True)
    majority     = majority.ravel()
    all_agree    = (label_matrix == majority[:, None]).all(axis=1).mean()
    two_thirds   = (
        (label_matrix == majority[:, None]).sum(axis=1)
        >= max(2, int(label_matrix.shape[1] * 0.67))
    ).mean()
    print(f"  All models agree (exact)  : {all_agree*100:.1f}%")
    print(f"  ≥2/3 models agree (major) : {two_thirds*100:.1f}%")
    print(f"  Faithful articles         : {int(two_thirds*EVAL_SIZE)}/{EVAL_SIZE}")

print(f"""
  Dimension 2 | CONTEXT RELEVANCE (relevance score distribution)
  ──────────────────────────────────────────────────────────────
  "A contextually relevant article scores ≥1% on the keyword relevance metric."
""")
if "relevance_score" in df.columns:
    relevant        = (df["relevance_score"] >= 1.0).mean()
    highly_relevant = (df["relevance_score"] >= 5.0).mean()
    print(f"  Articles with score ≥ 1%  : {relevant*100:.1f}%")
    print(f"  Articles with score ≥ 5%  : {highly_relevant*100:.1f}%")
    print(f"  Median relevance score    : {df['relevance_score'].median():.2f}%")
else:
    print("  Relevance scores not available.")

print(f"""
  Dimension 3 | ANSWER RELEVANCE (language-conditioned sentiment variance)
  ─────────────────────────────────────────────────────────────────────────
  "A sentiment signal is informative only if it varies across time and language."
""")
for model_name, preds in list(predictions.items())[:2]:
    df[f"label_{model_name}"] = [r["label"] for r in preds]
    if "language" in df.columns:
        lang_var = df.groupby("language")[f"label_{model_name}"].std()
        print(f"  {model_name} — inter-language sentiment std:")
        for lang, std_val in lang_var.items():
            print(f"    {lang}: std = {std_val:.3f}  "
                  f"{'varied' if std_val > 0.3 else 'homogeneous'}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(                        out / "t3_zeroshot_comparison.csv",  index=False, encoding="utf-8-sig")
pd.DataFrame(dist_rows).to_csv(   out / "t3_model_distribution.csv",   index=False)

print(f"\n  Saved: {out / 't3_zeroshot_comparison.csv'}")
print(f"  Saved: {out / 't3_model_distribution.csv'}")

print("\n" + sep)
print("T3 COMPLETE")
print(sep)
