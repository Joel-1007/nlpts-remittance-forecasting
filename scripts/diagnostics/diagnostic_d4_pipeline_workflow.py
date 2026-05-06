"""
DIAGNOSTIC D4 — Pipeline Workflow Evidence
==========================================
Reviewer concern: "Black box" pipeline — VADER→mBERT labeling lacks
accuracy evidence; pipeline appears redundant.

HOW TO USE ON KAGGLE:
  • Requires ablation_results.json and/or sentiment_vectors.csv in
    /kaggle/working/ or /kaggle/input/. At least one must be present.

Output: printed report
"""

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

ablation_path  = find("ablation_results.json")
sentiment_path = find("sentiment_vectors.csv")

if ablation_path is None and sentiment_path is None:
    raise FileNotFoundError(
        "D4 requires ablation_results.json and/or sentiment_vectors.csv "
        "in /kaggle/working/ or /kaggle/input/.\n"
        "Upload your pipeline output files and rerun."
    )

print(sep)
print("D4 — PIPELINE WORKFLOW & LABELING TRANSPARENCY")
print(f"     Ablation : {'REAL DATA' if ablation_path else 'MISSING — skipping'}")
print(f"     Sentiment: {'REAL DATA' if sentiment_path else 'MISSING — skipping'}")
print(sep)

abl = json.load(open(ablation_path, encoding="utf-8")) if ablation_path else None
sv  = pd.read_csv(sentiment_path)                       if sentiment_path else None

LANG_NAMES = {"en": "English", "hi": "Hindi",  "ta": "Tamil",  "te": "Telugu",
              "ml": "Malayalam","bn": "Bengali","pa": "Punjabi","gu": "Gujarati"}

# ── 1. Labeling chain ────────────────────────────────────────────────────────
print("""
[1] Automated Labeling Chain (VADER → mBERT pseudo-label fine-tuning)

  Stage 1: VADER Pseudo-Labeling
    compound > 0.05  → Positive  |  compound < -0.05 → Negative  |  else → Neutral
    Applied to TRAIN SET ONLY. Limitation: VADER is English-only.

  Stage 2: mBERT Fine-tuning
    bert-base-multilingual-cased fine-tuned on pseudo-labeled train set.
    TRUE METRIC: VADER→mBERT changed predictions % (ablation below).

  Stage 3: Sentiment Score = pos_prob − neg_prob (continuous, not argmax).
""")

# ── 2. Ablation results ──────────────────────────────────────────────────────
if abl is not None:
    print("[2] Ablation results")

    print("\n  Ablation A — VADER Baseline vs fine-tuned mBERT")
    vm        = abl.get("vader_vs_mbert", {})
    agreement = vm.get("agreement")
    changed   = vm.get("changed_predictions_pct")
    vd        = vm.get("vader_distribution", {})
    md        = vm.get("mbert_distribution", {})
    if agreement is not None:
        print(f"    VADER→mBERT agreement       : {agreement*100:.1f}%")
        print(f"    Predictions changed by mBERT: {changed:.1f}%")
        print(f"    VADER  dist: Pos={vd.get('positive',0):.1%}  Neg={vd.get('negative',0):.1%}  Neu={vd.get('neutral',0):.1%}")
        print(f"    mBERT  dist: Pos={md.get('positive',0):.1%}  Neg={md.get('negative',0):.1%}  Neu={md.get('neutral',0):.1%}")
        p_e   = sum(vd.get(k,0) * md.get(k,0) for k in ["positive","negative","neutral"])
        kappa = (agreement - p_e) / (1 - p_e + 1e-9)
        interp = ("Almost perfect" if kappa >= 0.80 else "Substantial" if kappa >= 0.60
                  else "Moderate" if kappa >= 0.40 else "Fair" if kappa >= 0.20
                  else "Slight" if kappa >= 0.00 else "Poor")
        print(f"\n    Cohen's Kappa (proxy): κ ≈ {kappa:.3f}  [{interp}]")

    print("\n  Ablation B — English-only vs Multilingual mBERT")
    em = abl.get("english_vs_multilingual", {})
    if em.get("skipped"):
        print(f"    Skipped: {em.get('reason')}")
    else:
        ed  = em.get("english_distribution", {})
        md2 = em.get("multilingual_distribution", {})
        print(f"    Agreement: {em.get('agreement',0)*100:.1f}%")
        print(f"    English-only dist  : Pos={ed.get('positive',0):.1%}  Neg={ed.get('negative',0):.1%}  Neu={ed.get('neutral',0):.1%}")
        print(f"    Multilingual dist  : Pos={md2.get('positive',0):.1%}  Neg={md2.get('negative',0):.1%}  Neu={md2.get('neutral',0):.1%}")

    print("\n  Ablation C — Per-language confidence (mBERT on test set)")
    lang_perf = abl.get("per_language", {})
    print(f"  {'Language':<14} {'n':>6}  {'Conf':>8}  {'Pos':>7}  {'Neg':>7}  {'Neu':>7}")
    print("  " + "-" * 55)
    for iso, stats in sorted(lang_perf.items(), key=lambda x: x[1]["n"], reverse=True):
        dist = stats.get("distribution", {})
        print(f"  {LANG_NAMES.get(iso,iso):<14} {stats['n']:>6}  "
              f"{stats['avg_confidence']:>8.3f}  "
              f"{dist.get('positive',0):>7.1%}  "
              f"{dist.get('negative',0):>7.1%}  "
              f"{dist.get('neutral',0):>7.1%}")

    print("\n  Ablation D — Crisis detection (keyword vs zero-shot)")
    cd = abl.get("crisis_detection", {})
    if cd:
        print(f"    Keyword ↔ zero-shot agreement: {cd.get('agreement',0)*100:.1f}%")
        print(f"    Precision (zero-shot): {cd.get('precision',0):.3f}")
        print(f"    Recall  (zero-shot) : {cd.get('recall',0):.3f}")
else:
    print("[2] Ablation results — SKIPPED (ablation_results.json not found)")

# ── 3. Sentiment vector variance ─────────────────────────────────────────────
if sv is not None:
    print("\n[3] Sentiment vector quality")
    train_sv = sv[sv["data_split"] == "train"] if "data_split" in sv.columns else sv
    test_sv  = sv[sv["data_split"] == "test"]  if "data_split" in sv.columns else sv
    print(f"  Train quarters: {len(train_sv)}  |  Test quarters: {len(test_sv)}")
    for split, sub in [("Train", train_sv), ("Test", test_sv)]:
        if len(sub) and "sentiment_mean" in sub.columns:
            sm = sub["sentiment_mean"]
            print(f"\n  {split}: mean={sm.mean():.4f}  std={sm.std():.4f}  "
                  f"min={sm.min():.4f}  max={sm.max():.4f}  range={sm.max()-sm.min():.4f}")
            print(f"    {'⚠  Very low variance — sentiment may not add signal.' if sm.std() < 0.02 else '✓ Meaningful variance — sentiment provides signal.'}")
else:
    print("\n[3] Sentiment vector quality — SKIPPED (sentiment_vectors.csv not found)")

# ── 4. GARCH gate justification ──────────────────────────────────────────────
print("""
[4] Structural justification for GARCH gate

  h_t  = GARCH(1,1) variance of EPU_Index
  gate = σ(α·h_t + β)
  ŷ_t  = (1 − gate)·SARIMA(t) + gate·f(Sentiment_t, EPU_t)

  High EPU volatility → gate↑ (sentiment amplified, migrants respond to crisis).
  Low  EPU volatility → gate↓ (SARIMA dominates, sentiment adds noise).

  Mirrors Engle & Rangel (2008) Spline-GARCH; consistent with Baker, Bloom,
  Davis (2016) EPU literature.
""")

print(sep)
print("D4 COMPLETE")
print(sep)
