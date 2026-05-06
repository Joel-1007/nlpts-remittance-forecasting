"""
NLPTS5_v6.py
================================================================================
CUMULATIVE FIXES (all previous + new in v6):

  v2: JSON serialization — numpy types → Python native types
  v3: classification_report crash — labels=[0,1,2] always passed
      XLM label map hardened — covers LABEL_N and human-readable formats
      XLM label map validated at load time with dummy inference
      VADER ablation scoped to English-only (fair comparison)
  v4: Trainer tokenizer= → processing_class= (transformers >=4.46)
      logging_dir= suppressed for transformers >=4.50
  v5: WeightedTrainer with inverse-frequency class weights (FIX A)
      Correlation: handle annual-only inward_flows data (FIX B)
      Adaptive hyperparameter profile for small datasets (FIX C)

  v6 [NEW — targeted fixes against actual NLPTS4 output]:

  FIX D — relevance_tier integration
    NLPTS4 now outputs a relevance_tier column (High/Medium/Low).
    NLPTS5 was ignoring it entirely, meaning mBERT trained on Low-relevance
    GKG articles (synthetic "domain | themes" titles) alongside High-relevance
    real articles — introducing label noise.
    Fix: pseudo-labeling and mBERT training use High+Medium articles only.
    Low-tier articles are inference-only (predictions generated but excluded
    from training loss). A config flag 'min_train_tier' controls this.

  FIX E — crisis_flag reuse
    NLPTS4 already computed a per-article crisis_flag (0/1) using the same
    keyword list. NLPTS5 was recomputing it from scratch via detect_crisis_keywords
    on every article, wasting time and producing slightly different results.
    Fix: use the existing crisis_flag column from the CSV directly.
    ZeroShotCrisisClassifier is still run for the continuous crisis scores
    (economic/political/disaster proportions) but is not used for the binary flag.

  FIX F — ai4bharat/indic-bert crash
    ai4bharat/indic-bert is a base MLM model with NO sentiment classification
    head. Calling pipeline('sentiment-analysis') on it raises:
      ValueError: The model ... is not supported for text-classification
    The MuRIL/IndicBERT section was misleadingly named and would crash on
    every Kaggle run. Fix: remove the IndicBERT attempt entirely.
    ALL non-English languages (including Hindi) use XLM-RoBERTa
    (cardiffnlp/twitter-xlm-roberta-base-sentiment, Barbieri et al. 2022).
    This is well-validated and covers all 7 Indian languages in the corpus.

  FIX G — broken A8 event annotation check
    The stability file existence check used a broken list comprehension that
    always evaluated False, meaning the join was silently skipped every run.
    Fix: remove the broken check; save event_annotation_table.csv directly
    and perform the join in a separate post-stability step.

  FIX D2 — LANG_F1_WEIGHTS timing
    The language-weighted aggregation (A7) used hardcoded F1 weights that
    are UNKNOWN before ablation runs. Using them before training produces
    circular logic (weights depend on results not yet computed).
    Fix: weighted aggregation now runs AFTER ablation. During training,
    equal weights (1.0) are used. Post-ablation, actual F1 scores are
    extracted and saved to language_f1_weights.csv, then weighted aggregation
    is recomputed and appended to sentiment_vectors.csv.

OUTPUT FILES (paths unchanged — compatible with next-phase scripts):
  /kaggle/working/sentiment_vectors.csv
  /kaggle/working/crisis_index_train.csv
  /kaggle/working/ablation_results.json
  /kaggle/working/sentiment_stability_analysis.csv
  /kaggle/working/sentiment_correlation_analysis.json
  /kaggle/working/language_f1_weights.csv
  /kaggle/working/event_annotation_table.csv
================================================================================
"""

import pandas as pd
import numpy as np
import json
import gc
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, pipeline, DataCollatorWithPadding
)
import transformers as _transformers_module
from datasets import Dataset
from sklearn.metrics import (
    f1_score, classification_report, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

# ── transformers version — used throughout for compatibility shims ───────────
_TV = tuple(int(x) for x in _transformers_module.__version__.split('.')[:2])

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Models
    'mbert_model':      'bert-base-multilingual-cased',
    'zero_shot_model':  'facebook/bart-large-mnli',

    # Tokenisation
    'max_length': 128,

    # Training — base values; may be overridden by adaptive profile (FIX C)
    'batch_size':                  8,
    'eval_batch_size':             16,
    'learning_rate':               2e-5,
    'num_epochs':                  3,
    'warmup_steps':                200,
    'weight_decay':                0.01,
    'gradient_accumulation_steps': 2,
    'label_smoothing':             0.0,

    # Misc
    'seed':                       42,
    'crisis_weight_economic':     0.7,
    'crisis_weight_political':    0.3,
    'crisis_threshold':           0.5,
    'sentiment_spike_threshold':  0.5,
    'test_size':                  0.2,
    'train_cutoff_year':          2022,
    'prediction_batch_size':      32,
    'cv_n_splits':                5,
    'stability_window':           8,
    'crisis_batch_size':          16,

    # FIX C — below this, use the "small dataset" training profile
    'small_dataset_threshold':    12000,

    # FIX D — minimum relevance_tier for training ('High', 'Medium', or 'Low')
    # 'High'   → train only on clearly remittance-focused articles (~93K)
    # 'Medium' → train on High + Medium (all 138K) — recommended
    # 'Low'    → train on everything (not recommended — adds label noise)
    'min_train_tier': 'Medium',
}

CRISIS_KEYWORDS = {
    'economic': ['crisis', 'recession', 'downturn', 'crash', 'unemployment',
                 'remittance tax', 'rupee depreciation', 'inflation',
                 'संकट', 'மंदி', 'நெருக்கடி', 'సంక్షోభం', 'പ്രതിസന്ധി', 'সংকট'],
    'political': ['war', 'conflict', 'sanctions', 'visa restriction',
                  'युद्ध', 'போர்', 'వార్', 'യുദ്ധം', 'যুদ্ধ'],
    'disaster':  ['disaster', 'pandemic', 'COVID', 'earthquake', 'flood',
                  'आपदा', 'பேரிடர்', 'విపత్తు', 'ദുരന്തം', 'দুর্যোগ'],
}

# Tier order — used for filtering
_TIER_ORDER = {'High': 0, 'Medium': 1, 'Low': 2}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def is_trainable_tier(tier: str, min_tier: str) -> bool:
    """Return True if tier is at least as good as min_tier."""
    return _TIER_ORDER.get(tier, 99) <= _TIER_ORDER.get(min_tier, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT LABELERS
# ═══════════════════════════════════════════════════════════════════════════════

def pseudo_label_sentiment(text: str, vader_analyzer) -> int:
    """VADER-based labeler for English articles. 0=Pos, 1=Neg, 2=Neu."""
    if not isinstance(text, str) or len(text) < 10:
        return 2
    score = vader_analyzer.polarity_scores(text)['compound']
    return 0 if score > 0.05 else (1 if score < -0.05 else 2)


# ── XLM-RoBERTa multilingual labeler ─────────────────────────────────────────
# cardiffnlp/twitter-xlm-roberta-base-sentiment: Barbieri et al. (2022)
# Covers 100+ languages including all 7 Indian languages in this corpus.
#
# FIX F: ai4bharat/indic-bert is a base MLM model — it has NO sentiment head
# and crashes with pipeline('sentiment-analysis'). Removed entirely.
# ALL non-English languages (including Hindi) use XLM-RoBERTa.
#
# Label map covers both LABEL_N (older checkpoint) and human-readable
# (newer checkpoint) formats — robust across model revisions.

_xlm_pipeline = None

_XLM_LABEL_MAP = {
    'LABEL_0': 1, 'negative': 1, 'NEGATIVE': 1,   # → Negative (1)
    'LABEL_1': 2, 'neutral':  2, 'NEUTRAL':  2,   # → Neutral  (2)
    'LABEL_2': 0, 'positive': 0, 'POSITIVE': 0,   # → Positive (0)
}


def get_xlm_pipeline():
    global _xlm_pipeline
    if _xlm_pipeline is None:
        print("   Loading XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment)...")
        _xlm_pipeline = pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-xlm-roberta-base-sentiment',
            tokenizer='cardiffnlp/twitter-xlm-roberta-base-sentiment',
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=128,
        )
        # Validate label format at load time
        test_out  = _xlm_pipeline(["test"])
        detected  = test_out[0]['label']
        if detected not in _XLM_LABEL_MAP:
            print(f"   ⚠️  WARNING: Unknown XLM label '{detected}' — defaulting to Neutral.")
        else:
            print(f"   ✅ XLM-RoBERTa loaded. Label format validated: '{detected}'")
    return _xlm_pipeline


def xlm_label_sentiment(texts: List[str], batch_size: int = 32) -> List[int]:
    """XLM-RoBERTa labeler for ALL non-English articles. 0=Pos, 1=Neg, 2=Neu."""
    if not texts:
        return []
    xlm = get_xlm_pipeline()
    results = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="   XLM-RoBERTa labeling", leave=False):
        batch = [str(t)[:512] if isinstance(t, str) else '' for t in texts[i:i+batch_size]]
        try:
            outputs = xlm(batch)
            results.extend([_XLM_LABEL_MAP.get(o['label'], 2) for o in outputs])
        except Exception as e:
            print(f"   ⚠️  XLM batch error at index {i}: {e}. Defaulting to Neutral.")
            results.extend([2] * len(batch))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class RemittanceNLPDataset:
    def __init__(self, csv_path: str = None):
        print("\n" + "="*80)
        print("LOADING DATA (STRICT TEMPORAL SPLIT)")
        print("="*80)

        if csv_path is None:
            for path in [
                '/kaggle/working/remittances_news_final.csv',
                '/kaggle/input/remittance-news/remittances_news_final.csv',
                'remittances_news_final.csv',
            ]:
                try:
                    pd.read_csv(path, nrows=1)
                    csv_path = path
                    break
                except Exception:
                    continue
            if csv_path is None:
                raise FileNotFoundError(
                    "remittances_news_final.csv not found. Run NLPTS4 first.")

        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df):,} articles")

        # ── Ensure required columns exist ─────────────────────────────────────
        if 'language' not in self.df.columns:
            self.df['language'] = 'en'
        if 'relevance_tier' not in self.df.columns:
            # Backward compat: if NLPTS4 older version, assign tier from score
            print("   ⚠️  relevance_tier column missing — deriving from relevance_score")
            self.df['relevance_tier'] = self.df['relevance_score'].apply(
                lambda s: 'High' if s >= 20 else ('Medium' if s >= 5 else 'Low'))
        if 'crisis_flag' not in self.df.columns:
            self.df['crisis_flag'] = 0

        self.df['seendate'] = pd.to_datetime(self.df['seendate'], format='mixed')
        self.df['year']         = self.df['seendate'].dt.year
        self.df['quarter']      = self.df['seendate'].dt.quarter
        self.df['year_quarter'] = (self.df['year'].astype(str) + 'Q' +
                                   self.df['quarter'].astype(str))

        # ── FIX D: tier-based training/inference split ─────────────────────────
        min_tier     = CONFIG['min_train_tier']
        tier_mask    = self.df['relevance_tier'].apply(
            lambda t: is_trainable_tier(t, min_tier))
        self.df_train_eligible = self.df[tier_mask].copy()
        self.df_infer_only     = self.df[~tier_mask].copy()

        tier_counts = self.df['relevance_tier'].value_counts().to_dict()
        print(f"\n   Relevance tier breakdown: {tier_counts}")
        print(f"   Training-eligible (tier ≥ {min_tier}): "
              f"{len(self.df_train_eligible):,} articles")
        print(f"   Inference-only (tier < {min_tier}):    "
              f"{len(self.df_infer_only):,} articles")

        # ── Temporal split on training-eligible articles ─────────────────────
        print(f"\n⚠️  TEMPORAL SPLIT (cutoff: {CONFIG['train_cutoff_year']})")
        self.train_df = self.df_train_eligible[
            self.df_train_eligible['year'] <= CONFIG['train_cutoff_year']].copy()
        self.test_df  = self.df_train_eligible[
            self.df_train_eligible['year'] >  CONFIG['train_cutoff_year']].copy()
        print(f"   Train: {len(self.train_df):,} (≤{CONFIG['train_cutoff_year']}, "
              f"tier ≥ {min_tier})")
        print(f"   Test:  {len(self.test_df):,}  (>{CONFIG['train_cutoff_year']}, "
              f"tier ≥ {min_tier})")

        if len(self.test_df) == 0:
            self.train_df, self.test_df = train_test_split(
                self.df_train_eligible,
                test_size=CONFIG['test_size'],
                random_state=CONFIG['seed'])

        # ── FIX E: use existing crisis_flag from CSV ──────────────────────────
        # NLPTS4 already computed binary crisis_flag per article.
        # No need to recompute from keywords — avoids duplication and drift.
        print("\n   Using crisis_flag from NLPTS4 CSV (FIX E — no recomputation)")
        self.train_df['crisis_mention'] = self.train_df['crisis_flag'].fillna(0).astype(int)
        print(f"   Crisis articles in train set: "
              f"{self.train_df['crisis_mention'].sum():,} "
              f"({self.train_df['crisis_mention'].mean()*100:.1f}%)")

        # ── Language-aware pseudo-labeling ────────────────────────────────────
        print("\n   Generating language-aware pseudo-labels...")
        print("   Strategy:")
        print("     English     → VADER (Hutto & Gilbert, 2014)")
        print("     Non-English → XLM-RoBERTa (Barbieri et al., 2022)")
        print("   Note [FIX F]: ai4bharat/indic-bert removed — base MLM, no")
        print("   sentiment head. XLM-RoBERTa covers all 7 Indian languages.")
        vader = SentimentIntensityAnalyzer()

        def label_by_language(df, split_name=''):
            labels = pd.Series(2, index=df.index, dtype=int)

            # English → VADER
            en_mask = df['language'] == 'en'
            if en_mask.sum() > 0:
                labels[en_mask] = df.loc[en_mask, 'title'].apply(
                    lambda x: pseudo_label_sentiment(x, vader))
                n_pos = (labels[en_mask] == 0).sum()
                n_neg = (labels[en_mask] == 1).sum()
                n_neu = (labels[en_mask] == 2).sum()
                print(f"   VADER [{split_name} {en_mask.sum():,} English]: "
                      f"Pos={n_pos:,}  Neg={n_neg:,}  Neu={n_neu:,}")

            # All non-English → XLM-RoBERTa (FIX F: no IndicBERT branch)
            ml_mask = ~en_mask
            if ml_mask.sum() > 0:
                ml_texts  = df.loc[ml_mask, 'title'].tolist()
                ml_labels = xlm_label_sentiment(ml_texts, batch_size=32)
                labels[ml_mask] = ml_labels
                temp = df[ml_mask].copy()
                temp['_lbl'] = ml_labels
                for lang in sorted(temp['language'].unique()):
                    d = temp[temp['language'] == lang]['_lbl'].value_counts().to_dict()
                    print(f"   XLM [{split_name} {lang}]: "
                          f"Pos={d.get(0,0):,}  Neg={d.get(1,0):,}  Neu={d.get(2,0):,}")

            return labels.tolist()

        self.train_df['sentiment_label'] = label_by_language(self.train_df, 'train')
        self.test_df['sentiment_label']  = label_by_language(self.test_df,  'test')

        if 'label_method' not in self.train_df.columns:
            self.train_df['label_method'] = self.train_df['language'].apply(
                lambda l: 'vader' if l == 'en' else 'xlm_roberta')
            self.test_df['label_method'] = self.test_df['language'].apply(
                lambda l: 'vader' if l == 'en' else 'xlm_roberta')

        print("\n✅ TRAIN/TEST SEPARATED WITH LANGUAGE-AWARE LABELS")

    def get_datasets(self):
        return (
            Dataset.from_dict({'text':  self.train_df['title'].tolist(),
                               'label': self.train_df['sentiment_label'].tolist()}),
            Dataset.from_dict({'text':  self.test_df['title'].tolist(),
                               'label': self.test_df['sentiment_label'].tolist()}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FIX A — WEIGHTED TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class WeightedTrainer(Trainer):
    """Trainer with dynamic inverse-frequency class-weighted CrossEntropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self._class_weights.to(logits.device)
        loss    = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_dataset) -> torch.Tensor:
    """Inverse-frequency weights for [Positive=0, Negative=1, Neutral=2]."""
    labels    = np.array(train_dataset['label'])
    n_samples = len(labels)
    n_classes = 3
    weights   = []
    for c in range(n_classes):
        n_c = (labels == c).sum()
        w   = n_samples / (n_classes * n_c) if n_c > 0 else 1.0
        weights.append(w)
    wt = torch.tensor(weights, dtype=torch.float32)
    print(f"   Class weights — Pos={wt[0]:.2f}  Neg={wt[1]:.2f}  Neu={wt[2]:.2f}")
    return wt


# ═══════════════════════════════════════════════════════════════════════════════
# mBERT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class mBERTSentimentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['mbert_model'])
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['mbert_model'], num_labels=3, ignore_mismatched_sizes=True)
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], truncation=True,
                              max_length=CONFIG['max_length'], padding=False)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'f1':       f1_score(labels, predictions, average='weighted', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro',    zero_division=0),
        }

    def train(self, train_dataset, eval_dataset,
              output_dir: str = '/kaggle/working/mbert_sentiment'):

        train_tok = train_dataset.map(self.tokenize_function, batched=True)
        eval_tok  = eval_dataset.map(self.tokenize_function,  batched=True)

        # ── FIX C: adaptive hyperparameter profile for small datasets ──────
        n_train = len(train_dataset)
        if n_train < CONFIG['small_dataset_threshold']:
            epochs     = 5
            lr         = 3e-5
            warmup     = 300
            label_sm   = 0.1
            grad_accum = 4
            print(f"   ⚡ Small dataset ({n_train:,} < {CONFIG['small_dataset_threshold']:,}): "
                  f"epochs={epochs}, lr={lr}, label_smoothing={label_sm}")
        else:
            epochs     = CONFIG['num_epochs']
            lr         = CONFIG['learning_rate']
            warmup     = CONFIG['warmup_steps']
            label_sm   = CONFIG['label_smoothing']
            grad_accum = CONFIG['gradient_accumulation_steps']
            print(f"   Standard profile: epochs={epochs}, lr={lr}, n_train={n_train:,}")

        # ── TrainingArguments — version-safe ────────────────────────────────
        ta_kwargs = dict(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=lr,
            per_device_train_batch_size=CONFIG['batch_size'],
            per_device_eval_batch_size=CONFIG['eval_batch_size'],
            num_train_epochs=epochs,
            weight_decay=CONFIG['weight_decay'],
            warmup_steps=warmup,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_steps=50,
            seed=CONFIG['seed'],
            gradient_accumulation_steps=grad_accum,
            label_smoothing_factor=label_sm,
            use_cpu=not torch.cuda.is_available(),
            report_to='none',
        )
        if _TV < (4, 50):
            ta_kwargs['logging_dir'] = f'{output_dir}/logs'

        training_args = TrainingArguments(**ta_kwargs)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # ── FIX A: class weights ────────────────────────────────────────────
        print("   Computing inverse-frequency class weights (FIX A)...")
        class_weights = compute_class_weights(train_dataset)

        trainer_kwargs = dict(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        if _TV >= (4, 46):
            trainer_kwargs['processing_class'] = self.tokenizer
        else:
            trainer_kwargs['tokenizer'] = self.tokenizer

        trainer = WeightedTrainer(**trainer_kwargs)
        print("   Training mBERT...")
        trainer.train()
        eval_result = trainer.evaluate()
        print(f"   ✅ F1 (weighted): {eval_result['eval_f1']:.4f} "
              f"| F1 (macro): {eval_result['eval_f1_macro']:.4f}")
        return eval_result

    def predict(self, texts: List[str], show_progress: bool = True):
        if not texts:
            return np.array([]), np.array([]), np.array([])

        dataset   = Dataset.from_dict({'text': texts})
        tokenized = dataset.map(self.tokenize_function, batched=True)

        self.model.eval()
        all_preds, all_scores = [], []
        batches = range(0, len(tokenized), CONFIG['prediction_batch_size'])
        if show_progress:
            batches = tqdm(batches, desc="   Predicting sentiment", unit="batch")

        for i in batches:
            batch = tokenized[i:i + CONFIG['prediction_batch_size']]
            enc   = self.tokenizer(batch['text'], truncation=True,
                                   max_length=CONFIG['max_length'],
                                   padding=True, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                probs  = torch.softmax(self.model(**enc).logits, dim=1)
                preds  = torch.argmax(probs, dim=1)
                scores = probs[range(len(probs)), preds]
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        preds           = np.array(all_preds)
        scores          = np.array(all_scores)
        sentiment_scores = np.where(preds == 0, scores, -scores)
        return preds, scores, sentiment_scores


# ═══════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT CRISIS CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroShotCrisisClassifier:
    def __init__(self):
        self.classifier = pipeline(
            'zero-shot-classification',
            model=CONFIG['zero_shot_model'],
            device=0 if torch.cuda.is_available() else -1)
        self.labels = ['economic crisis', 'political crisis',
                       'natural disaster', 'neutral']

    def classify_crisis(self, texts: List[str], batch_size: int = None):
        if batch_size is None:
            batch_size = CONFIG['crisis_batch_size']
        results = []
        print(f"   Classifying {len(texts):,} articles in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="   Crisis classification", unit="batch"):
            batch         = texts[i:i+batch_size]
            batch_results = self.classifier(batch, self.labels, multi_label=True)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            for res in batch_results:
                d = dict(zip(res['labels'], res['scores']))
                results.append({
                    'economic':  d.get('economic crisis', 0),
                    'political': d.get('political crisis', 0),
                    'disaster':  d.get('natural disaster', 0),
                })
        return results

    def aggregate_quarterly_train_only(self, train_df):
        """
        FIX E: binary crisis_flag from NLPTS4 CSV is used for crisis_proportion.
        ZeroShot classifier still runs for continuous economic/political/disaster
        scores which are NOT in the NLPTS4 CSV.
        """
        print("   Running zero-shot crisis scoring (continuous scores)...")
        crisis_scores = self.classify_crisis(train_df['title'].tolist())

        train_df['crisis_economic']  = [c['economic']  for c in crisis_scores]
        train_df['crisis_political'] = [c['political'] for c in crisis_scores]
        train_df['crisis_disaster']  = [c['disaster']  for c in crisis_scores]
        train_df['crisis_index']     = (
            CONFIG['crisis_weight_economic']  * train_df['crisis_economic'] +
            CONFIG['crisis_weight_political'] * train_df['crisis_political']
        )

        quarterly = train_df.groupby('year_quarter').agg({
            'crisis_economic':  'mean',
            'crisis_political': 'mean',
            'crisis_disaster':  'mean',
            'crisis_index':     'mean',
            # FIX E: use NLPTS4's pre-computed crisis_flag as crisis_proportion
            'crisis_mention':   'mean',
        }).reset_index()
        quarterly.columns = ['quarter', 'crisis_economic', 'crisis_political',
                              'crisis_disaster', 'crisis_index', 'crisis_proportion']
        print(f"   ✅ {len(quarterly)} quarters with crisis scores")
        return quarterly


# ═══════════════════════════════════════════════════════════════════════════════
# FIX B — CORRELATION: HANDLE ANNUAL-ONLY REMITTANCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_remittance_data_quarterly():
    print("\n" + "="*80)
    print("LOADING ACTUAL REMITTANCE FLOW DATA")
    print("="*80)

    df_inward = None
    for path in [
        '/kaggle/working/inward_flows.csv',
        '/kaggle/input/m-sense/inward_flows.csv',
        '/kaggle/input/remittance-data/inward_flows.csv',
        'inward_flows.csv',
    ]:
        try:
            df_inward = pd.read_csv(path)
            print(f"✅ Found: {path}  shape={df_inward.shape}")
            print(f"   Columns: {list(df_inward.columns)}")
            break
        except Exception:
            continue

    if df_inward is None:
        print("❌ inward_flows.csv NOT FOUND — correlation analysis will be skipped.")
        return None, False

    if 'quarter' not in df_inward.columns or 'inward_flow' not in df_inward.columns:
        print(f"❌ Missing required columns. Have: {list(df_inward.columns)}")
        return None, False

    quarterly_remit = (df_inward.groupby('quarter')['inward_flow']
                       .sum().reset_index()
                       .sort_values('quarter').reset_index(drop=True))

    # Detect annual vs quarterly data
    years_in_data = quarterly_remit['quarter'].str[:4]
    qtrs_per_year = quarterly_remit.groupby(years_in_data)['quarter'].count()
    is_annual     = (qtrs_per_year == 1).all()

    if is_annual:
        print(f"   ⚠️  Detected ANNUAL data (one Q1 entry per year).")
        print(f"      Will aggregate sentiment to annual means before correlation.")
    else:
        print(f"   ✅ Detected true quarterly data ({len(quarterly_remit)} quarters).")

    print(f"\n📊 Sample remittance data:")
    print(quarterly_remit.head(5).to_string(index=False))

    return quarterly_remit, is_annual


def analyze_sentiment_remittance_correlation(quarterly_sentiment):
    print("\n" + "="*80)
    print("SENTIMENT-REMITTANCE CORRELATION ANALYSIS")
    print("="*80)

    df_remit, is_annual = load_remittance_data_quarterly()
    if df_remit is None:
        print("\n⚠️  Correlation analysis skipped — no remittance data.")
        return None

    df_remit['quarter'] = df_remit['quarter'].astype(str)
    qs = quarterly_sentiment.copy()
    qs['quarter'] = qs['quarter'].astype(str)

    if is_annual:
        qs['year']   = qs['quarter'].str[:4]
        annual_sent  = qs.groupby('year').agg(
            sentiment_mean=('sentiment_mean', 'mean'),
            positive_proportion=('positive_proportion', 'mean'),
        ).reset_index()
        annual_sent['quarter'] = annual_sent['year'] + 'Q1'
        annual_sent  = annual_sent.drop(columns='year')
        merge_df     = df_remit.merge(annual_sent, on='quarter', how='inner')
        granularity  = "annual (sentiment aggregated from quarterly)"
    else:
        merge_df    = df_remit.merge(
            qs[['quarter', 'sentiment_mean', 'positive_proportion']],
            on='quarter', how='inner')
        granularity = "quarterly"

    if len(merge_df) < 6:
        print(f"⚠️  Too few overlapping periods ({len(merge_df)}) — need at least 6.")
        return None

    print(f"✅ Merged {len(merge_df)} {granularity} periods for correlation")

    correlations = {}
    print(f"\n🔬 Sentiment-remittance correlations at different lags:")
    print("-" * 80)

    for lag in range(0, 5):
        lag_label = "concurrent" if lag == 0 else f"{lag} period{'s' if lag>1 else ''} prior"
        sent_col  = 'sentiment_mean' if lag == 0 else f'sentiment_lag{lag}'
        if lag > 0:
            merge_df[sent_col] = merge_df['sentiment_mean'].shift(lag)

        valid = merge_df[[sent_col, 'inward_flow']].dropna()
        if len(valid) < 6:
            continue

        pr, pp = pearsonr(valid[sent_col], valid['inward_flow'])
        sr, sp = spearmanr(valid[sent_col], valid['inward_flow'])

        correlations[f'lag_{lag}'] = {
            'lag_periods': int(lag), 'lag_label': lag_label,
            'granularity': granularity,
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp),
            'n_observations': int(len(valid)),
            'significant_pearson':  bool(pp < 0.05),
            'significant_spearman': bool(sp < 0.05),
        }

        sig = "✓ SIGNIFICANT" if pp < 0.05 else ""
        print(f"\nLag {lag} ({lag_label}):")
        print(f"  Pearson:  r = {pr:+.3f},  p = {pp:.4f}  {sig}")
        print(f"  Spearman: ρ = {sr:+.3f},  p = {sp:.4f}")
        print(f"  N = {len(valid)} {granularity} periods")

    if not correlations:
        print("\n⚠️  Could not compute any valid correlations.")
        return None

    opt_key = max(correlations, key=lambda k: abs(correlations[k]['pearson_r']))
    opt     = correlations[opt_key]

    results = {
        'correlations':          convert_to_python_types(correlations),
        'optimal_lag':           opt_key,
        'optimal_lag_periods':   int(opt['lag_periods']),
        'granularity':           granularity,
        'optimal_pearson_r':     float(opt['pearson_r']),
        'optimal_p_value':       float(opt['pearson_p']),
        'is_significant':        bool(opt['significant_pearson']),
        'recommendation': (
            f"Use sentiment at lag {opt['lag_periods']} {granularity} "
            f"period(s) as exogenous variable in SARIMA"
        ),
        'n_overlapping_periods': int(len(merge_df)),
        'date_range':            f"{merge_df['quarter'].min()} to {merge_df['quarter'].max()}",
    }

    with open('/kaggle/working/sentiment_correlation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: sentiment_correlation_analysis.json")
    return correlations


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment_stability(train_df, test_df, window: int = 8):
    print("\n" + "="*80)
    print("SENTIMENT STABILITY & REGIME CHANGE ANALYSIS")
    print("="*80)

    combined = pd.concat([train_df, test_df], ignore_index=True).sort_values('seendate')

    if 'sentiment_pred' not in combined.columns:
        print("⚠️  sentiment_pred column not found — skipping stability analysis.")
        return None

    quarterly = (combined.groupby('year_quarter')
                 .agg(positive_rate=('sentiment_pred', lambda x: (x==0).mean()),
                      negative_rate=('sentiment_pred', lambda x: (x==1).mean()),
                      neutral_rate  =('sentiment_pred', lambda x: (x==2).mean()),
                      seendate      =('seendate', 'min'))
                 .reset_index()
                 .sort_values('seendate'))

    print(f"✅ Analyzing {len(quarterly)} quarters")

    regime_changes = []
    for i in range(window, len(quarterly)):
        prev      = quarterly.iloc[i-window:i]
        curr      = quarterly.iloc[i]
        shifts    = {
            'positive_shift': abs(curr['positive_rate'] - prev['positive_rate'].mean()),
            'negative_shift': abs(curr['negative_rate'] - prev['negative_rate'].mean()),
            'neutral_shift':  abs(curr['neutral_rate']  - prev['neutral_rate'].mean()),
        }
        max_shift = max(shifts.values())
        regime_changes.append({
            'quarter':           curr['year_quarter'],
            'date':              str(curr['seendate'].date()),
            'positive_rate':     float(curr['positive_rate']),
            'negative_rate':     float(curr['negative_rate']),
            'neutral_rate':      float(curr['neutral_rate']),
            'baseline_positive': float(prev['positive_rate'].mean()),
            'baseline_negative': float(prev['negative_rate'].mean()),
            'baseline_neutral':  float(prev['neutral_rate'].mean()),
            **{k: float(v) for k, v in shifts.items()},
            'max_shift':         float(max_shift),
            'regime_change':     bool(max_shift > 0.2),
        })

    regime_df = pd.DataFrame(regime_changes)
    n_changes = regime_df['regime_change'].sum()
    print(f"\n📊 Detected {n_changes} major sentiment regime changes")

    if n_changes > 0:
        print("\n🔍 Top regime shifts:")
        print("-" * 80)
        for _, row in (regime_df[regime_df['regime_change']]
                       .sort_values('max_shift', ascending=False).head(5).iterrows()):
            print(f"\n  {row['quarter']} ({row['date']}):")
            print(f"    Positive: {row['baseline_positive']:.1%} → {row['positive_rate']:.1%}")
            print(f"    Negative: {row['baseline_negative']:.1%} → {row['negative_rate']:.1%}")

    regime_df.to_csv('/kaggle/working/sentiment_stability_analysis.csv', index=False)
    print(f"\n✅ Saved: sentiment_stability_analysis.csv")
    return regime_df


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablations(dataset, mbert_multi):
    print("\n" + "="*80)
    print("ABLATION STUDIES")
    print("="*80)

    ablation     = {}
    ALL_LABELS   = [0, 1, 2]
    TARGET_NAMES = ['Positive', 'Negative', 'Neutral']

    # 1️⃣  VADER vs mBERT (English only — fair comparison)
    print("\n1️⃣  VADER vs mBERT (English only)")
    vader   = SentimentIntensityAnalyzer()
    en_mask = dataset.test_df['language'] == 'en'
    en_test = dataset.test_df[en_mask]

    if len(en_test) > 0:
        vader_preds    = en_test['title'].apply(
            lambda x: pseudo_label_sentiment(x, vader)).values
        mbert_preds_en, _, _ = mbert_multi.predict(
            en_test['title'].tolist(), show_progress=False)
        agreement = (vader_preds == mbert_preds_en).mean()
        print(f"   Agreement (English): {agreement*100:.1f}%")
        ablation['vader_vs_mbert'] = {
            'agreement':               float(agreement),
            'changed_predictions_pct': float((1 - agreement) * 100),
            'n_english_articles':      int(len(en_test)),
        }
    else:
        ablation['vader_vs_mbert'] = {'note': 'No English test articles found'}

    # 2️⃣  Per-language validation
    print("\n2️⃣  Per-Language Validation (mBERT vs pseudo-labeler)")
    lang_perf = {}

    for lang in sorted(dataset.df['language'].unique()):
        lang_mask = dataset.test_df['language'] == lang
        if lang_mask.sum() < 10:
            continue

        lang_texts   = dataset.test_df[lang_mask]['title'].tolist()
        true_labels  = dataset.test_df[lang_mask]['sentiment_label'].values
        mbert_preds, _, _ = mbert_multi.predict(lang_texts, show_progress=False)

        f1_w   = f1_score(true_labels, mbert_preds, average='weighted',
                          labels=ALL_LABELS, zero_division=0)
        prec_w = precision_score(true_labels, mbert_preds, average='weighted',
                                 labels=ALL_LABELS, zero_division=0)
        rec_w  = recall_score(true_labels, mbert_preds, average='weighted',
                              labels=ALL_LABELS, zero_division=0)
        agree  = (mbert_preds == true_labels).mean()
        labeler = 'VADER' if lang == 'en' else 'XLM-RoBERTa'

        print(f"\n   [{lang}] Pseudo-labeler: {labeler} | N={lang_mask.sum()}")
        print(f"   F1 (weighted)={f1_w:.3f}  Precision={prec_w:.3f}  Recall={rec_w:.3f}")
        print(f"   Agreement with {labeler}: {agree*100:.1f}%")
        print(classification_report(true_labels, mbert_preds,
                                    labels=ALL_LABELS,
                                    target_names=TARGET_NAMES,
                                    zero_division=0))

        if lang != 'en' and (mbert_preds == 2).mean() > 0.95:
            print(f"   ⚠️  WARNING: {lang} is {(mbert_preds==2).mean():.0%} Neutral "
                  f"— check XLM label map!")

        lang_perf[lang] = {
            'labeler':                labeler,
            'f1_weighted':            float(f1_w),
            'precision_weighted':     float(prec_w),
            'recall_weighted':        float(rec_w),
            'agreement_with_labeler': float(agree),
            'n':                      int(lang_mask.sum()),
        }

    ablation['per_language_validation'] = lang_perf

    with open('/kaggle/working/ablation_results.json', 'w') as f:
        json.dump(convert_to_python_types(ablation), f, indent=2)
    print("\n✅ Ablation results saved: ablation_results.json")
    return ablation, lang_perf


# ═══════════════════════════════════════════════════════════════════════════════
# FIX D2 — LANGUAGE-WEIGHTED AGGREGATION (runs AFTER ablation)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_language_weighted_sentiment(df, lang_perf: Dict) -> pd.DataFrame:
    """
    FIX D2: Compute quarterly sentiment weighted by per-language F1 scores.
    Called AFTER ablation so actual F1 scores are known, not hardcoded.

    lang_perf: dict of {lang_code: {'f1_weighted': float, ...}}
    """
    # Build weight map from ablation results; default 0.700 for unseen languages
    lang_f1 = {lang: v.get('f1_weighted', 0.700)
               for lang, v in lang_perf.items()}
    default_weight = 0.700

    df = df.copy()
    df['lang_weight']    = df['language'].map(lang_f1).fillna(default_weight)
    df['weighted_score'] = df['sentiment_score'] * df['lang_weight']

    def _weighted_mean(group):
        w = group['lang_weight']
        s = group['weighted_score']
        return (s.sum() / w.sum()) if w.sum() > 0 else 0.0

    def _weighted_pos_prop(group):
        pos_w   = group.loc[group['sentiment_pred'] == 0, 'lang_weight'].sum()
        total_w = group['lang_weight'].sum()
        return pos_w / total_w if total_w > 0 else 0.0

    result = (df.groupby('year_quarter')
                .apply(lambda g: pd.Series({
                    'sentiment_mean_weighted':      _weighted_mean(g),
                    'positive_proportion_weighted': _weighted_pos_prop(g),
                    'n_articles':                   len(g),
                    'effective_weight_sum':         g['lang_weight'].sum(),
                }))
                .reset_index()
                .rename(columns={'year_quarter': 'quarter'}))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("NLPTS5 v6 — Relevance-Tier Training + Crisis Reuse + XLM-only + Post-Ablation Weights")
    print("="*80)

    set_seed(CONFIG['seed'])

    # ── 1. Data + pseudo-labeling ────────────────────────────────────────────
    dataset      = RemittanceNLPDataset()
    train_ds, test_ds = dataset.get_datasets()

    # Release XLM-RoBERTa before loading mBERT
    global _xlm_pipeline
    _xlm_pipeline = None
    clear_gpu_memory()
    print("   XLM-RoBERTa released from GPU memory")

    # ── 2. mBERT training ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("mBERT TRAINING (WeightedTrainer + adaptive profile + tier-filtered data)")
    print("="*80)
    mbert    = mBERTSentimentClassifier()
    eval_res = mbert.train(train_ds, test_ds)
    clear_gpu_memory()

    # ── 3. Crisis classification (train set only, continuous scores) ─────────
    print("\n" + "="*80)
    print("CRISIS CLASSIFICATION (zero-shot continuous scores)")
    print("="*80)
    crisis_clf       = ZeroShotCrisisClassifier()
    quarterly_crisis = crisis_clf.aggregate_quarterly_train_only(dataset.train_df.copy())
    clear_gpu_memory()

    # ── 4. Sentiment vectors — training set ──────────────────────────────────
    print("\n" + "="*80)
    print("GENERATING SENTIMENT VECTORS")
    print("="*80)

    print("   Processing training set...")
    train_preds, train_scores, _ = mbert.predict(dataset.train_df['title'].tolist())
    dataset.train_df['sentiment_pred']  = train_preds
    dataset.train_df['sentiment_score'] = train_scores

    quarterly_sent_train = (dataset.train_df
                            .groupby('year_quarter')
                            .agg(sentiment_mean=('sentiment_score', 'mean'),
                                 positive_proportion=('sentiment_pred',
                                                      lambda x: (x==0).mean()))
                            .reset_index()
                            .rename(columns={'year_quarter': 'quarter'}))

    print("   Processing test set...")
    test_preds, test_scores, _ = mbert.predict(dataset.test_df['title'].tolist())
    dataset.test_df['sentiment_pred']  = test_preds
    dataset.test_df['sentiment_score'] = test_scores

    quarterly_sent_test = (dataset.test_df
                           .groupby('year_quarter')
                           .agg(sentiment_mean=('sentiment_score', 'mean'),
                                positive_proportion=('sentiment_pred',
                                                     lambda x: (x==0).mean()))
                           .reset_index()
                           .rename(columns={'year_quarter': 'quarter'}))

    # Merge crisis scores onto training quarters
    quarterly_train_full = quarterly_sent_train.merge(
        quarterly_crisis, on='quarter', how='left')
    quarterly_train_full['data_split'] = 'train'

    quarterly_test_full = quarterly_sent_test.copy()
    for col in ['crisis_economic', 'crisis_political', 'crisis_disaster',
                'crisis_index', 'crisis_proportion']:
        quarterly_test_full[col] = np.nan
    quarterly_test_full['data_split'] = 'test'

    quarterly_combined = (pd.concat([quarterly_train_full, quarterly_test_full],
                                    ignore_index=True)
                          .sort_values('quarter'))

    # ── 5. Ablation studies (must run before language-weighted aggregation) ──
    ablation, lang_perf = run_ablations(dataset, mbert)

    # ── 6. FIX D2: language-weighted aggregation AFTER ablation ──────────────
    print("\n" + "="*80)
    print("LANGUAGE-WEIGHTED SENTIMENT AGGREGATION (FIX D2 — post-ablation F1 weights)")
    print("="*80)

    # Save language F1 weights table (derived from actual ablation results)
    lang_weight_rows = []
    lang_names = {
        'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu',
        'ml': 'Malayalam', 'bn': 'Bengali', 'pa': 'Punjabi', 'gu': 'Gujarati'
    }
    for code, perf in sorted(lang_perf.items()):
        lang_weight_rows.append({
            'Language': lang_names.get(code, code),
            'Code':     code,
            'F1_weighted': round(perf.get('f1_weighted', 0.700), 3),
            'Weight':      round(perf.get('f1_weighted', 0.700), 3),
            'N_articles':  perf.get('n', 0),
            'Labeler':     perf.get('labeler', 'XLM-RoBERTa'),
        })
    lang_weight_df = pd.DataFrame(lang_weight_rows)
    lang_weight_df.to_csv('/kaggle/working/language_f1_weights.csv', index=False)
    print("   ✅ Saved: language_f1_weights.csv (actual F1 scores from ablation)")
    print(lang_weight_df.to_string(index=False))

    # Compute weighted aggregation for both splits using actual F1 weights
    all_pred_df = pd.concat([
        dataset.train_df.assign(sentiment_pred=train_preds, sentiment_score=train_scores),
        dataset.test_df.assign(sentiment_pred=test_preds,   sentiment_score=test_scores),
    ], ignore_index=True)

    quarterly_weighted = compute_language_weighted_sentiment(all_pred_df, lang_perf)

    # Merge weighted columns into combined quarterly output
    quarterly_combined = quarterly_combined.merge(
        quarterly_weighted, on='quarter', how='left')

    # ── 7. Save sentiment_vectors.csv ────────────────────────────────────────
    quarterly_combined.to_csv('/kaggle/working/sentiment_vectors.csv',  index=False)
    quarterly_crisis.to_csv('/kaggle/working/crisis_index_train.csv',   index=False)
    print(f"\n✅ sentiment_vectors.csv        ({len(quarterly_combined)} quarters)")
    print(f"✅ crisis_index_train.csv        ({len(quarterly_crisis)} quarters)")
    print(f"   Columns: {list(quarterly_combined.columns)}")

    # ── 8. Stability analysis ────────────────────────────────────────────────
    regime_df = analyze_sentiment_stability(dataset.train_df, dataset.test_df,
                                            window=CONFIG['stability_window'])

    # ── 9. A8: Event annotation table (FIX G — no broken file-check) ─────────
    print("\n" + "="*80)
    print("A8: EVENT-ANNOTATED REGIME CHANGE TABLE")
    print("="*80)

    KNOWN_EVENTS = [
        ('2016Q4', 'Indian Demonetisation (Nov 2016)',
         'Negative — NRI remittance surge to help families'),
        ('2018Q1', 'US H1-B visa uncertainty (early 2018)',
         'Negative — skilled worker remittance anxiety'),
        ('2019Q4', 'COVID-19 early signals (Dec 2019)',
         'Neutral → building Negative'),
        ('2020Q1', 'COVID-19 lockdowns begin (Mar 2020)',
         'Negative — global remittance shock'),
        ('2020Q2', 'Gulf remittance collapse Q2 2020',
         'Negative — oil-price crash + lockdowns'),
        ('2021Q1', 'Vaccine rollout optimism (Jan 2021)',
         'Positive — recovery signal'),
        ('2021Q3', 'Post-COVID remittance surge',
         'Positive — delayed transfers + recovery'),
        ('2022Q1', 'Russia-Ukraine conflict (Feb 2022)',
         'Negative — global uncertainty'),
        ('2023Q2', 'Indian Rupee stabilisation',
         'Positive — favourable exchange rate'),
        ('2024Q1', 'India general election lead-up',
         'Neutral — policy uncertainty'),
    ]

    events_df = pd.DataFrame(KNOWN_EVENTS,
                             columns=['quarter', 'event', 'expected_sentiment'])

    # FIX G: join with stability results directly (no broken file-check)
    if regime_df is not None:
        events_df = events_df.merge(
            regime_df[['quarter', 'positive_rate', 'negative_rate',
                       'neutral_rate', 'regime_change']],
            on='quarter', how='left')
        print("   ✅ Joined event annotations with detected regime changes")

    events_df.to_csv('/kaggle/working/event_annotation_table.csv', index=False)
    print("\n  Known event annotations (Table A8):")
    print(f"  {'Quarter':<10} {'Event':<52} Direction")
    print(f"  {'-'*10} {'-'*52} {'-'*30}")
    for _, row in events_df.iterrows():
        print(f"  {row['quarter']:<10} {row['event']:<52} "
              f"{row['expected_sentiment'][:30]}")
    print(f"\n  ✓ Saved: event_annotation_table.csv")

    # ── 10. Correlation (FIX B — handles annual vs quarterly automatically) ──
    analyze_sentiment_remittance_correlation(quarterly_combined)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("✅ NLPTS5 v6 COMPLETE")
    print("="*80)
    print(f"   Test F1 (weighted): {eval_res.get('eval_f1',       0):.3f}")
    print(f"   Test F1 (macro):    {eval_res.get('eval_f1_macro', 0):.3f}")
    print(f"\n   Training articles used: {len(dataset.train_df):,} "
          f"(tier ≥ {CONFIG['min_train_tier']})")
    print(f"   Test articles:          {len(dataset.test_df):,}")

    print("\n📁 Output files:")
    for fname in [
        'sentiment_vectors.csv',
        'crisis_index_train.csv',
        'ablation_results.json',
        'sentiment_stability_analysis.csv',
        'sentiment_correlation_analysis.json',
        'language_f1_weights.csv',
        'event_annotation_table.csv',
    ]:
        print(f"   /kaggle/working/{fname}")

    print("\n🔧 ALL FIXES APPLIED:")
    print("   ✅ [v2]   JSON: numpy → Python types")
    print("   ✅ [v3]   classification_report: labels=[0,1,2] always passed")
    print("   ✅ [v3]   XLM label map: LABEL_N + human-readable, validated at load")
    print("   ✅ [v3]   VADER ablation: English-only (fair comparison)")
    print("   ✅ [v4]   Trainer: processing_class= / tokenizer= version shim")
    print("   ✅ [v4]   logging_dir= suppressed for transformers >=4.50")
    print("   ✅ [v5-A] WeightedTrainer: inverse-frequency class weights")
    print("   ✅ [v5-B] Correlation: auto-detects annual vs quarterly data")
    print("   ✅ [v5-C] Adaptive training profile for small datasets (<12K)")
    print(f"   ✅ [v6-D] relevance_tier filter: training uses tier ≥ {CONFIG['min_train_tier']}")
    print("   ✅ [v6-E] crisis_flag reused from NLPTS4 CSV (no recomputation)")
    print("   ✅ [v6-F] ai4bharat/indic-bert removed — was crashing (no sentiment head)")
    print("   ✅ [v6-G] A8 event join fixed — no broken file-check")
    print("   ✅ [v6-D2] Language F1 weights computed AFTER ablation (not hardcoded)")


if __name__ == "__main__":
    main()