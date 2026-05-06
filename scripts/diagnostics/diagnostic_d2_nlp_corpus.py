"""
DIAGNOSTIC D2 — NLP Corpus Transparency
=========================================
Reviewer concern: sampling protocol & news collection poorly described.

HOW TO USE ON KAGGLE:
  • Paste this entire cell into a new Kaggle code cell and run it.
  • It does NOT require any prior cells or files.
  • If remittances_news_final.csv exists in /kaggle/working/ it will be used;
    otherwise a synthetic multilingual corpus is generated so the diagnostic
    always completes with a realistic full report.

Output: printed report + d2_corpus_stats.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json, re, warnings
warnings.filterwarnings("ignore")

sep = "=" * 72

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

LANG_NAMES = {
    "en": "English", "hi": "Hindi",  "ta": "Tamil",
    "te": "Telugu",  "ml": "Malayalam", "bn": "Bengali",
    "pa": "Punjabi", "gu": "Gujarati",
}

SAMPLE_QUERIES = {
    "en": ["india remittance", "NRI money transfer", "hawala regulation",
           "SWIFT india", "bank transfer india"],
    "hi": ["भारत प्रेषण", "विदेश से पैसा"],
    "ta": ["இந்தியா பணம்", "வெளிநாட்டு பணம்"],
    "te": ["భారత్ రెమిటెన్స్", "విదేశీ నగదు"],
    "ml": ["ഇന്ത്യ പണം", "പ്രവാസി"],
    "bn": ["ভারতে রেমিট্যান্স", "বৈদেশিক অর্থ"],
    "pa": ["ਭਾਰਤ ਪੈਸੇ", "ਵਿਦੇਸ਼ੀ ਫੰਡ"],
    "gu": ["ભારત રેમિટન્સ", "વિદેશ ફ઼ंड"],
}

SAMPLE_SOURCES = [
    "timesofindia.com","economictimes.com","hindustantimes.com",
    "thehindu.com","ndtv.com","moneycontrol.com","livemint.com",
    "businessstandard.com","financialexpress.com","deccanherald.com",
    "bbc.com","reuters.com","bloomberg.com","ft.com","wsj.com",
]

def make_synthetic_news(n=2000):
    np.random.seed(99)
    langs = list(LANG_NAMES.keys())
    weights = [0.55, 0.15, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03]
    records = []
    for i in range(n):
        lang = np.random.choice(langs, p=weights)
        q = np.random.choice(SAMPLE_QUERIES.get(lang, SAMPLE_QUERIES["en"]))
        title_en = f"India remittance {['rises','falls','stable'][np.random.randint(3)]} amid {['COVID','policy','inflation','reform'][np.random.randint(4)]} news item {i}"
        date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(np.random.uniform(0, 365*8)))
        domain = np.random.choice(SAMPLE_SOURCES)
        rel = float(np.random.uniform(0.1, 15.0))
        records.append({
            "title": title_en,
            "language": lang,
            "seendate": date,
            "url": f"https://{domain}/article-{i}",
            "domain": domain,
            "query": q,
            "source_api": np.random.choice(["gdelt","newsapi","gdelt"]),
            "relevance_score": rel,
        })
    return pd.DataFrame(records)

# ── Load or synthesise ─────────────────────────────────────────────────────────
# ── Load or synthesise (FIXED VERSION) ─────────────────────────────────────────
news_path  = find("remittances_news_final.csv")
meta_path  = find("collection_metadata.json")
quarterly_path = find("remittances_quarterly_by_language.csv")

SYNTHETIC_MODE = news_path is None

print(sep)
print("D2 — NLP CORPUS TRANSPARENCY DIAGNOSTIC")
print(f"     Mode: {'SYNTHETIC (demo)' if SYNTHETIC_MODE else 'REAL DATA'}")
print(sep)

if SYNTHETIC_MODE:
    print("⚠  remittances_news_final.csv not found — using SYNTHETIC corpus.\n")
    df = make_synthetic_news(2000)
else:
    df = pd.read_csv(news_path)
    
    # FIX: Explicitly convert to datetime. 'format="ISO8601"' or 'dayfirst' 
    # is usually unnecessary if we use errors='coerce'. 
    df['seendate'] = pd.to_datetime(df['seendate'], errors='coerce')
    
    # Check if conversion failed for any rows
    invalid_count = df['seendate'].isna().sum()
    if invalid_count > 0:
        print(f"⚠  Warning: {invalid_count} rows had invalid date formats and were dropped.")
        df = df.dropna(subset=['seendate'])

# ── 1. Overall corpus summary ─────────────────────────────────────────────────
print("\n[1] Overall corpus summary")
print(f"  Total articles   : {len(df):,}")
print(f"  Unique URLs      : {df['url'].nunique():,}")
print(f"  Duplicate URLs   : {len(df) - df['url'].nunique():,}")

# This line will no longer crash because seendate is now a DatetimeIndex
print(f"  Date range       : {df['seendate'].min().date()} → {df['seendate'].max().date()}")
print(f"  Unique sources   : {df['domain'].nunique():,}")
print(f"  Unique queries   : {df['query'].nunique() if 'query' in df.columns else 'n/a'}")

total = len(df)
# ── 2. Per-language article counts ────────────────────────────────────────────
print("\n[2] Per-language article counts")
lang_counts = df["language"].value_counts()
print(f"  {'Language':<12} {'ISO':>5}  {'Count':>8}  {'%':>7}  {'Source APIs'}")
print("  " + "-" * 60)
for iso, count in lang_counts.items():
    apis = df[df["language"] == iso]["source_api"].unique() if "source_api" in df.columns else []
    print(f"  {LANG_NAMES.get(iso, iso):<12} {iso:>5}  {count:>8,}  {count/total*100:>6.1f}%  "
          f"{', '.join(apis[:3])}")

# ── 3. Query selection criteria ───────────────────────────────────────────────
print("\n[3] Query selection criteria")
if "query" in df.columns:
    en_queries = df[df["language"] == "en"]["query"].value_counts()
    print(f"  English queries used     : {en_queries.index.nunique()}")
    print(f"  Top 5 English queries by yield:")
    for q, c in en_queries.head(5).items():
        print(f"    '{q}'  →  {c} articles")

    print(f"\n  Multilingual queries used:")
    for iso in [l for l in df["language"].unique() if l != "en"]:
        sub = df[df["language"] == iso]
        nq = sub["query"].nunique() if "query" in sub.columns else 0
        print(f"    {LANG_NAMES.get(iso, iso):<12} ({iso}): {nq} unique queries")
else:
    print("  [SKIP] 'query' column not found.")

# ── 4. Top news sources ───────────────────────────────────────────────────────
print("\n[4] Top news sources by volume")
top_sources = df["domain"].value_counts().head(15)
for src, cnt in top_sources.items():
    print(f"  {src:<40} {cnt:>6,}")

if "source_api" in df.columns:
    print("\n[5] Articles by collection API")
    for api, cnt in df["source_api"].value_counts().items():
        print(f"  {api:<30} {cnt:>8,}  ({cnt/total*100:.1f}%)")

# ── 5. Duplicate handling ─────────────────────────────────────────────────────
print("\n[6] Duplicate handling")
n_dup_url   = len(df) - df["url"].nunique()
n_dup_title = len(df) - df["title"].nunique()
print(f"  Exact-URL duplicates   : {n_dup_url}")
print(f"  Duplicate titles       : {n_dup_title}")
print("  Deduplication method   : URL-level seen_urls set (in collector)")
print("  Near-duplicate check   : not applied (future work)")

# ── 6. Text quality metrics ───────────────────────────────────────────────────
print("\n[7] Text quality descriptive statistics")
df["title_len"] = df["title"].fillna("").str.len()
df["word_count"] = df["title"].fillna("").str.split().str.len()
stats = df[["title_len", "word_count"]].describe().T
stats.columns = [c.title() for c in stats.columns]
print(stats.to_string())

print(f"\n  Empty titles   : {(df['title'].isna() | (df['title'] == '')).sum()}")
print(f"  Min word count : {df['word_count'].min()}")
print(f"  Max word count : {df['word_count'].max()}")
titles_lt5 = (df["word_count"] < 5).sum()
print(f"  Titles < 5 words (low quality): {titles_lt5} ({titles_lt5/total*100:.1f}%)")

# ── 7. Relevance score distribution ──────────────────────────────────────────
if "relevance_score" in df.columns:
    print("\n[8] Relevance score distribution")
    print(f"  English   threshold: 0.8%  (from notebook CONFIG)")
    print(f"  Multilingual threshold: 0.05% (very lenient, to maximise coverage)")
    print()
    for iso in df["language"].unique():
        sub = df[df["language"] == iso]["relevance_score"]
        print(f"  {LANG_NAMES.get(iso, iso):<12} ({iso}): "
              f"mean={sub.mean():.2f}  median={sub.median():.2f}  "
              f"std={sub.std():.2f}  min={sub.min():.2f}  max={sub.max():.2f}")

# ── 8. Temporal coverage ───────────────────────────────────────────────────────
print("\n[9] Quarterly article volume (last 16 quarters)")
df["year_quarter"] = (df["seendate"].dt.year.astype(str) + "Q" +
                      df["seendate"].dt.quarter.astype(str))
q_counts = df.groupby("year_quarter").size().sort_index()
print(f"  {'Quarter':<10}  {'Articles':>10}  Bar")
for q, n in q_counts.tail(16).items():
    bar = "█" * min(int(n / max(q_counts.max(), 1) * 30), 30)
    print(f"  {q:<10}  {n:>10,}  {bar}")

# ── 9. Collection metadata ────────────────────────────────────────────────────
print("\n[10] Collection metadata")
if meta_path:
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    print(f"  Collected at    : {meta.get('collected', 'n/a')}")
    for k, v in meta.get("config", {}).items():
        print(f"    {k}: {v}")
else:
    print("  collection_metadata.json not found.")
    print("  (When running from real notebook, this file is saved by Cell 6.)")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
lang_summary = df.groupby("language").agg(
    count=("title", "count"),
    mean_words=("word_count", "mean"),
).reset_index()
if "relevance_score" in df.columns:
    lang_summary["mean_relevance"] = df.groupby("language")["relevance_score"].mean().values
lang_summary["language_name"] = lang_summary["language"].map(LANG_NAMES)
lang_summary.to_csv(out / "d2_corpus_stats.csv", index=False)
print(f"\n  Per-language summary → {out / 'd2_corpus_stats.csv'}")

if SYNTHETIC_MODE:
    print("\n  NOTE: All figures above are from SYNTHETIC data for demonstration.")
    print("  Upload remittances_news_final.csv to /kaggle/working/ for real results.")

print("\n" + sep)
print("D2 COMPLETE")
print(sep)
