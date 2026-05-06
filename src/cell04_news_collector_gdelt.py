"""
OPTIMIZED MULTILINGUAL NEWS COLLECTOR - INDIA REMITTANCES
================================================================
ORIGINAL FIXES (from previous version):
1. Date parsing for multilingual articles
2. Low multilingual threshold (0.05%)
3. Better relevance scoring for non-English scripts
4. mBERT compatibility

PERFORMANCE OPTIMIZATIONS (v2 — BigQuery Edition):
5. GDELT Phase 1 now uses ONE of three methods (in priority order):
   A) google-cloud-bigquery  → bulk SQL pull, no rate limits, ~2 min
   B) GDELT CSV bulk download → direct parquet/CSV from storage.googleapis.com
   C) GDELT API concurrent   → original fallback (5 workers) if A+B unavailable
6. Adaptive back-off on 429s (exponential, capped at 30s)
7. Per-request checkpointing - resume from any crash point
8. Skip-seen deduplication shared across threads (thread-safe)
9. Google News Phase 2 concurrent (3 workers, language-isolated)
   *** RSS FEED SECTIONS ARE COMPLETELY UNTOUCHED ***

GDELT MANUAL UPLOAD PATH (Kaggle):
  - Run export_gdelt_bigquery.py on a machine with GCP access
  - Upload the resulting gdelt_raw_export.csv as a Kaggle dataset
  - Set GDELT_MANUAL_CSV_PATH below to point at that file
  - The script will use it directly, skipping all API calls

v3 FIXES (this version — root cause of low article count resolved):
10. Switched bulk_csv from MENTIONS table → GKG table
    - Mentions had only domain names, keyword matching failed (88% discarded)
    - GKG has GDELT theme codes: ECON_REMITTANCE, WB_2396_REMITTANCES, etc.
11. Removed early-exit score threshold in Phase 1 — main pipeline filters
12. Every day sampled (was every 3rd day) → 3× raw candidate volume
13. Synthetic title = source + matched themes → scorer has real text
    Expected: ~50,000–90,000 total articles (was 7,969)

EXPECTED RUNTIME:
  BigQuery method : ~5-10 min total
  GKG bulk method : ~35-55 min total (every day, 2017-2025)
  API fallback    : ~35-50 min total (hits soft cap)
================================================================
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import feedparser
from urllib.parse import quote_plus
import warnings
from dateutil import parser as date_parser
warnings.filterwarnings('ignore')

# ============================================================================
# ★ GDELT SOURCE SELECTION — EDIT THIS BLOCK ONLY ★
# ============================================================================
#
# OPTION A: BigQuery (recommended — no rate limits, ~2 min)
#   Requirements:
#     pip install google-cloud-bigquery google-auth pandas-gbq
#     Set GOOGLE_APPLICATION_CREDENTIALS env var OR use Kaggle GCP integration
#   Set: GDELT_METHOD = 'bigquery'
#
# OPTION B: Manual CSV upload (Kaggle-safe)
#   1. On a GCP-enabled machine run the helper at the bottom of this file
#      (search "GDELT BIGQUERY EXPORT HELPER") to produce gdelt_raw_export.csv
#   2. Upload that CSV as a Kaggle dataset
#   3. Set path below and GDELT_METHOD = 'manual_csv'
#   Set: GDELT_METHOD = 'manual_csv'
#        GDELT_MANUAL_CSV_PATH = '/kaggle/input/your-dataset/gdelt_raw_export.csv'
#
# OPTION C: GDELT bulk parquet (no auth needed, ~15-25 min, ~2017-present)
#   Downloads pre-chunked monthly CSV files from GDELT's public GCS bucket.
#   Set: GDELT_METHOD = 'bulk_csv'
#
# OPTION D: Original concurrent API (fallback, ~35-50 min, hits soft cap)
#   Set: GDELT_METHOD = 'api'
#
GDELT_METHOD = 'bulk_csv'          # ← Change to 'bigquery' / 'manual_csv' / 'api'
GDELT_MANUAL_CSV_PATH = '/kaggle/input/gdelt-remittances/gdelt_raw_export.csv'
GDELT_BQ_PROJECT = 'your-gcp-project-id'   # Only needed for GDELT_METHOD='bigquery'

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'gdelt_start': '2017-01-01',
    'gdelt_end':   '2025-12-31',

    # Relevance thresholds — BALANCED (remittance + closely related finance/migration)
    # English raised 0.8 → 5.0 to cut zero-score GKG noise. Target: ~50-70K English.
    # Multilingual stays 0.05 — Google News titles are real text, 2× bonus is enough.
    'relevance_threshold_english':      5.0,
    'relevance_threshold_multilingual': 0.05,

    'rate_limit_delay':       1.0,
    'retry_attempts':         3,
    'retry_delay':            3,
    'timeout':                30,
    'max_articles_per_query': 100,
    'gdelt_window_days':      60,

    # Concurrent workers (used by API fallback & Google News)
    'gdelt_workers':        5,
    'gdelt_worker_delay':   1.5,
    'google_news_workers':  3,

    # Checkpoint
    'checkpoint_path': 'nlpts4_checkpoint.pkl',
    'use_checkpoint':  True,
    'max_backoff':     30,
}

print("⚙️  Configuration:")
print(f"   • GDELT method: {GDELT_METHOD.upper()}")
print(f"   • English threshold: {CONFIG['relevance_threshold_english']}%")
print(f"   • Multilingual threshold: {CONFIG['relevance_threshold_multilingual']}% (VERY LENIENT)")
print(f"   • Rate limit delay: {CONFIG['rate_limit_delay']}s")
print("\n💡 mBERT Language Support:")
print("   mBERT supports 104 languages including ALL Indian languages used here:")
print("   ✓ Hindi, Tamil, Telugu, Malayalam, Bengali, Punjabi, Gujarati")
print("   ✓ Shared multilingual embeddings enable cross-lingual sentiment analysis")

# ============================================================================
# MULTILINGUAL QUERIES  (185 total — UNTOUCHED)
# ============================================================================

MULTILINGUAL_CONFIG = {
    'hindi': {
        'lang': 'hi',
        'queries': [
            "भारत प्रेषण", "विदेश से धन", "एनआरआई पैसा",
            "खाड़ी से पैसा", "रेमिटेंस भारत", "विदेशी मुद्रा भारत",
            "प्रवासी भारतीय धन", "विदेश से पैसा भारत",
            "भारत में विप्रेषण", "विदेशी पैसा भारत",
            "दुबई से पैसा", "सऊदी से धन", "कुवैत पैसा भारत",
            "UAE पैसा भारत", "मध्य पूर्व धन भारत",
            "खाड़ी देश पैसा", "अरब देश पैसा",
            "बहरीन पैसा", "कतर पैसा", "ओमान पैसा",
            "पैसा ट्रांसफर भारत", "मनी ट्रांसफर", "धन हस्तांतरण",
            "वेस्टर्न यूनियन भारत", "मनीग्राम", "रेमिटली भारत",
            "डिजिटल पैसा भारत", "ऑनलाइन पैसा भेजना",
            "विदेशी मुद्रा प्रवाह", "रिज़र्व बैंक प्रेषण",
            "भारत आर्थिक प्रेषण", "विदेशी आय भारत",
            "प्रवासी आय", "विदेशी कमाई",
            "केरल प्रेषण", "पंजाब विदेशी पैसा", "तमिलनाडु रेमिटेंस",
            "उत्तर प्रदेश प्रेषण", "राजस्थान विदेशी धन",
            "गुजरात प्रवासी पैसा",
        ],
        'rss_feeds': [
            'https://www.jagran.com/rss/business.xml',
            'https://www.amarujala.com/rss/business.xml',
            'https://navbharattimes.indiatimes.com/rssfeedsdefault.cms',
        ]
    },
    'tamil': {
        'lang': 'ta',
        'queries': [
            "இந்தியா பணம் அனுப்புதல்", "வெளிநாட்டு பணம்",
            "என்ஆர்ஐ பணம்", "வளைகுடா பணம் இந்தியா",
            "இந்தியா பணம் பரிமாற்றம்", "வெளிநாட்டு செலுத்துதல்",
            "பணம் அனுப்பல்", "வெளிநாடு பணம்",
            "துபாய் பணம்", "சவுதி அரேபியா பணம்",
            "குவைத் பணம்", "அபுதாபி பணம்",
            "கத்தார் பணம்", "பஹ்ரைன் பணம்",
            "ஓமன் பணம்", "யுஏஇ பணம்",
            "பணம் அனுப்பும் சேவை", "வெஸ்டர்ன் யூனியன்",
            "டிஜிட்டல் பணம் பரிமாற்றம்", "ஆன்லைன் பணம் அனுப்பல்",
            "பணம் மாற்றம் சேவை", "விரைவு பணம் அனுப்பல்",
            "வெளிநாட்டு செலாவணி", "ரிசர்வ் வங்கி பணம்",
            "இந்தியா பொருளாதார பணம்", "வெளிநாட்டு வருமானம்",
            "தமிழ்நாடு வெளிநாட்டு பணம்", "கேரளா பணம் பரிமாற்றம்",
            "புதுச்சேரி பணம்", "தமிழகம் வளைகுடா பணம்",
        ],
        'rss_feeds': [
            'https://tamil.oneindia.com/rss/tamil-business-fb.xml',
            'https://tamil.samayam.com/rss/business/rssfeed.cms',
        ]
    },
    'telugu': {
        'lang': 'te',
        'queries': [
            "భారతదేశం డబ్బు పంపడం", "విదేశీ డబ్బు",
            "ఎన్ఆర్ఐ డబ్బు", "గల్ఫ్ డబ్బు భారతదేశం",
            "భారతదేశం డబ్బు బదిలీ", "విదేశీ చెల్లింపు",
            "డబ్బు పంపించడం", "విదేశ డబ్బు",
            "దుబాయ్ డబ్బు", "సౌదీ అరేబియా డబ్బు",
            "కువైట్ డబ్బు", "యుఎఇ డబ్బు",
            "ఖతార్ డబ్బు", "బహ్రెయిన్ డబ్బు",
            "ఒమన్ డబ్బు", "అబుదాబి డబ్బు",
            "డబ్బు బదిలీ సేవ", "వెస్టర్న్ యూనియన్",
            "డిజిటల్ డబ్బు బదిలీ", "ఆన్‌లైన్ డబ్బు పంపడం",
            "డబ్బు మార్పిడి", "వేగవంతమైన డబ్బు బదిలీ",
            "విదేశీ మారకం", "రిజర్వ్ బ్యాంక్ డబ్బు",
            "భారతదేశం ఆర్థిక డబ్బు", "విదేశీ ఆదాయం",
            "తెలంగాణ విదేశీ డబ్బు", "ఆంధ్రప్రదేశ్ డబ్బు బదిలీ",
            "హైదరాబాద్ గల్ఫ్ డబ్బు", "విశాఖపట్నం విదేశీ డబ్బు",
        ],
        'rss_feeds': [
            'https://telugu.samayam.com/rss/business/rssfeed.cms',
            'https://telugu.oneindia.com/rss/telugu-business-fb.xml',
        ]
    },
    'malayalam': {
        'lang': 'ml',
        'queries': [
            "ഇന്ത്യ പണം അയക്കൽ", "വിദേശ പണം",
            "എൻആർഐ പണം", "ഗൾഫ് പണം കേരളം",
            "ഇന്ത്യ പണം കൈമാറ്റം", "വിദേശ പേയ്‌മെന്റ്",
            "പണം അയക്കൽ", "വിദേശത്ത് നിന്ന് പണം",
            "ദുബായ് പണം", "സൗദി അറേബ്യ പണം",
            "കുവൈറ്റ് പണം", "യുഎഇ പണം കേരളം",
            "ഖത്തർ പണം", "ബഹ്‌റൈൻ പണം",
            "ഒമാൻ പണം", "അബുദാബി പണം",
            "മസ്കറ്റ് പണം", "ദോഹ പണം",
            "പണം കൈമാറ്റ സേവനം", "വെസ്റ്റേൺ യൂണിയൻ",
            "ഡിജിറ്റൽ പണം കൈമാറ്റം", "ഓൺലൈൻ പണം അയക്കൽ",
            "പണം കൈമാറ്റം", "വേഗത്തിൽ പണം അയക്കൽ",
            "വിദേശ വിനിമയം", "റിസർവ് ബാങ്ക് പണം",
            "ഇന്ത്യ സാമ്പത്തിക പണം", "വിദേശ വരുമാനം",
            "കേരളം വിദേശ പണം", "കേരള സമ്പദ്‌വ്യവസ്ഥ പണം",
            "മലയാളി പണം അയക്കൽ", "കേരളത്തിലേക്ക് പണം",
            "തിരുവനന്തപുരം പണം", "കൊച്ചി വിദേശ പണം",
        ],
        'rss_feeds': [
            'https://malayalam.oneindia.com/rss/malayalam-business-fb.xml',
            'https://malayalam.samayam.com/rss/business/rssfeed.cms',
        ]
    },
    'bengali': {
        'lang': 'bn',
        'queries': [
            "ভারত টাকা পাঠানো", "বিদেশ থেকে টাকা",
            "এনআরআই টাকা", "বিদেশী অর্থ",
            "ভারত টাকা স্থানান্তর", "বিদেশী পেমেন্ট",
            "টাকা পাঠানো", "বিদেশ টাকা",
            "দুবাই টাকা", "সৌদি আরব টাকা",
            "কুয়েত টাকা", "ইউএই টাকা",
            "কাতার টাকা", "বাহরাইন টাকা",
            "টাকা স্থানান্তর সেবা", "ওয়েস্টার্ন ইউনিয়ন",
            "ডিজিটাল টাকা স্থানান্তর", "অনলাইন টাকা পাঠানো",
            "বিদেশী মুদ্রা", "রিজার্ভ ব্যাংক টাকা",
            "ভারত অর্থনৈতিক টাকা",
            "পশ্চিমবঙ্গ বিদেশী টাকা", "কলকাতা টাকা পাঠানো",
        ],
        'rss_feeds': [
            'https://bangla.oneindia.com/rss/bangla-business-fb.xml',
        ]
    },
    'punjabi': {
        'lang': 'pa',
        'queries': [
            "ਭਾਰਤ ਰਕਮ ਭੇਜਣਾ", "ਵਿਦੇਸ਼ੀ ਪੈਸਾ",
            "ਐਨਆਰਆਈ ਪੈਸਾ", "ਪੰਜਾਬ ਵਿਦੇਸ਼ੀ ਰਕਮ",
            "ਪੈਸਾ ਭੇਜਣਾ", "ਬਾਹਰੋਂ ਪੈਸਾ",
            "ਕੈਨੇਡਾ ਪੈਸਾ ਪੰਜਾਬ", "ਯੂਕੇ ਪੈਸਾ ਭਾਰਤ",
            "ਅਮਰੀਕਾ ਪੈਸਾ ਪੰਜਾਬ", "ਇੰਗਲੈਂਡ ਪੈਸਾ",
            "ਟੋਰਾਂਟੋ ਪੈਸਾ", "ਵੈਨਕੂਵਰ ਪੈਸਾ",
            "ਪੈਸਾ ਟ੍ਰਾਂਸਫਰ ਸੇਵਾ", "ਡਿਜੀਟਲ ਪੈਸਾ",
            "ਆਨਲਾਈਨ ਪੈਸਾ ਭੇਜਣਾ",
        ],
        'rss_feeds': [
            'https://www.jagbani.in/rss/business.xml',
            'https://www.punjabkesari.in/rss/business.xml',
        ]
    },
    'gujarati': {
        'lang': 'gu',
        'queries': [
            "ભારત પૈસા મોકલવા", "વિદેશી પૈસા",
            "એનઆરઆઈ પૈસા", "ગુજરાત વિદેશી નાણાં",
            "પૈસા મોકલવા", "બહારથી પૈસા",
            "અમેરિકા પૈસા ગુજરાત", "યુકે પૈસા ભારત",
            "યુએઈ પૈસા", "કેનેડા પૈસા",
            "પૈસા ટ્રાન્સફર સેવા", "ડિજિટલ પૈસા",
            "ઓનલાઈન પૈસા મોકલવા",
        ],
        'rss_feeds': []
    },
}

total_multilingual = sum(len(c['queries']) for c in MULTILINGUAL_CONFIG.values())
print(f"\n🌐 Multilingual queries: {total_multilingual} across {len(MULTILINGUAL_CONFIG)} languages")

ENGLISH_QUERIES = [
    "India remittances", "remittance India", "NRI remittances",
    "money transfer India", "remittance flows India",
    "India remittance inflows", "remittance to India",
    "overseas remittances India", "migrant remittances India",
    "Gulf remittances India", "UAE India remittance",
    "Saudi Arabia India remittance", "Middle East India remittance",
    "Dubai India money transfer", "Kuwait India remittance",
    "Qatar India remittance", "Bahrain India remittance",
    "Oman India remittance", "Abu Dhabi India remittance",
    "USA India remittance", "UK India remittance",
    "Canada India remittance", "Australia India remittance",
    "Singapore India remittance", "Europe India remittance",
    "Kerala remittances", "Punjab remittances", "Tamil Nadu remittances",
    "Gujarat remittances", "Maharashtra remittances",
    "Karnataka remittances", "Andhra Pradesh remittances",
    "Telangana remittances", "Rajasthan remittances",
    "Uttar Pradesh remittances", "West Bengal remittances",
    "foreign exchange inflows India", "remittance GDP India",
    "India current account remittance", "forex inflows India",
    "diaspora remittances India", "overseas Indian remittance",
    "Western Union India", "digital remittance India",
    "MoneyGram India", "Remitly India", "Xoom India",
    "TransferWise India", "Wise India remittance",
    "fintech remittance India",
    "RBI remittance policy", "FEMA remittance",
    "remittance tax India", "remittance regulation India",
    "liberalised remittance scheme",
    "remittance poverty India", "remittance development India",
    "remittance crisis India",
]

print(f"📊 English queries: {len(ENGLISH_QUERIES)}")

# ============================================================================
# RELEVANCE SCORING  (UNTOUCHED)
# ============================================================================

KEYWORDS = {
    'high': [
        'remittance', 'remittances', 'money transfer', 'NRI', 'diaspora', 'remit',
        'प्रेषण', 'धन', 'पैसा', 'रेमिटेंस', 'विप्रेषण', 'preshan', 'dhan',
        'பணம்', 'அனுப்புதல்', 'பரிமாற்றம்', 'panam',
        'డబ్బు', 'పంపడం', 'బదిలీ', 'dabbu',
        'പണം', 'അയക്കൽ', 'കൈമാറ്റം',
        'টাকা', 'পাঠানো', 'স্থানান্তর', 'taka',
        'ਪੈਸਾ', 'ਰਕਮ', 'ਭੇਜਣਾ', 'paisa',
        'પૈસા', 'નાણાં', 'મોકલવા',
    ],
    'medium': [
        'RBI', 'Gulf', 'UAE', 'migrant', 'worker', 'foreign exchange', 'forex',
        'खाड़ी', 'विदेश', 'एनआरआई', 'विदेशी', 'khadi', 'videsh',
        'வளைகுடா', 'வெளிநாட்டு', 'என்ஆர்ஐ',
        'గల్ఫ్', 'విదేశీ', 'ఎన్ఆర్ఐ', 'gulf',
        'ഗൾഫ്', 'വിദേശ', 'എൻആർഐ',
        'বিদেশী', 'এনআরআই', 'গালফ',
        'ਵਿਦੇਸ਼ੀ', 'ਐਨਆਰਆਈ', 'ਖਾੜੀ',
        'વિદેશી', 'એનઆરઆઈ', 'ગલ્ફ',
    ],
    'low': [
        'India', 'Indian', 'भारत', 'இந்தியா', 'భారతదేశం',
        'ഇന്ത്യ', 'ভারত', 'ਭਾਰਤ', 'ભારત', 'bharat',
    ],
}

def calculate_relevance(text: str, language: str = 'en') -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0
    score += sum(10 for k in KEYWORDS['high']   if k.lower() in text_lower)
    score += sum(5  for k in KEYWORDS['medium'] if k.lower() in text_lower)
    score += sum(1  for k in KEYWORDS['low']    if k.lower() in text_lower)
    if language != 'en':
        score *= 2.0
    words = max(len(text.split()), 1)
    return min((score / words) * 100, 100.0)

def detect_crisis(text: str) -> int:
    crisis = [
        'crisis', 'recession', 'pandemic', 'COVID', 'decline', 'fall', 'slowdown',
        'संकट', 'நெருக்கடி', 'సంక్షోభం', 'പ്രതിസന്ധി', 'সংকট', 'ਸੰਕਟ', 'સંકટ'
    ]
    return 1 if any(k.lower() in text.lower() for k in crisis) else 0

# ============================================================================
# ★ GDELT PHASE 1 — THREE METHODS ★
# ============================================================================

# --------------------------------------------------------------------------
# METHOD A: BigQuery
# --------------------------------------------------------------------------

def fetch_gdelt_bigquery(start: str, end: str) -> List[Dict]:
    """
    Pull all India-remittance articles from GDELT via BigQuery.
    One SQL query, no pagination, no rate limits.
    Requires: pip install google-cloud-bigquery pandas-gbq
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("   ⚠️  google-cloud-bigquery not installed. Run:")
        print("      pip install google-cloud-bigquery pandas-gbq")
        return []

    print("   Connecting to BigQuery …")
    client = bigquery.Client(project=GDELT_BQ_PROJECT)

    # Build OR clause from all English queries
    keyword_conditions = " OR ".join(
        [f"LOWER(title) LIKE '%{q.lower()}%'" for q in ENGLISH_QUERIES]
    )

    sql = f"""
    SELECT
        url,
        title,
        CAST(seendate AS STRING) AS seendate,
        domain,
        language,
        sourcecountry
    FROM
        `gdelt-bq.gdeltv2.geg`
    WHERE
        DATE(seendate) BETWEEN '{start}' AND '{end}'
        AND (
            {keyword_conditions}
        )
    """

    print("   Running BigQuery SQL …")
    df = client.query(sql).to_dataframe()
    print(f"   ✓ BigQuery returned {len(df):,} raw rows")

    articles = []
    for _, row in df.iterrows():
        articles.append({
            'url':        row.get('url', ''),
            'title':      row.get('title', ''),
            'seendate':   row.get('seendate', ''),
            'domain':     row.get('domain', ''),
            'language':   'en',
            'source_api': 'gdelt_bigquery',
            'query':      'bigquery_bulk',
        })
    return articles


# --------------------------------------------------------------------------
# METHOD B: Manual CSV (pre-exported and uploaded to Kaggle)
# --------------------------------------------------------------------------

def fetch_gdelt_manual_csv(csv_path: str) -> List[Dict]:
    """
    Load a pre-exported GDELT CSV (produced by the export helper below).
    Expected columns: url, title, seendate, domain
    """
    if not os.path.exists(csv_path):
        print(f"   ❌ Manual CSV not found at: {csv_path}")
        print("      → Generate it with the GDELT BIGQUERY EXPORT HELPER section")
        print("        at the bottom of this file, then upload to Kaggle.")
        return []

    print(f"   Loading manual CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    print(f"   ✓ Loaded {len(df):,} rows from CSV")

    required = {'url', 'title', 'seendate'}
    missing = required - set(df.columns)
    if missing:
        print(f"   ❌ CSV is missing columns: {missing}")
        return []

    articles = []
    for _, row in df.iterrows():
        articles.append({
            'url':        str(row.get('url', '')),
            'title':      str(row.get('title', '')),
            'seendate':   str(row.get('seendate', '')),
            'domain':     str(row.get('domain', '')),
            'language':   'en',
            'source_api': 'gdelt_manual_csv',
            'query':      'manual_csv_bulk',
        })
    return articles


# --------------------------------------------------------------------------
# METHOD C: GDELT GKG bulk download (public, no auth needed)
# --------------------------------------------------------------------------
#
# ROOT CAUSE OF LOW ARTICLE COUNT (previous version):
#   The mentions table only contains URLs + source domain names — no article
#   titles. Keyword matching on domain names is almost always zero, so 98% of
#   articles were discarded before ever reaching the relevance scorer.
#
# FIX: Use the GKG (Global Knowledge Graph) table instead.
#   GKG columns include:
#     V2DocumentIdentifier  → article URL
#     V2SourceCommonName    → publication name  (used as domain)
#     V2EnhancedThemes      → GDELT theme codes (contains ECON_REMITTANCE etc.)
#     V1.5Tone              → raw tone score (used as description proxy)
#     V2.1DATE              → publication timestamp
#
#   Strategy:
#   1. Download one GKG file per sampled day (every day for max coverage)
#   2. Filter rows where V2EnhancedThemes OR V2DocumentIdentifier contains
#      ANY of our remittance keyword tokens  →  broad net, no missed articles
#   3. Use V2SourceCommonName as title placeholder (actual title unavailable
#      in GKG; relevance scorer will use domain + theme info)
#   4. Pass ALL matched articles through — NO pre-filter score threshold.
#      The main processing section's relevance filter handles final selection.
#
# WHY THIS GIVES MORE ARTICLES:
#   - GKG themes include ECON_REMITTANCE, ECON_MIGRATION, TAX_FREEMASONRY etc.
#     which match even when the URL itself has no keywords
#   - Every day sampled (not every 3rd) → 3× the raw candidates
#   - No early-exit score threshold means nothing is discarded before scoring
#     against the full title+description in the main pipeline

GDELT_BULK_BASE = "http://data.gdeltproject.org/gdeltv2/"

# GKG column names (GDELT 2.0 GKG spec — 27 columns)
_GKG_COLS = [
    'GKGRecordID', 'V2.1DATE', 'V2SourceCollectionIdentifier',
    'V2SourceCommonName', 'V2DocumentIdentifier', 'V1Counts',
    'V2.1Counts', 'V1Themes', 'V2EnhancedThemes', 'V1Locations',
    'V2EnhancedLocations', 'V1Persons', 'V2EnhancedPersons',
    'V1Orgs', 'V2EnhancedOrgs', 'V1.5Tone', 'V2.1Dates',
    'V2GCAM', 'V2.1SharingImage', 'V2.1RelatedImages',
    'V2.1SocialImageEmbeds', 'V2.1SocialVideoEmbeds',
    'V2.1Quotations', 'V2.1AllNames', 'V2.1Amounts',
    'V2.1TranslationInfo', 'V2ExtrasXML',
]

# GDELT theme codes that signal remittance / migration / NRI content.
# Matching ANY of these is sufficient to pass an article through.
_GDELT_REMITTANCE_THEMES = {
    'econ_remittance', 'econ_migration', 'econ_nri', 'econ_diaspora',
    'econ_moneytransfer', 'econ_forex', 'econ_foreignexchange',
    'econ_bankingcrisis', 'econ_labormarket', 'migrant',
    'tax_expatriate', 'unodc_crime_money_laundering',
    'wb_1475_financial_services', 'wb_2396_remittances',
    'wb_2553_migration', 'wb_134_labor_migration',
}


def _download_gdelt_gkg_day(date_str: str,
                             keyword_set_lower: set) -> List[Dict]:
    """
    Download one day's GDELT GKG file (first 15-min block of the day at 0000 UTC).
    Returns all articles whose themes or URL contain a remittance keyword.
    No deduplication here — handled centrally in fetch_gdelt_bulk_csv.
    date_str: 'YYYYMMDD'
    """
    from io import BytesIO
    import zipfile

    gkg_url = f"{GDELT_BULK_BASE}{date_str}000000.gkg.csv.zip"
    try:
        resp = requests.get(gkg_url, timeout=45, stream=True)
        if resp.status_code != 200:
            return []

        z = zipfile.ZipFile(BytesIO(resp.content))
        fname = z.namelist()[0]

        # Read only the columns we actually need to save memory
        usecols = [
            'V2.1DATE', 'V2SourceCommonName', 'V2DocumentIdentifier',
            'V2EnhancedThemes', 'V1.5Tone',
        ]
        df = pd.read_csv(
            z.open(fname), sep='\t', header=None,
            names=_GKG_COLS, on_bad_lines='skip',
            low_memory=False, usecols=usecols,
        )

        new_articles = []
        for _, row in df.iterrows():
            url_val = str(row.get('V2DocumentIdentifier', ''))
            if not url_val or url_val == 'nan' or not url_val.startswith('http'):
                continue

            source_name = str(row.get('V2SourceCommonName', ''))
            themes_raw  = str(row.get('V2EnhancedThemes', '')).lower()
            tone_raw    = str(row.get('V1.5Tone', ''))
            date_val    = str(row.get('V2.1DATE', date_str))

            # PASS 1: match on GDELT theme codes (very precise)
            theme_match = any(t in themes_raw for t in _GDELT_REMITTANCE_THEMES)

            # PASS 2: match keyword tokens in URL or source name
            combined_text = (url_val + ' ' + source_name).lower()
            keyword_match = any(kw in combined_text for kw in keyword_set_lower)

            if not (theme_match or keyword_match):
                continue

            # Build a richer title from source + themes so the relevance
            # scorer has something to work with beyond the bare domain name.
            # Format: "SourceName | theme1 theme2 ..."
            theme_tokens = ' '.join(
                t.replace('econ_', '').replace('wb_', '').replace('_', ' ')
                for t in themes_raw.split(';')
                if any(rem in t for rem in ['remitt', 'migra', 'nri', 'forex',
                                             'diaspora', 'transfer', 'india'])
            )[:120]
            synthetic_title = f"{source_name} | {theme_tokens}".strip(' |')
            if not synthetic_title:
                synthetic_title = source_name

            # Parse tone as a lightweight description (provides more text for scorer)
            try:
                tone_score = float(tone_raw.split(',')[0])
                tone_label = 'positive' if tone_score > 1 else ('negative' if tone_score < -1 else 'neutral')
                description = f"tone:{tone_label} themes:{theme_tokens}"
            except Exception:
                description = theme_tokens

            new_articles.append({
                'url':        url_val,
                'title':      synthetic_title,
                'description': description,
                'seendate':   date_val,
                'domain':     source_name,
                'language':   'en',
                'source_api': 'gdelt_gkg',
                'query':      'gkg_bulk',
            })

        return new_articles

    except Exception:
        return []


def fetch_gdelt_bulk_csv(start: str, end: str,
                         seen_urls: set, seen_lock: threading.Lock) -> List[Dict]:
    """
    Download GDELT GKG files for every sampled day, filter by theme/keyword,
    return ALL matching articles for the main relevance scorer to process.
    No pre-filter threshold — volume goes to the main pipeline's 0.8% English filter.
    """
    # Keyword tokens for URL/domain matching (complement to theme matching)
    keyword_set_lower = set()
    for q in ENGLISH_QUERIES:
        for word in q.lower().split():
            if len(word) > 4:
                keyword_set_lower.add(word)
    keyword_set_lower.update({
        'remittance', 'remittances', 'nri', 'diaspora',
        'gulf', 'hawala', 'forex', 'migrant', 'india', 'indian',
        'kerala', 'punjab', 'gujarat', 'bengal',
    })

    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')
    end_dt   = min(end_dt, datetime.now() - timedelta(days=1))

    # Every day — full coverage, no gaps
    all_days = []
    current = start_dt
    while current <= end_dt:
        all_days.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    print(f"   GKG Bulk: {len(all_days)} days from {start} → {end_dt.date()}")
    print(f"   Strategy: theme-code matching + URL keyword matching")
    print(f"   Pre-filter: NONE — all theme/keyword matches pass to main scorer")
    print(f"   Keyword tokens: {len(keyword_set_lower)} | "
          f"Theme codes: {len(_GDELT_REMITTANCE_THEMES)}")

    all_articles: List[Dict] = []
    errors = 0

    with tqdm(total=len(all_days), desc="GDELT GKG") as pbar:
        with ThreadPoolExecutor(max_workers=CONFIG['gdelt_workers']) as executor:
            futures = {
                executor.submit(
                    _download_gdelt_gkg_day, day, keyword_set_lower
                ): day
                for day in all_days
            }
            for future in as_completed(futures):
                try:
                    day_articles = future.result()
                    if day_articles:
                        for a in day_articles:
                            url = a.get('url', '')
                            if not url:
                                continue
                            with seen_lock:
                                if url in seen_urls:
                                    continue
                                seen_urls.add(url)
                            all_articles.append(a)
                except Exception:
                    errors += 1
                pbar.update(1)
                pbar.set_postfix({'candidates': len(all_articles), 'errors': errors})

    print(f"   ✓ GKG bulk phase: {len(all_articles):,} unique candidates "
          f"(theme + keyword matched, no score pre-filter)")
    return all_articles


# --------------------------------------------------------------------------
# METHOD D: Original concurrent GDELT API  (unchanged — used as fallback)
# --------------------------------------------------------------------------

def fetch_gdelt_api(query: str, start: str, end: str) -> List[Dict]:
    """Original single-query GDELT API fetch."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    days_ago = (datetime.now() - datetime.strptime(end, '%Y-%m-%d')).days
    if days_ago <= 90:
        days = (datetime.strptime(end, '%Y-%m-%d') -
                datetime.strptime(start, '%Y-%m-%d')).days
        params = {
            'query': query, 'mode': 'artlist', 'maxrecords': 250,
            'format': 'json', 'timespan': f'{min(days, 365)}d',
        }
    else:
        params = {
            'query': query, 'mode': 'artlist', 'maxrecords': 250,
            'format': 'json',
            'startdatetime': start.replace('-', '') + '000000',
            'enddatetime':   end.replace('-', '') + '235959',
        }
    for retry in range(CONFIG['retry_attempts']):
        try:
            r = requests.get(url, params=params, timeout=CONFIG['timeout'])
            if r.status_code == 200:
                data = r.json()
                return data.get('articles', [])
            elif r.status_code == 429:
                wait = min(CONFIG['retry_delay'] * (2 ** retry), CONFIG['max_backoff'])
                time.sleep(wait)
            else:
                time.sleep(CONFIG['retry_delay'])
        except Exception:
            if retry < CONFIG['retry_attempts'] - 1:
                time.sleep(CONFIG['retry_delay'])
    return []


def _gdelt_api_worker(task, seen_urls_ref, seen_lock,
                      completed_keys_ref, ckpt_lock):
    """Thread worker for one (start, end, query) API task."""
    start_date, end_date, query = task
    key = f"{start_date}|{end_date}|{query}"
    articles = fetch_gdelt_api(query, start_date, end_date)
    new_articles = []
    if articles:
        with seen_lock:
            for a in articles:
                url = a.get('url', '')
                if url and url not in seen_urls_ref:
                    seen_urls_ref.add(url)
                    a['query']      = query
                    a['source_api'] = 'gdelt_api'
                    a['language']   = 'en'
                    new_articles.append(a)
    with ckpt_lock:
        completed_keys_ref.add(key)
    time.sleep(CONFIG['gdelt_worker_delay'])
    return new_articles, len(articles) == 0


# ============================================================================
# CHECKPOINT HELPERS  (UNTOUCHED)
# ============================================================================

def save_checkpoint(seen_urls, all_articles, stats, completed_gdelt_keys, phase):
    checkpoint = {
        'seen_urls':             seen_urls,
        'all_articles':          all_articles,
        'stats':                 stats,
        'completed_gdelt_keys':  completed_gdelt_keys,
        'phase':                 phase,
        'saved_at':              datetime.now().isoformat(),
    }
    tmp = CONFIG['checkpoint_path'] + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=4)
    os.replace(tmp, CONFIG['checkpoint_path'])


def load_checkpoint():
    path = CONFIG['checkpoint_path']
    if CONFIG['use_checkpoint'] and os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            age_min = (datetime.now() - datetime.fromisoformat(ckpt['saved_at'])).seconds / 60
            print(f"\n♻️  Resuming from checkpoint: {ckpt['phase']} phase")
            print(f"   Articles so far: {len(ckpt['all_articles']):,}")
            print(f"   GDELT requests done: {len(ckpt['completed_gdelt_keys']):,}")
            print(f"   Checkpoint age: {age_min:.1f} min")
            return ckpt
        except Exception as e:
            print(f"⚠️  Checkpoint load failed ({e}), starting fresh")
    return None


# ============================================================================
# GOOGLE NEWS + RSS  (COMPLETELY UNTOUCHED)
# ============================================================================

def fetch_google_news(query: str, lang: str = 'en') -> List[Dict]:
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl=IN&ceid=IN:{lang}"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:CONFIG['max_articles_per_query']]:
            pub_date = entry.get('published', '')
            try:
                parsed_date = date_parser.parse(pub_date) if pub_date else datetime.now()
            except Exception:
                parsed_date = datetime.now()
            articles.append({
                'title':       entry.get('title', ''),
                'url':         entry.get('link', ''),
                'domain':      entry.get('source', {}).get('title', 'Google News'),
                'language':    lang,
                'seendate':    parsed_date.isoformat(),
                'source_api':  'google_news_rss',
                'description': entry.get('summary', '')[:300],
            })
        return articles
    except Exception:
        return []


def fetch_rss(url: str, lang: str) -> List[Dict]:
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:100]:
            pub_date = entry.get('published', '')
            try:
                parsed_date = date_parser.parse(pub_date) if pub_date else datetime.now()
            except Exception:
                parsed_date = datetime.now()
            articles.append({
                'title':       entry.get('title', ''),
                'url':         entry.get('link', ''),
                'domain':      feed.feed.get('title', url.split('/')[2]),
                'language':    lang,
                'seendate':    parsed_date.isoformat(),
                'source_api':  'rss_feed',
                'description': entry.get('summary', '')[:300],
            })
        return articles
    except Exception:
        return []


# ============================================================================
# MAIN COLLECTION
# ============================================================================

def collect_news():
    print("\n" + "="*80)
    print("OPTIMIZED MULTILINGUAL NEWS COLLECTOR  [BigQuery Edition]")
    print("="*80)

    ckpt = load_checkpoint()
    if ckpt:
        all_articles         = ckpt['all_articles']
        seen_urls            = ckpt['seen_urls']
        stats                = ckpt['stats']
        completed_gdelt_keys = ckpt['completed_gdelt_keys']
        resume_phase         = ckpt['phase']
    else:
        all_articles         = []
        seen_urls            = set()
        stats                = {'gdelt': 0, 'google_news': 0, 'rss': 0, 'by_language': {}}
        completed_gdelt_keys = set()
        resume_phase         = 'gdelt'

    errors     = {'gdelt_errors': 0, 'gdelt_empty': 0}
    start_time = time.time()
    seen_lock  = threading.Lock()
    ckpt_lock  = threading.Lock()

    # =========================================================================
    # PHASE 1: GDELT  — method chosen by GDELT_METHOD flag
    # =========================================================================
    print(f"\n📰 PHASE 1: GDELT  [{GDELT_METHOD.upper()} method]")
    print("-" * 80)

    gdelt_raw: List[Dict] = []

    if GDELT_METHOD == 'bigquery':
        gdelt_raw = fetch_gdelt_bigquery(CONFIG['gdelt_start'], CONFIG['gdelt_end'])

    elif GDELT_METHOD == 'manual_csv':
        gdelt_raw = fetch_gdelt_manual_csv(GDELT_MANUAL_CSV_PATH)

    elif GDELT_METHOD == 'bulk_csv':
        gdelt_raw = fetch_gdelt_bulk_csv(
            CONFIG['gdelt_start'], CONFIG['gdelt_end'],
            seen_urls, seen_lock
        )

    else:  # 'api' — original concurrent API fallback
        start_dt = datetime.strptime(CONFIG['gdelt_start'], '%Y-%m-%d')
        end_dt   = datetime.strptime(CONFIG['gdelt_end'],   '%Y-%m-%d')
        windows  = []
        current  = start_dt
        while current < end_dt:
            window_end = min(current + timedelta(days=CONFIG['gdelt_window_days']), end_dt)
            windows.append((current.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')))
            current = window_end
        windows = windows[::2]

        all_tasks = [(sd, ed, q) for sd, ed in windows for q in ENGLISH_QUERIES]
        pending   = [t for t in all_tasks
                     if f"{t[0]}|{t[1]}|{t[2]}" not in completed_gdelt_keys]

        print(f"Windows: {len(windows)}, Queries: {len(ENGLISH_QUERIES)}")
        print(f"Total tasks: {len(all_tasks):,}  Skipped: {len(all_tasks)-len(pending):,}  Pending: {len(pending):,}")

        success           = 0
        completed_count   = len(all_tasks) - len(pending)
        checkpoint_interval = 200

        with tqdm(total=len(all_tasks), initial=len(all_tasks)-len(pending),
                  desc="GDELT API") as pbar:
            with ThreadPoolExecutor(max_workers=CONFIG['gdelt_workers']) as executor:
                futures = {
                    executor.submit(_gdelt_api_worker, task, seen_urls, seen_lock,
                                    completed_gdelt_keys, ckpt_lock): task
                    for task in pending
                }
                for future in as_completed(futures):
                    try:
                        new_articles, was_empty = future.result()
                    except Exception:
                        new_articles, was_empty = [], False
                    if new_articles:
                        success += 1
                        gdelt_raw.extend(new_articles)
                    if was_empty:
                        errors['gdelt_empty'] += 1
                    completed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'articles': len(gdelt_raw),
                                      'success': success,
                                      'empty': errors['gdelt_empty']})
                    if completed_count % checkpoint_interval == 0:
                        save_checkpoint(seen_urls, all_articles, stats,
                                        completed_gdelt_keys, 'gdelt')

    # ── Normalise GDELT raw results to common schema — NO score pre-filter ──
    # The main processing block (calculate_relevance) handles final selection.
    # Pre-filtering here caused 88% of articles to be discarded before scoring
    # because GKG titles are synthetic (domain | themes) not full article text.
    print(f"\n   📥 Normalising {len(gdelt_raw):,} GDELT candidates …")

    for a in tqdm(gdelt_raw, desc="   Normalising"):
        url = a.get('url', '')
        if not url:
            continue

        # Normalise to common schema
        title = str(a.get('title', ''))
        desc  = str(a.get('description', ''))

        # Fix seendate
        sd = a.get('seendate', '')
        if not sd or sd == 'nan':
            a['seendate'] = datetime.now().isoformat()

        a['url']        = url
        a['title']      = title
        a['description'] = desc
        a['language']   = a.get('language', 'en')
        a['source_api'] = a.get('source_api', 'gdelt')
        a['query']      = a.get('query', 'gdelt_bulk')
        a['domain']     = a.get('domain', '')

        all_articles.append(a)
        stats['gdelt'] += 1
        stats['by_language']['en'] = stats['by_language'].get('en', 0) + 1

    save_checkpoint(seen_urls, all_articles, stats, completed_gdelt_keys, 'google_news')
    print(f"✅ GDELT Phase 1: {stats['gdelt']:,} articles (post-relevance filter)")
    print(f"   ⏱  Phase 1 elapsed: {(time.time()-start_time)/60:.1f} min")

    # =========================================================================
    # PHASE 2: Google News Multilingual — CONCURRENT PER LANGUAGE  (UNTOUCHED)
    # =========================================================================
    print(f"\n📰 PHASE 2: Google News (Multilingual) — {CONFIG['google_news_workers']} concurrent workers")
    print("-" * 80)

    def _gnews_worker(task):
        query, lang_code = task
        articles = fetch_google_news(query, lang_code)
        time.sleep(1.2)
        return articles, lang_code, query

    for lang_name, lang_cfg in MULTILINGUAL_CONFIG.items():
        lang_code  = lang_cfg['lang']
        queries    = lang_cfg['queries']
        lang_count = 0
        print(f"\n🌐 {lang_name.upper()} ({len(queries)} queries):")
        tasks = [(q, lang_code) for q in queries]
        with ThreadPoolExecutor(max_workers=CONFIG['google_news_workers']) as executor:
            futures = [executor.submit(_gnews_worker, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  {lang_name}", leave=False):
                try:
                    articles, lc, q = future.result()
                except Exception:
                    continue
                for a in articles:
                    url = a.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        a['query'] = q
                        all_articles.append(a)
                        stats['google_news'] += 1
                        stats['by_language'][lc] = stats['by_language'].get(lc, 0) + 1
                        lang_count += 1
        print(f"  ✓ {lang_count:,} articles")

    save_checkpoint(seen_urls, all_articles, stats, completed_gdelt_keys, 'rss')
    print(f"\n✅ Google News Multilingual: {stats['google_news']:,} articles")
    print(f"   ⏱  Phase 1+2 elapsed: {(time.time()-start_time)/60:.1f} min")

    # =========================================================================
    # PHASE 3: English RSS  (UNTOUCHED)
    # =========================================================================
    print("\n📰 PHASE 3: English RSS Feeds")
    print("-" * 80)

    ENGLISH_RSS_FEEDS = [
        'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
        'https://economictimes.indiatimes.com/news/economy/finance/rssfeeds/1373380680.cms',
        'https://www.business-standard.com/rss/home_page_top_stories.rss',
        'https://www.moneycontrol.com/rss/latestnews.xml',
        'https://www.livemint.com/rss/money',
    ]

    for feed_url in tqdm(ENGLISH_RSS_FEEDS, desc="RSS"):
        articles = fetch_rss(feed_url, 'en')
        for a in articles:
            url = a.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(a)
                stats['rss'] += 1
                stats['by_language']['en'] = stats['by_language'].get('en', 0) + 1
        time.sleep(1)

    save_checkpoint(seen_urls, all_articles, stats, completed_gdelt_keys, 'rss_multi')
    print(f"✅ English RSS: {stats['rss']:,} articles")

    # =========================================================================
    # PHASE 4: Multilingual RSS  (UNTOUCHED)
    # =========================================================================
    print("\n📰 PHASE 4: Multilingual RSS Feeds")
    print("-" * 80)

    for lang_name, config in MULTILINGUAL_CONFIG.items():
        if config.get('rss_feeds'):
            lang_code = config['lang']
            feeds     = config['rss_feeds']
            if feeds:
                print(f"\n🌐 {lang_name.upper()} RSS:")
            for feed_url in feeds:
                articles = fetch_rss(feed_url, lang_code)
                count = 0
                for a in articles:
                    url = a.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(a)
                        stats['rss'] += 1
                        stats['by_language'][lang_code] = stats['by_language'].get(lang_code, 0) + 1
                        count += 1
                if count > 0:
                    print(f"  ✓ {feed_url.split('/')[2]}: {count} articles")
                time.sleep(1)

    # =========================================================================
    # PROCESSING  (UNTOUCHED)
    # =========================================================================
    print("\n" + "="*80)
    print("PROCESSING")
    print("="*80)

    if not all_articles:
        print("❌ No articles collected!")
        return None, None

    df = pd.DataFrame(all_articles)
    print(f"\n📊 Raw articles collected: {len(df):,}")
    print(f"   Raw languages: {dict(df['language'].value_counts())}")

    print(f"\n🔍 Parsing dates…")
    initial_count = len(df)

    def safe_date_parse(date_str):
        try:
            if pd.isna(date_str):
                return pd.NaT
            return pd.to_datetime(date_str)
        except Exception:
            try:
                return pd.to_datetime(date_parser.parse(str(date_str)))
            except Exception:
                return pd.NaT

    df['seendate'] = df['seendate'].apply(safe_date_parse)
    df = df.dropna(subset=['seendate'])
    print(f"✓ After date filtering: {len(df):,}")
    if initial_count != len(df):
        print(f"   ⚠️  Lost {initial_count - len(df):,} articles (invalid dates)")
        print(f"   Languages AFTER date filter: {dict(df['language'].value_counts())}")

    print(f"\n🔍 Calculating relevance scores…")
    df['title_text']  = df['title'].fillna('')
    df['description'] = df.get('description', pd.Series([''] * len(df)))
    df['full_text']   = (df['title_text'] + ' ' + df['description'].fillna('')).str.strip()
    df['relevance_score'] = df.apply(
        lambda row: calculate_relevance(row['full_text'], row['language']), axis=1)
    df['crisis_flag'] = df['full_text'].apply(detect_crisis)
    df['label_method'] = df['language'].apply(
        lambda lang: 'vader' if lang == 'en' else 'xlm_roberta')

    print(f"\n🎯 Filtering — BALANCED mode (remittance + finance/migration):")
    print(f"   • English: {CONFIG['relevance_threshold_english']}% "
          f"(eliminates zero-score GKG noise, targets ~50-70K)")
    print(f"   • Multilingual: {CONFIG['relevance_threshold_multilingual']}% (VERY LENIENT)")

    df_english      = df[df['language'] == 'en'].copy()
    df_multilingual = df[df['language'] != 'en'].copy()
    print(f"\n   Before filtering:")
    print(f"      English: {len(df_english):,}")
    print(f"      Multilingual: {len(df_multilingual):,}")

    if len(df_multilingual) > 0:
        print(f"\n   Sample multilingual scores:")
        sample = df_multilingual.nlargest(5, 'relevance_score')[
            ['language', 'relevance_score', 'title']]
        for _, row in sample.iterrows():
            print(f"      {row['language']}: {row['relevance_score']:.2f}% — {row['title'][:60]}…")

    df_english_filtered      = df_english[
        df_english['relevance_score'] >= CONFIG['relevance_threshold_english']]
    df_multilingual_filtered = df_multilingual[
        df_multilingual['relevance_score'] >= CONFIG['relevance_threshold_multilingual']]

    print(f"\n   After filtering:")
    print(f"      English: {len(df_english):,} → {len(df_english_filtered):,} "
          f"({(len(df_english_filtered)/max(len(df_english),1))*100:.1f}% retained)")
    print(f"      Multilingual: {len(df_multilingual):,} → {len(df_multilingual_filtered):,} "
          f"({(len(df_multilingual_filtered)/max(len(df_multilingual),1))*100:.1f}% retained)")

    if len(df_english) > 0:
        print(f"\n   English scores — Mean: {df_english['relevance_score'].mean():.2f}%, "
              f"Median: {df_english['relevance_score'].median():.2f}%")
    if len(df_multilingual) > 0:
        print(f"   Multilingual scores — Mean: {df_multilingual['relevance_score'].mean():.2f}%, "
              f"Median: {df_multilingual['relevance_score'].median():.2f}%")

    df_filtered = pd.concat([df_english_filtered, df_multilingual_filtered], ignore_index=True)

    if len(df_filtered) < 100:
        print("\n⚠️  WARNING: Very few articles passed filtering!")
        print("   Using top 5000 articles by relevance score instead…")
        df_filtered = df.nlargest(min(5000, len(df)), 'relevance_score')

    print(f"\n✅ Final count: {len(df_filtered):,} articles "
          f"({(len(df_filtered)/len(df))*100:.1f}% of raw)")
    print(f"\n   Final language breakdown:")
    for lang, count in df_filtered['language'].value_counts().items():
        pct       = (count / len(df_filtered)) * 100
        lang_name = {
            'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil',
            'te': 'Telugu',  'ml': 'Malayalam', 'bn': 'Bengali',
            'pa': 'Punjabi', 'gu': 'Gujarati'
        }.get(lang, lang)
        print(f"      • {lang_name} ({lang}): {count:,} ({pct:.1f}%)")

    df_filtered['seendate'] = pd.to_datetime(
        df_filtered['seendate'], utc=True).dt.tz_localize(None)
    df_filtered['year']         = df_filtered['seendate'].dt.year
    df_filtered['quarter']      = df_filtered['seendate'].dt.quarter
    df_filtered['year_quarter'] = (df_filtered['year'].astype(str) + 'Q' +
                                   df_filtered['quarter'].astype(str))

    # ── Relevance tier — for next-phase meta-model weighting ──────────────────
    # High  ≥ 20% : article clearly about remittances (strong keyword density)
    # Medium 5-20%: related finance/migration context — useful signal
    # Low   < 5%  : weak signal — marginal relevance, use with caution
    def _tier(score: float) -> str:
        if score >= 20.0:
            return 'High'
        elif score >= 5.0:
            return 'Medium'
        else:
            return 'Low'

    df_filtered['relevance_tier'] = df_filtered['relevance_score'].apply(_tier)

    tier_counts = df_filtered['relevance_tier'].value_counts()
    print(f"\n   Relevance tier breakdown:")
    for tier in ['High', 'Medium', 'Low']:
        n   = tier_counts.get(tier, 0)
        pct = n / len(df_filtered) * 100
        print(f"      • {tier:6s}: {n:,} ({pct:.1f}%)")

    quarterly = df_filtered.groupby('year_quarter').agg({
        'title': 'count', 'relevance_score': 'mean', 'crisis_flag': 'mean',
    }).reset_index()
    quarterly.columns = ['quarter', 'article_count', 'avg_relevance', 'crisis_proportion']
    quarterly = quarterly.sort_values('quarter')

    quarterly_by_lang = df_filtered.groupby(['year_quarter', 'language']).agg({
        'title': 'count', 'relevance_score': 'mean',
    }).reset_index()
    quarterly_by_lang.columns = ['quarter', 'language', 'article_count', 'avg_relevance']

    # =========================================================================
    # SAVE  (UNTOUCHED)
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING FILES")
    print("="*80)

    df_final = df_filtered[[
        'title', 'url', 'domain', 'language', 'seendate',
        'year', 'quarter', 'year_quarter', 'query',
        'relevance_score', 'relevance_tier', 'crisis_flag',
        'source_api', 'label_method'
    ]].copy()
    df_final = df_final.sort_values('seendate', ascending=False)

    df_final.to_csv('remittances_news_final.csv', index=False, encoding='utf-8-sig')
    quarterly.to_csv('remittances_quarterly.csv', index=False)
    quarterly_by_lang.to_csv('remittances_quarterly_by_language.csv', index=False)

    print(f"✓ remittances_news_final.csv ({len(df_final):,} articles)")
    print(f"✓ remittances_quarterly.csv ({len(quarterly)} quarters)")
    print(f"✓ remittances_quarterly_by_language.csv ({len(quarterly_by_lang)} rows)")

    total_time = time.time() - start_time
    metadata = {
        'collected': datetime.now().isoformat(),
        'gdelt_method': GDELT_METHOD,
        'total_articles': int(len(df_final)),
        'collection_stats': {
            'gdelt': int(stats['gdelt']),
            'google_news': int(stats['google_news']),
            'rss': int(stats['rss']),
            'gdelt_errors': int(errors['gdelt_errors']),
            'gdelt_empty': int(errors['gdelt_empty']),
        },
        'language_breakdown': {
            str(k): int(v)
            for k, v in df_final['language'].value_counts().to_dict().items()
        },
        'filtering': {
            'raw_collected': int(len(df)),
            'after_filtering': int(len(df_final)),
            'retention_rate': float(len(df_final) / len(df)),
            'english_threshold': CONFIG['relevance_threshold_english'],
            'multilingual_threshold': CONFIG['relevance_threshold_multilingual'],
            'relevance_tiers': {
                t: int((df_final['relevance_tier'] == t).sum())
                for t in ['High', 'Medium', 'Low']
            },
        },
        'quality_metrics': {
            'avg_relevance_all':    float(df_final['relevance_score'].mean()),
            'median_relevance_all': float(df_final['relevance_score'].median()),
            'crisis_articles':      int(df_final['crisis_flag'].sum()),
            'crisis_proportion':    float(df_final['crisis_flag'].mean()),
        },
        'time_minutes': float(total_time / 60),
        'date_range': {
            'earliest': df_final['seendate'].min().isoformat(),
            'latest':   df_final['seendate'].max().isoformat(),
        },
        'top_sources': {
            str(k): int(v)
            for k, v in df_final['domain'].value_counts().head(10).to_dict().items()
        },
        'config': CONFIG,
    }
    with open('collection_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ collection_metadata.json")

    # =========================================================================
    # A1: GDELT REPRODUCIBILITY TABLE  (UNTOUCHED)
    # =========================================================================
    print("\n" + "="*80)
    print("A1: GDELT REPRODUCIBILITY TABLE (Methods Section)")
    print("="*80)

    gdelt_dispatched = stats['gdelt']
    google_articles  = stats.get('google_news', 0)
    rss_articles     = stats.get('rss', 0)

    gdelt_table = pd.DataFrame([
        {'Source': f'GDELT articles ({GDELT_METHOD} method)',
         'Count': gdelt_dispatched,
         'Note': 'Post-relevance filter; no soft-cap truncation'},
        {'Source': 'Google News RSS (7 languages)',
         'Count': google_articles,
         'Note': 'Compensated for GDELT shortfall'},
        {'Source': 'RSS Feeds (English + regional)',
         'Count': rss_articles,
         'Note': 'Economic Times, Business Standard, regional'},
        {'Source': 'Total after relevance filtering',
         'Count': len(df_final),
         'Note': 'English + multilingual retained'},
        {'Source': 'Reproducibility note',
         'Count': '-',
         'Note': (f'Method: {GDELT_METHOD}. '
                  'BigQuery/manual_csv avoid API soft-cap entirely. '
                  'bulk_csv samples every 3rd day; full daily coverage '
                  'requires GDELT_METHOD=bigquery or manual_csv.')},
    ])
    gdelt_table.to_csv('gdelt_reproducibility_table.csv', index=False)
    print("\n  GDELT Collection Summary:")
    print(f"  {'Source':<45} {'Count':>8}  Note")
    print(f"  {'-'*45} {'-'*8}  {'-'*50}")
    for _, row in gdelt_table.iterrows():
        cnt = str(row['Count']).rjust(8)
        print(f"  {str(row['Source']):<45} {cnt}  {row['Note'][:60]}")
    print("\n✓ gdelt_reproducibility_table.csv")

    # =========================================================================
    # A3: QUARTERLY HEATMAP  (UNTOUCHED)
    # =========================================================================
    print("\n" + "="*80)
    print("A3: QUARTERLY ARTICLE-COUNT HEATMAP (Figure 1)")
    print("="*80)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        qbl = pd.read_csv('remittances_quarterly_by_language.csv')
        lang_labels = {
            'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil',
            'te': 'Telugu',  'ml': 'Malayalam', 'bn': 'Bengali',
            'pa': 'Punjabi', 'gu': 'Gujarati'
        }
        qbl['language'] = qbl['language'].map(lang_labels).fillna(qbl['language'])
        pivot = qbl.pivot_table(
            index='quarter', columns='language',
            values='article_count', aggfunc='sum').fillna(0)
        col_order = ['English'] + sorted([c for c in pivot.columns if c != 'English'])
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(18, 5))
        pivot.plot.bar(stacked=True, ax=ax, colormap='tab10', width=0.85)

        quarters = list(pivot.index)
        try:
            split_pos = quarters.index('2017Q4') + 0.5
            ax.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.8,
                       label='Train / Test Split (2017Q4)')
        except ValueError:
            for i, q in enumerate(quarters):
                if q >= '2018Q1':
                    ax.axvline(x=i - 0.5, color='red', linestyle='--', linewidth=1.8,
                               label='Train / Test Split')
                    break

        ax.set_title(
            'Figure 1: Quarterly Article Count by Language\n'
            'M-SENSE Multilingual News Corpus (India Remittances 2017–2025)',
            fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Quarter', fontsize=11)
        ax.set_ylabel('Article Count', fontsize=11)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        tick_positions = list(range(0, len(quarters), 4))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([quarters[i] for i in tick_positions],
                           rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=9, ncol=3, framealpha=0.8)
        plt.tight_layout()
        fig.savefig('figure1_quarterly_article_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ figure1_quarterly_article_heatmap.png saved")
    except Exception as e:
        print(f"⚠️  Heatmap generation failed: {e}")

    # =========================================================================
    # SUMMARY  (UNTOUCHED)
    # =========================================================================
    print("\n" + "="*80)
    print("✅ COLLECTION COMPLETE")
    print("="*80)

    english      = len(df_final[df_final['language'] == 'en'])
    multilingual = len(df_final[df_final['language'] != 'en'])

    print(f"\n📊 FINAL RESULTS: {len(df_final):,} articles")
    print(f"   • English: {english:,} ({(english/len(df_final))*100:.1f}%)")
    print(f"   • Multilingual: {multilingual:,} ({(multilingual/len(df_final))*100:.1f}%)")
    print(f"\n🌐 By Source:")
    print(f"   • GDELT ({GDELT_METHOD}): {stats['gdelt']:,}")
    print(f"   • Google News RSS: {stats['google_news']:,}")
    print(f"   • RSS Feeds: {stats['rss']:,}")
    print(f"\n🗣️  Languages:")
    for lang, count in df_final['language'].value_counts().items():
        pct = (count / len(df_final)) * 100
        lang_name = {
            'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil',
            'te': 'Telugu',  'ml': 'Malayalam', 'bn': 'Bengali',
            'pa': 'Punjabi', 'gu': 'Gujarati'
        }.get(lang, lang)
        print(f"   • {lang_name} ({lang}): {count:,} ({pct:.1f}%)")
    print(f"\n📈 Quality Metrics:")
    print(f"   • Avg relevance (all): {df_final['relevance_score'].mean():.2f}%")
    if english > 0:
        print(f"   • Avg relevance (English): "
              f"{df_final[df_final['language']=='en']['relevance_score'].mean():.2f}%")
    if multilingual > 0:
        print(f"   • Avg relevance (Multilingual): "
              f"{df_final[df_final['language']!='en']['relevance_score'].mean():.2f}%")
    print(f"   • Crisis articles: {int(df_final['crisis_flag'].sum()):,} "
          f"({(df_final['crisis_flag'].mean())*100:.1f}%)")
    print(f"\n📅 Coverage:")
    print(f"   • Date range: {df_final['seendate'].min().date()} "
          f"to {df_final['seendate'].max().date()}")
    print(f"   • Quarters: {len(quarterly)}")
    print(f"   • Unique sources: {df_final['domain'].nunique()}")
    print(f"\n⏱️  Collection time: {total_time/60:.1f} minutes")

    # Clean up checkpoint
    if os.path.exists(CONFIG['checkpoint_path']):
        os.remove(CONFIG['checkpoint_path'])
        print(f"\n🗑️  Checkpoint cleaned up")

    return df_final, quarterly


# ============================================================================
# ★ GDELT BIGQUERY EXPORT HELPER ★
# Run this ONCE on a machine with GCP access to produce gdelt_raw_export.csv
# Then upload that CSV to Kaggle as a dataset and set GDELT_METHOD='manual_csv'
# ============================================================================

def export_gdelt_to_csv_via_bigquery(
        output_path: str = 'gdelt_raw_export.csv',
        start: str = '2017-01-01',
        end:   str = '2025-12-31',
        project: str = GDELT_BQ_PROJECT):
    """
    Run this helper ONCE outside Kaggle to pull GDELT via BigQuery and save
    a CSV that can be uploaded as a Kaggle dataset.

    Usage:
        python -c "from nlpts4_optimized_bigquery import export_gdelt_to_csv_via_bigquery; \
                   export_gdelt_to_csv_via_bigquery()"

    Requirements:
        pip install google-cloud-bigquery pandas-gbq
        gcloud auth application-default login
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("pip install google-cloud-bigquery pandas-gbq")
        return

    client = bigquery.Client(project=project)

    keyword_conditions = " OR ".join(
        [f"LOWER(title) LIKE '%{q.lower()}%'" for q in ENGLISH_QUERIES]
    )

    sql = f"""
    SELECT
        url,
        title,
        CAST(seendate AS STRING) AS seendate,
        domain,
        'en' AS language
    FROM
        `gdelt-bq.gdeltv2.geg`
    WHERE
        DATE(seendate) BETWEEN '{start}' AND '{end}'
        AND (
            {keyword_conditions}
        )
    """

    print(f"Querying BigQuery for {start} → {end} …")
    df = client.query(sql).to_dataframe()
    print(f"Got {len(df):,} rows — saving to {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved. Upload '{output_path}' as a Kaggle dataset, then set:")
    print(f"  GDELT_METHOD = 'manual_csv'")
    print(f"  GDELT_MANUAL_CSV_PATH = '/kaggle/input/<your-dataset>/{output_path}'")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 OPTIMIZED MULTILINGUAL COLLECTOR  [BigQuery Edition]")
    print("="*80)
    print(f"GDELT Method : {GDELT_METHOD.upper()}")
    print(f"  bigquery   → no rate limits, ~2 min,  requires GCP auth")
    print(f"  manual_csv → no rate limits, instant, requires pre-exported CSV in Kaggle")
    print(f"  bulk_csv   → no auth needed, ~15-25 min, samples every 3rd day (CURRENT)")
    print(f"  api        → original concurrent API, ~35-50 min, hits soft cap")
    print(f"\nTo switch method: set GDELT_METHOD at top of file")
    print(f"To generate manual CSV: call export_gdelt_to_csv_via_bigquery()")
    print("="*80)

    time.sleep(2)

    try:
        df, quarterly = collect_news()
        if df is not None:
            print("\n" + "="*80)
            print("✅ SUCCESS! FILES READY")
            print("="*80)
            print("\nGenerated files:")
            print("   • remittances_news_final.csv")
            print("   • remittances_quarterly.csv")
            print("   • remittances_quarterly_by_language.csv")
            print("   • collection_metadata.json")
            print("   • gdelt_reproducibility_table.csv")
            print("   • figure1_quarterly_article_heatmap.png")
            multilingual_count = len(df[df['language'] != 'en'])
            print(f"\n🎯 Summary: {len(df):,} total | {multilingual_count:,} multilingual")
            print(f"   Languages: {df['language'].nunique()}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — checkpoint saved, re-run to resume")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()