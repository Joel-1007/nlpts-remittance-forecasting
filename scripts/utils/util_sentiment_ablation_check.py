import pandas as pd, json

sv = pd.read_csv('/kaggle/working/sentiment_vectors.csv')
with open('/kaggle/working/ablation_results.json') as f:
    abl = json.load(f)

print("=== Per-language macro F1 ===")
for lang, res in abl.items():
    macro = res.get('f1_macro', res.get('macro avg', {}).get('f1-score', 'N/A'))
    weighted = res.get('f1_weighted', res.get('weighted avg', {}).get('f1-score', 'N/A'))
    print(f"  {lang}: macro={macro}  weighted={weighted}")

print("\n=== Sentiment vector columns ===")
print(sv.columns.tolist())
print(sv[['sentiment_mean','positive_proportion','sentiment_mean_weighted','positive_proportion_weighted']].describe())