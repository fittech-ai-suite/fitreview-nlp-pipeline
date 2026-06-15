"""
EDA + cleaning for the scraped fitness review data.
Checks data quality, looks at class balance and aspect mentions,
then writes out a cleaned csv + class weights for training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
import json

sns.set_style("whitegrid")
os.makedirs("notebooks/eda_outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

RAW = "data/raw/fitness_reviews.csv"
CLEAN = "data/processed/fitness_reviews_clean.csv"

df = pd.read_csv(RAW)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns\n")

# ── quality snapshot ──
print("="*60)
print("1. DATA QUALITY")
print("="*60)
print(f"Missing values:        {df.isnull().sum().sum()}")
print(f"Exact duplicate texts: {df['text'].duplicated().sum()}")

# sanity check that the labels line up with the star ratings
ct = pd.crosstab(df['rating'], df['sentiment'])
mismatches = (
    ct.loc[[1, 2], ['neutral', 'positive']].values.sum()
    + ct.loc[[3], ['negative', 'positive']].values.sum()
    + ct.loc[[4, 5], ['negative', 'neutral']].values.sum()
)
print(f"Label/rating mismatches: {mismatches}")

# ── class balance ──
print("\n" + "="*60)
print("2. CLASS BALANCE")
print("="*60)
vc = df['sentiment'].value_counts()
for s in ['negative', 'neutral', 'positive']:
    print(f"  {s:<9} {vc[s]:>5}  ({vc[s]/len(df)*100:4.1f}%)")
print(f"  imbalance (max/min): {vc.max()/vc.min():.2f}x")

# ── length features ──
df['word_count'] = df['text'].str.split().str.len()
df['char_count'] = df['text'].str.len()
df['exclaim'] = df['text'].str.count('!')
df['caps_ratio'] = df['text'].apply(
    lambda x: sum(c.isupper() for c in x) / len(x) if len(x) else 0
)

print("\n" + "="*60)
print("3. REVIEW LENGTH BY SENTIMENT")
print("="*60)
print(df.groupby('sentiment')[['word_count', 'char_count', 'exclaim', 'caps_ratio']]
      .mean().round(2).to_string())

# ── aspect keywords - what people actually mention ──
ASPECTS = {
    "price":        r'\$|\bprice|\bcost|\bexpensive|\bcheap|\bsubscription|\bpaywall|\bpremium|\bpay\b|\bfree\b',
    "bugs_crashes": r'\bcrash|\bbug|\bglitch|\bfreeze|\bbroke|\bbroken|\berror|\bnot working',
    "ads":          r'\bads\b|\badvert|\bpop.?up',
    "ui_ux":        r'\binterface|\bdesign|\beasy to use|\bintuitive|\blayout|\buser.?friendly|\bclunky',
    "features":     r'\bfeature|\bfunction|\boption|\bupdate',
    "accuracy":     r'\baccura|\bwrong|\bincorrect|\bsync',
    "support":      r'\bsupport|\bcustomer service|\brefund',
}
for name, pat in ASPECTS.items():
    df[f"asp_{name}"] = df['text'].str.contains(pat, case=False, regex=True).astype(int)

print("\n" + "="*60)
print("4. ASPECT MENTIONS BY SENTIMENT (%)")
print("="*60)
asp_cols = [f"asp_{a}" for a in ASPECTS]
asp_table = (df.groupby('sentiment')[asp_cols].mean() * 100).round(1)
asp_table.columns = list(ASPECTS.keys())
print(asp_table.T.to_string())

# ── which categories are people happiest with ──
print("\n" + "="*60)
print("5. % POSITIVE BY CATEGORY")
print("="*60)
cat_pos = (df.groupby('category')['sentiment']
           .apply(lambda s: (s == 'positive').mean() * 100)
           .sort_values(ascending=False).round(1))
print(cat_pos.to_string())

# ── top words per sentiment ──
print("\n" + "="*60)
print("6. TOP WORDS PER SENTIMENT")
print("="*60)
STOP = set("""the a an and or but in on at to for of with is it this that i my me we you have
has had be been are was were will would can could do does did not no so if as by app use used
using get got just very really good great bad like love hate one all also my im its dont
this they them their there here what when which who you your me my our""".split())

def top_words(texts, n=12):
    words = re.findall(r'\b[a-z]+\b', " ".join(texts).lower())
    words = [w for w in words if w not in STOP and len(w) > 3]
    return Counter(words).most_common(n)

for s in ['positive', 'negative', 'neutral']:
    print(f"\n{s.upper():>8}: " + ", ".join(
        f"{w}({c})" for w, c in top_words(df[df['sentiment'] == s]['text'].tolist())
    ))

# ── summary chart ──
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
vc.reindex(['negative', 'neutral', 'positive']).plot(
    kind='bar', ax=ax[0, 0], color=['#ef4444', '#f5a623', '#10b981'])
ax[0, 0].set_title('Class imbalance')
ax[0, 0].set_xlabel('')
df.boxplot(column='word_count', by='sentiment', ax=ax[0, 1])
ax[0, 1].set_title('Word count by sentiment'); ax[0, 1].set_xlabel('')
plt.suptitle('')
asp_table.T.plot(kind='bar', ax=ax[1, 0])
ax[1, 0].set_title('Aspect mentions by sentiment (%)')
ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel('')
cat_pos.plot(kind='barh', ax=ax[1, 1], color='#10b981')
ax[1, 1].set_title('% positive by category')
plt.tight_layout()
plt.savefig("notebooks/eda_outputs/eda_summary.png", dpi=120, bbox_inches='tight')
print("\nSaved chart -> notebooks/eda_outputs/eda_summary.png")

# ── cleaning ──
print("\n" + "="*60)
print("7. CLEANING")
print("="*60)
before = len(df)

df = df.drop_duplicates(subset=['text']).copy()
print(f"  dropped duplicates       -> -{before - len(df)}")

step = len(df)
df = df[df['word_count'] >= 4].copy()
print(f"  dropped < 4 word reviews -> -{step - len(df)}")

step = len(df)
def na_ratio(s):
    s = str(s)
    return sum(ord(c) > 127 for c in s) / len(s) if s else 0
df = df[df['text'].apply(na_ratio) <= 0.30].copy()
print(f"  dropped non-English      -> -{step - len(df)}")

print(f"\n  {before:,} -> {len(df):,} rows  (removed {before - len(df)})")

# class weights to feed into the weighted loss during training
counts = df['label'].value_counts().sort_index()
weights = len(df) / (len(counts) * counts)
label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
print("\n  Class weights:")
for lbl in sorted(counts.index):
    print(f"    {label_names[lbl]:<9} (label {lbl}): {weights[lbl]:.3f}")

df.to_csv(CLEAN, index=False)
with open("data/processed/class_weights.json", "w") as f:
    json.dump({str(k): round(float(v), 4) for k, v in weights.items()}, f, indent=2)
print(f"\n  Saved -> {CLEAN}")
print(f"  Saved -> data/processed/class_weights.json")
