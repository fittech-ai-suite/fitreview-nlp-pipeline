# FitReview NLP Pipeline

Sentiment analysis pipeline for fitness app reviews — from raw Google Play data through fine-tuned DistilBERT to a REST API with aspect-level insights.

## Results

| Model | Accuracy | Macro-F1 | neg F1 | pos F1 |
|---|---|---|---|---|
| Majority class baseline | 57.8% | 0.366 | 0.732 | 0.000 |
| TF-IDF + Logistic Regression | 86.9% | 0.864 | 0.891 | 0.838 |
| TF-IDF + LinearSVC | 88.3% | 0.879 | 0.903 | 0.854 |
| DistilBERT 3-class (neg/neu/pos) | 77.0% | ~0.70 | — | — |
| **DistilBERT binary (neg/pos)** | **91.2%** | **0.909** | **0.927** | **0.891** |

The 3-class model's neutral class (3-star reviews) had F1 = 0.37 — genuinely ambiguous text that even humans label inconsistently. Dropping neutral and reframing as binary lifted accuracy by 14 points and resolved the class imbalance problem cleanly.

## Key Findings

Aggregating binary predictions across 5,700 reviews surfaces concrete product insights:

**Aspect pain points** (% positive among reviews mentioning each aspect):

| Aspect | % Positive | n |
|---|---|---|
| Bugs & Crashes | 19% | 707 |
| Customer Support | 30% | 380 |
| Ads | 34% | 282 |
| Accuracy | 34% | 782 |
| Price / Value | 39% | 1,398 |
| Features | 36% | 1,961 |
| UI / UX | 53% | 666 |

**By app**: Fitbit (2.4% positive) and MyFitnessPal (8.2%) are the most negative despite their market position. FatSecret (82.6%) and Cronometer (74.7%) are the most positive.

Visualisations in [`results/analysis/`](results/analysis/).

## Data

6,549 reviews scraped from Google Play across 23 fitness apps and 12 categories (nutrition, cycling, running, yoga, strength, sleep, etc.). Each review is labelled with a 3-class sentiment (`negative`/`neutral`/`positive`) derived from star rating, plus 7 binary aspect flags extracted during EDA.

Raw scraping: `01_eda_and_clean.py`

## Project Structure

```
src/
  training/
    train.py              # 3-class DistilBERT fine-tuning
    train_binary.py       # binary fine-tuning (neg/pos only)
  inference/
    predictor.py          # SentimentPredictor class, single + batch
  evaluation/
    baselines.py          # TF-IDF baselines for ablation
    evaluate_binary.py    # confusion matrix, calibration, error analysis
  analysis/
    aspect_sentiment.py   # per-app / per-category / per-aspect breakdowns
  api/
    app.py                # FastAPI serving endpoint
results/
  evaluation/             # confusion_matrix.png, calibration.png, error_analysis.json
  analysis/               # aspect_heatmap.png, sentiment_by_app.png, ...
  baselines/              # baseline_results.json
```

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.10, Apple Silicon (MPS) and CUDA. CPU works but training is slow.

## Training

```bash
# binary model (recommended)
python src/training/train_binary.py

# 3-class model
python src/training/train.py
```

Saves checkpoints to `models/distilbert-fitness-binary/` (best epoch by macro-F1).  
Experiment tracking via MLflow: `mlflow ui`

## Evaluation

```bash
# sklearn baselines
python src/evaluation/baselines.py

# deep evaluation (confusion matrix, error analysis, calibration)
python src/evaluation/evaluate_binary.py

# aspect-level analysis
python src/analysis/aspect_sentiment.py
```

## Inference

```python
from src.inference.predictor import SentimentPredictor

model = SentimentPredictor()
model.predict("Great app, tracks everything I need")
# {'label': 'positive', 'confidence': 0.981, 'scores': {'negative': 0.019, 'positive': 0.981}}

model.predict_batch(["love it", "crashes constantly", "decent but buggy"])
```

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

```
GET  /health
POST /predict        {"text": "..."}
POST /predict/batch  {"texts": ["...", "..."]}
```

Interactive docs at `http://localhost:8000/docs`.

## Error Analysis

100 / 1,141 test samples are misclassified (91.2% correct). The hardest cases are:
- **False positives**: reviews that open positively ("great app overall") but end with a specific complaint — the model weights the positive framing more than the closing issue.
- **False negatives**: bug reports written with calm resignation ("since the October update GPS doesn't work") — no emotional signal for the model to pick up on.

Both failure modes are things a human would also struggle with. Full list in [`results/evaluation/error_analysis.json`](results/evaluation/error_analysis.json).
