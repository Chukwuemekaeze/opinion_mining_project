# Opinion Mining Project (v1.1)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://opinionminingproject-jma9y5rmmhrujuj5vuvf9p.streamlit.app/)

Live and multimodal sentiment-analysis system for the **2024 Edo State Governorship Election**.

---

## 🔍 Overview

- **Live pipeline** (`v1.0`):

  - Collect tweets, news headlines, or web text
  - Clean using VADER + custom Pidgin lexicon
  - Store into SQLite (`edo_election_sentiment.db`)
  - Visualize distribution

- **Evaluation mode** (`v1.1`):

  - Hand-labeled 60+ Edo-themed sentences
  - Compute accuracy, precision, recall, F1, confusion matrix

- **Streamlit dashboard** (`v1.1`):

  - Pie chart, time-series, source breakdown, word-cloud
  - Evaluation metrics & confusion matrix

- **Multimodal stub** (`v1.1`):
  - Preprocessing & model templates for text/audio/image fusion

---

## ⚙️ Usage

### 1) Live sentiment ingestion  
```bash
python src/main.py live twitter "#EdoDecides2024"
python src/main.py live news "Edo State election 2024"
python src/main.py live web https://inecnigeria.org/edo-results
```

### 2) Evaluate text-only VADER performance
```bash
python src/main.py eval --labels src/text_labels.csv
```

### 3) Multimodal training & evaluation (stub)
```bash
python src/main.py multimodal --dataset data/raw/sample_dataset.csv --test-size 0.2
```

### 4) Run Streamlit dashboard locally
```bash
streamlit run src/dashboard.py
```

---

## 📂 File Structure

```bash
OPINION_MINING_PROJECT/
├── data/
│   └── raw/
│       └── edo_election_sentiment.db
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_model.py
│   │   ├── image_model.py
│   │   ├── multimodal_fusion.py
│   │   └── text_model.py
│   ├── dashboard.py             # Streamlit visualization & evaluation
│   ├── data_collection.py       # Fetch tweets, news, web content
│   ├── data_preprocessing.py    # Clean text, extract MFCC, resize images
│   ├── database.py              # SQLite setup & insertion
│   ├── evaluation.py            # accuracy/precision/recall/F1/confusion matrix
│   ├── main.py                  # CLI: live / eval / multimodal
│   ├── pidgin_lexicon.csv       # Domain lexicon for Pidgin sentiment
│   ├── sentiment_analysis.py    # VADER + Pidgin lexicon scoring
│   └── text_labels.csv          # 60+ hand-labeled Edo-2024 sentences
├── README.md
└── requirements.txt             # All Python dependencies
```

---

## 📦 Dependencies

See `requirements.txt`. Key packages include:

```text
streamlit
pandas
numpy
matplotlib
wordcloud
nltk
librosa
opencv-python-headless
scikit-learn
transformers
torch
```

Install with:

```bash
pip install -r requirements.txt
```

---

## ☁️ Deployment

This app is live on Streamlit Community Cloud:

[Open the live dashboard](https://opinionminingproject-jma9y5rmmhrujuj5vuvf9p.streamlit.app/)

To deploy your own:

1. Push this repo to GitHub.

2. In Streamlit Cloud, connect the repo, set main file to src/dashboard.py.

3. Add any API keys under “Settings → Secrets”.

4. Click **Deploy**.

---

## 📝 Version History

- **v1.1** (current)
  - Added text-eval mode & metrics
  - Integrated Pidgin lexicon into VADER
  - Enhanced dashboard (time-series, source breakdown, evaluation)
  - Prepared multimodal model stubs
- **v1.0**
  - Core VADER live pipeline
  - Basic dashboard & DB storage

---

## 👥 Collaborators

- **Lead Dev**: Chukwuemekaeze
- **Contributor**: Kachi Osuji
- Members of **Group 4**, **CSC 421**, Nnamdi Azikiwe University
  
