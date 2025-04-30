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
  - **Dark/light toggle** in sidebar
  - Wide-mode layout via `.streamlit/config.toml`

- **Multimodal stub** (`v1.1`):
  - Preprocessing & model templates for text/audio/image fusion

---

## 🚀 Quickstart

```bash
git clone https://github.com/Chukwuemekaeze/opinion_mining_project.git
cd opinion_mining_project
```

# 1. Create & activate venv

python -m venv .venv

## Windows

.\.venv\Scripts\activate

## macOS/Linux

source .venv/bin/activate

# 2. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

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
├── .streamlit/
│   └── config.toml              # wide-mode & theme settings
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
│   ├── dashboard.py             # Streamlit app with dark/light toggle
│   ├── data_collection.py       # tweet/news/web ingestion
│   ├── data_preprocessing.py    # text/audio/image cleaning
│   ├── database.py              # SQLite helpers
│   ├── evaluation.py            # metrics & confusion matrix
│   ├── main.py                  # CLI: live / eval / multimodal
│   ├── pidgin_lexicon.csv       # Pidgin sentiment lexicon
│   ├── sentiment_analysis.py    # VADER + Pidgin lexicon
│   └── text_labels.csv          # 60+ hand-labeled Edo-2024 sentences
├── README.md
└── requirements.txt             # Python dependencies
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

2. Ensure `.streamlit/config.toml` is present.

3. In Streamlit Cloud, connect the repo, set main file to `src/dashboard.py`.

4. Add any API keys under **Settings → Secrets**.

5. Click **Deploy**.

---

## 📝 Version History

- **v1.1** (current)
  - Added text-eval mode & metrics
  - Integrated Pidgin lexicon into VADER
  - Enhanced dashboard (time-series, source breakdown, evaluation, dark/light toggle)
  - Added `.streamlit/config.toml` for wide-mode & theme
  - Prepared multimodal model stubs
- **v1.0**
  - Core VADER live pipeline
  - Basic dashboard & DB storage

---

## 👥 Collaborators

- **Lead Dev**: Chukwuemekaeze
- **Contributor**: Kachi Osuji
- Members of **Group 4**, **CSC 421**, Nnamdi Azikiwe University
