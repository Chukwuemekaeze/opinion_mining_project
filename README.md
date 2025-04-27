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
