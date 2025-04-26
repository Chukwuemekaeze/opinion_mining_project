import streamlit as st
import sqlite3
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Database path explicitly set
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "edo_election_sentiment.db")

# Connect explicitly to SQLite database
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sentiment_data", conn, parse_dates=["date_collected"])
    conn.close()
    return df

data = load_data()

# Dashboard Title
st.title("📊 Edo State Election, 2024: Opinion Mining Dashboard")

# Explicitly show data
st.header("📋 Sentiment Analysis Data (first 50 rows)")
st.dataframe(data.head(50))

# Explicitly show sentiment distribution (Pie Chart)
st.header("🥧 Overall Sentiment Distribution")
sentiment_counts = data['sentiment_label'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999','#99ff99'])
st.pyplot(fig1)

# 3) Time-series trend
st.header("📈 Daily Sentiment Trend")
# extract date
data['date'] = data['date_collected'].dt.date
ts = data.groupby(['date','sentiment_label']).size().unstack(fill_value=0)
fig2, ax2 = plt.subplots(figsize=(8,4))
ts.plot(ax=ax2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Count")
ax2.legend(title="Sentiment", loc="upper left")
st.pyplot(fig2)

# 4) Source breakdown
st.header("📊 Sentiment by Source")
src = data.groupby(['source','sentiment_label']).size().unstack(fill_value=0)
fig3, ax3 = plt.subplots(figsize=(6,4))
src.plot(kind='bar', stacked=True, ax=ax3)
ax3.set_xlabel("Source")
ax3.set_ylabel("Count")
st.pyplot(fig3)

# Explicitly generate and show Word Cloud
st.header("☁️ Word Cloud (Most Frequent Words)")

# Combine all text explicitly for word cloud
text_combined = " ".join(data['content'].values)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.imshow(wordcloud, interpolation='bilinear')
ax4.axis("off")
st.pyplot(fig4)

# 6) Evaluation (if you have a labeled CSV)
LABELS_CSV = os.path.join(os.path.dirname(__file__), "text_labels.csv")

if os.path.exists(LABELS_CSV):
    st.header("🧪 VADER Evaluation on Hand-Labeled Text")
    eval_df = pd.read_csv(LABELS_CSV)
    from sentiment_analysis import analyze_and_classify
    from evaluation import evaluate

    gold = eval_df['gold_label'].tolist()
    preds = [analyze_and_classify(t)[1] for t in eval_df['text']]

    m = evaluate(gold, preds, labels=['Negative','Neutral','Positive'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{m['accuracy']:.2%}")
    col2.metric("Precision", f"{m['precision']:.2%}")
    col3.metric("Recall", f"{m['recall']:.2%}")
    col4.metric("F1 Score", f"{m['f1']:.2%}")

    cm = m['confusion_matrix']
    cm_df = pd.DataFrame(cm,
                         index=['Neg','Neu','Pos'],
                         columns=['Neg','Neu','Pos'])
    st.subheader("Confusion Matrix")
    st.table(cm_df)

st.markdown("---")
st.write("Built with ❤️ by Group 4")
st.write("Edo State Election, 2024: Opinion Mining System")
