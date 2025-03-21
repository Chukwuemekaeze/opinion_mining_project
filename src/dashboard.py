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
    df = pd.read_sql_query("SELECT * FROM sentiment_data", conn)
    conn.close()
    return df

data = load_data()

# Dashboard Title
st.title("📊 Edo Election 2024 Opinion Mining Dashboard")

# Explicitly show data
st.header("📋 Sentiment Analysis Data")
st.dataframe(data.head(10))

# Explicitly show sentiment distribution (Pie Chart)
st.header("🥧 Sentiment Distribution")
sentiment_counts = data['sentiment_label'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999','#99ff99'])
st.pyplot(fig1)

# Explicitly generate and show Word Cloud
st.header("☁️ Word Cloud (Most Frequent Words)")

# Combine all text explicitly for word cloud
text_combined = " ".join(data['content'].values)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis("off")

st.pyplot(fig2)

