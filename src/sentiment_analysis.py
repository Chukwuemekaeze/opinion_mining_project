"""
sentiment_analysis.py

This module provides functions to analyze sentiment using VADER, 
after cleaning raw text for best results.
"""

import os
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# pull in our cleaning helper
from data_preprocessing import clean_text_for_vader

# Ensure that the VADER lexicon is downloaded.
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER analyzer
sia = SentimentIntensityAnalyzer()

# ── Load Pidgin lexicon and merge into VADER ─────────────────────────────────
pidgin_path = os.path.join(os.path.dirname(__file__), "..//src/pidgin_lexicon.csv")  
with open(pidgin_path, newline='', encoding='utf-8') as f:                                            
    reader = csv.DictReader(f)                                                                         
    for row in reader:                                                                                 
        word = row['word']                                                                             
        score = float(row['sentiment_score'])                                                                    
        sia.lexicon[word] = score

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Parameters:
        text (str): The text to analyze (already cleaned).

    Returns:
        dict: A dictionary containing VADER sentiment scores.
    """
    scores = sia.polarity_scores(text)
    return scores

def classify_sentiment(compound_score):
    """
    Classify sentiment based on the compound score.

    Parameters:
        compound_score (float): The compound sentiment score.

    Returns:
        str: 'Positive', 'Negative', or 'Neutral' sentiment.
    """
    if compound_score >= 0.10:
        return "Positive"
    elif compound_score <= -0.10:
        return "Negative"
    else:
        return "Neutral"

def analyze_and_classify(raw_text):
    """
    Clean then analyze the sentiment of the text, returning both the scores and a label.

    Parameters:
        raw_text (str): The original, uncleaned text.

    Returns:
        tuple: (sentiment_scores (dict), label (str))
    """
    # 1) Clean for VADER
    cleaned = clean_text_for_vader(raw_text)

    # 2) Compute VADER scores on cleaned text
    scores = analyze_sentiment(cleaned)

    # 3) Derive label
    label = classify_sentiment(scores['compound'])
    return scores, label