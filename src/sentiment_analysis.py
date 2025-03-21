"""
sentiment_analysis.py

This module provides functions to analyze sentiment using VADER.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure that the VADER lexicon is downloaded.
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Parameters:
        text (str): The text to analyze.

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
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_and_classify(text):
    """
    Analyze the sentiment of the text and return both the scores and a classification label.

    Parameters:
        text (str): The text to analyze.

    Returns:
        tuple: (sentiment_scores (dict), label (str))
    """
    scores = analyze_sentiment(text)
    label = classify_sentiment(scores['compound'])
    return scores, label
