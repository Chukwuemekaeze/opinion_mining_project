# src/main.py

"""
main.py

Opinion Mining System for analyzing the 2024 Edo State Governmental Election.
This script collects data from a specified source, analyzes the sentiment using VADER,
visualizes sentiment distribution, and saves results into SQLite database.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Existing imports from your modules
try:
    from data_collection import collect_tweets, scrape_web_page, fetch_news
    from sentiment_analysis import analyze_and_classify
    from database import initialize_db, insert_sentiment_data  # <- ADDED explicitly here ✅
except ImportError:
    from src.data_collection import collect_tweets, scrape_web_page, fetch_news
    from src.sentiment_analysis import analyze_and_classify
    from src.database import initialize_db, insert_sentiment_data  # <- ADDED explicitly here ✅

def main():
    parser = argparse.ArgumentParser(
        description="Opinion Mining System for the 2024 Edo State Governmental Election"
    )
    parser.add_argument("source", choices=["twitter", "web", "news"],
                        help="Data source: twitter, web, or news")
    parser.add_argument("query_or_url", help="For 'twitter' and 'news', provide a search query; for 'web', provide a URL to scrape")
    args = parser.parse_args()

    # Collect data based on chosen source
    if args.source == "twitter":
        texts = collect_tweets(query=args.query_or_url)
        if not texts:
            print("No tweets collected.")
            sys.exit(1)
    elif args.source == "web":
        texts = scrape_web_page(args.query_or_url)
        if not texts:
            print("No content scraped from the webpage.")
            sys.exit(1)
    elif args.source == "news":
        articles = fetch_news(query=args.query_or_url)
        if not articles:
            print("No news articles found.")
            sys.exit(1)
        texts = [
            f"{article.get('title', '')} {article.get('description', '')}".strip()
            for article in articles
        ]
    else:
        print("Invalid source. Choose one of: twitter, web, news.")
        sys.exit(1)

    # Initialize DB (run this before inserting data)
    initialize_db()  # <- Clearly ADDED HERE ✅

    # Process texts with sentiment analysis
    results = []
    for text in texts:
        scores, label = analyze_and_classify(text)
        
        # === STEP 3: INSERT INTO DATABASE HERE ===
        insert_sentiment_data(
            source=args.source,
            content=text,
            sentiment_label=label,
            sentiment_score=scores['compound']
        )  # <- Explicitly ADDED HERE ✅
        # === END OF STEP 3 ===
        
        results.append({
            "text": text,
            "compound": scores['compound'],
            "label": label
        })

    # Create a DataFrame to display the results
    df = pd.DataFrame(results)
    print("\nSentiment Analysis Results (first 5 rows):")
    print(df.head())

    # Visualize sentiment distribution
    distribution = df['label'].value_counts()
    plt.figure(figsize=(8, 6))
    distribution.plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
