"""
data_collection.py

This module collects data from three sources:
1. Twitter API (using Tweepy v2)
2. Web scraping (using requests + BeautifulSoup)
3. News API (using NewsAPI.org)

Requirements:
- A config.py file containing:
    BEARER_TOKEN = "<your Twitter v2 Bearer Token>"
    NEWS_API_KEY = "<your News API key>"
- Add config.py to your .gitignore to avoid committing credentials.
"""

import sys
import io
import ssl
import warnings
import requests
import tweepy
from bs4 import BeautifulSoup

# ==============================
# 1) GLOBAL SSL CONTEXT & WARNINGS
# ==============================
# Disable SSL verification (insecure: use only for testing)
ssl._create_default_https_context = ssl._create_unverified_context

from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Force UTF-8 output (fixes encoding issues with special characters)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==============================
# 2) CONFIGURATION
# ==============================
# Attempt to import config.py for API credentials.
try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("WARNING: config.py not found. API-based functions will not work without credentials.")

# ==============================
# 3) TWITTER API DATA COLLECTION
# ==============================
def collect_tweets(query="Edo State election", max_results=10):
    """
    Collect tweets using Twitter API v2 via Tweepy.
    
    Parameters:
        query (str): The search query for tweets.
        max_results (int): Maximum number of tweets to return (up to 100).

    Returns:
        list: A list of tweet texts.
    """
    if not HAS_CONFIG or not hasattr(config, 'BEARER_TOKEN'):
        print("No BEARER_TOKEN found in config.py. Please add 'BEARER_TOKEN' for Twitter API.")
        return []

    client = tweepy.Client(bearer_token=config.BEARER_TOKEN, wait_on_rate_limit=True)

    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "text"]
        )
    except tweepy.errors.Forbidden as e:
        print(f"403 Forbidden: {e}")
        print("Your account may not have access to the needed v2 endpoints.")
        return []
    except tweepy.TweepyException as e:
        print(f"Error using Tweepy v2: {e}")
        return []

    if not response.data:
        return []

    tweets = [tweet.text for tweet in response.data]
    return tweets

# ==============================
# 4) WEB SCRAPING DATA COLLECTION
# ==============================
def scrape_web_page(url):
    """
    Scrape the given URL and extract all non-empty paragraph texts.

    Parameters:
        url (str): The URL of the web page to scrape.

    Returns:
        list: A list of paragraph strings.
    """
    try:
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/91.0.4472.124 Safari/537.36')
        }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        return paragraphs
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

# ==============================
# 5) NEWS API DATA COLLECTION
# ==============================
def fetch_news(query, page_size=10):
    """
    Fetch news articles from NewsAPI.org based on the query.
    
    Parameters:
        query (str): The search query for news articles.
        page_size (int): Number of articles to retrieve per request.

    Returns:
        list: A list of news articles (each article is a dict).
    """
    if not HAS_CONFIG or not hasattr(config, 'NEWS_API_KEY'):
        print("No NEWS_API_KEY found in config.py. Please add 'NEWS_API_KEY' for news functionality.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "apiKey": config.NEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# ==============================
# 6) MAIN FUNCTION FOR TESTING
# ==============================
def main():
    """
    Main function to test data collection functions.

    Usage:
        python data_collection.py [source] [query_or_url]

    source: 'twitter', 'web', or 'news'
        - For 'twitter' and 'news', the remaining arguments are treated as the search query.
        - For 'web', the next argument must be the URL to scrape.
    """
    if len(sys.argv) < 2:
        print("Usage: python data_collection.py [source] [query_or_url]")
        print("source options: twitter, web, news")
        sys.exit(1)
    
    source = sys.argv[1].lower()

    if source == "twitter":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Edo State election"
        tweets = collect_tweets(query=query)
        if tweets:
            print("Collected Tweets:")
            for tweet in tweets:
                print(f"- {tweet}")
        else:
            print("No tweets collected.")

    elif source == "web":
        if len(sys.argv) < 3:
            print("Usage: python data_collection.py web [URL]")
            sys.exit(1)
        url = sys.argv[2]
        paragraphs = scrape_web_page(url)
        if paragraphs:
            print("Paragraphs from web page (first 5 shown):")
            for para in paragraphs[:5]:
                print(f"- {para}")
        else:
            print("No content scraped from the webpage.")

    elif source == "news":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Edo State election"
        articles = fetch_news(query=query)
        if articles:
            print(f"News articles for '{query}':")
            for article in articles:
                print(f"- {article.get('title')} ({article.get('url')})")
        else:
            print(f"No news articles found for '{query}'.")
    else:
        print("Invalid source. Choose one of: twitter, web, news.")

if __name__ == "__main__":
    main()