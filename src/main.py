# src/main.py

"""
main.py

Opinion Mining System for analyzing the 2024 Edo State Governmental Election.
This script collects data from a specified source, analyzes the sentiment using VADER,
visualizes sentiment distribution, and saves results into SQLite database.

Extended to support multimodal dataset training & evaluation (text, audio, image) with metrics.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ensure project root in path
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

# New multimodal imports
from data_preprocessing import preprocess_text, preprocess_audio, preprocess_image
from models.text_model import TextModel
from models.audio_model import AudioModel
from models.image_model import ImageModel
from models.multimodal_fusion import MultimodalFusion
from evaluation import evaluate


def load_dataset(csv_path):
    """
    Load a CSV dataset with columns: text, audio_path, image_path, label
    """
    df = pd.read_csv(csv_path)
    return df['text'].tolist(), df['audio_path'].tolist(), df['image_path'].tolist(), df['label'].tolist()


def live_pipeline(source, query):
    """Run the existing VADER-based live pipeline."""
    # Collect data based on chosen source
    if source == "twitter":
        texts = collect_tweets(query=query)
    elif source == "web":
        texts = scrape_web_page(query)
    else:  # news
        articles = fetch_news(query=query)
        texts = [f"{a.get('title','')} {a.get('description','')}".strip() for a in articles]

    if not texts:
        print("No data collected for live pipeline.")
        sys.exit(1)

    # Initialize DB
    initialize_db()

    results = []
    for text in texts:
        scores, label = analyze_and_classify(text)
        # debug: show raw vs cleaned vs scores
        from data_preprocessing import clean_text_for_vader
        print("RAW:    ", text)
        print("CLEAN:  ", clean_text_for_vader(text))
        print("SCORES: ", scores, "→", label)
        print("-" * 40)
        insert_sentiment_data(source, text, label, scores['compound'])
        results.append({"text": text, "compound": scores['compound'], "label": label})

    df = pd.DataFrame(results)
    print("\nSentiment Analysis Results (first 5 rows):")
    print(df.head())

    dist = df['label'].value_counts()
    plt.figure(figsize=(8,6))
    dist.plot(kind='bar')
    plt.title('Sentiment Distribution (Live Data)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def multimodal_pipeline(dataset, test_size):
    """Run multimodal training & evaluation on labeled CSV dataset."""
    texts, aud_paths, img_paths, labels = load_dataset(dataset)
    txt_tr, txt_te, aud_tr, aud_te, img_tr, img_te, y_tr, y_te = train_test_split(
        texts, aud_paths, img_paths, labels,
        test_size=test_size, random_state=42, stratify=labels
    )

    # preprocess and encode
    txt_feats_tr = TextModel().encode(preprocess_text(txt_tr))
    aud_feats_tr = preprocess_audio(aud_tr)
    img_feats_tr = ImageModel().encode(preprocess_image(img_tr))

    txt_feats_te = TextModel().encode(preprocess_text(txt_te))
    aud_feats_te = preprocess_audio(aud_te)
    img_feats_te = ImageModel().encode(preprocess_image(img_te))

    # train fusion model
    fusion = MultimodalFusion()
    fusion.fit(txt_feats_tr, aud_feats_tr, img_feats_tr, y_tr)
    y_pred = fusion.predict(txt_feats_te, aud_feats_te, img_feats_te)

    # evaluate
    metrics = evaluate(y_te, y_pred, labels=list(sorted(set(labels))))
    print("\nMultimodal Evaluation Metrics:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


def main():
    parser = argparse.ArgumentParser(description="Opinion Mining System: live VADER or multimodal dataset")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # live subcommand
    p_live = subparsers.add_parser('live', help='Run live VADER sentiment on twitter/web/news')
    p_live.add_argument('source', choices=['twitter','web','news'], help="Data source for live mode")
    p_live.add_argument('query', help="Query (for twitter/news) or URL (for web)")

    # multimodal subcommand
    p_mm = subparsers.add_parser('multimodal', help='Run multimodal training & evaluation')
    p_mm.add_argument('--dataset', required=True, help="CSV: text,audio_path,image_path,label")
    p_mm.add_argument('--test-size', type=float, default=0.2, help="Test split proportion")

    # eval subcommand for text-only VADER evaluation
    p_eval = subparsers.add_parser('eval', help='Evaluate text pipeline against labeled CSV')
    p_eval.add_argument('--labels', required=True, 
                        help="CSV file with columns: text,gold_label (Positive/Neutral/Negative)")

    args = parser.parse_args()
    if args.mode == 'live':
        # Use exactly what the user passed in as the query
        print(f"\n>>> Fetching sentiment for {args.query}")
        live_pipeline(args.source, args.query)
    elif args.mode == 'multimodal':
         multimodal_pipeline(args.dataset, args.test_size)
    else:  # args.mode == 'eval'
        # Load gold-labels
        df = pd.read_csv(args.labels)
        texts = df['text'].tolist()
        gold = df['gold_label'].tolist()

        # Run VADER pipeline on each
        preds = []
        for t in texts:
            _, label = analyze_and_classify(t)
            preds.append(label)

        # Compute metrics
        from evaluation import evaluate
        m = evaluate(gold, preds, labels=['Negative','Neutral','Positive'])
        print("\nEvaluation on gold-labeled text:")
        print(f"Accuracy : {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"F1 Score : {m['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(m['confusion_matrix'])


if __name__ == "__main__":
    main()
