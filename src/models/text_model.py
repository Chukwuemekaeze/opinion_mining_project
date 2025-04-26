# src/models/text_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TextModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def encode(self, texts):
        # returns TF-IDF features
        return self.vectorizer.transform(texts)

    def predict(self, texts):
        X = self.encode(texts)
        return self.clf.predict(X)