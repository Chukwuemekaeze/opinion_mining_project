# src/models/audio_model.py
from sklearn.ensemble import RandomForestClassifier

class AudioModel:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100)

    def fit(self, features, labels):
        # features: numpy array of MFCCs
        self.clf.fit(features, labels)

    def encode(self, features):
        # for audio we’ll just use raw MFCC means
        return features

    def predict(self, features):
        return self.clf.predict(features)