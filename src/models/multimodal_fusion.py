# src/models/multimodal_fusion.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class MultimodalFusion:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, txt_feats, aud_feats, img_feats, labels):
        # flatten image feats if needed
        if img_feats.ndim > 2:
            img_feats = img_feats.reshape(len(img_feats), -1)
        X = np.hstack([txt_feats.toarray() if hasattr(txt_feats, "toarray") else txt_feats,
                       aud_feats,
                       img_feats])
        self.clf.fit(X, labels)

    def predict(self, txt_feats, aud_feats, img_feats):
        if img_feats.ndim > 2:
            img_feats = img_feats.reshape(len(img_feats), -1)
        X = np.hstack([txt_feats.toarray() if hasattr(txt_feats, "toarray") else txt_feats,
                       aud_feats,
                       img_feats])
        return self.clf.predict(X)