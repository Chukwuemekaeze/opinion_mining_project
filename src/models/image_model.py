# src/models/image_model.py
import numpy as np
from sklearn.decomposition import PCA

class ImageModel:
    def __init__(self, n_components=50):
        # we'll flatten and reduce via PCA
        self.pca = PCA(n_components=n_components)

    def fit(self, images, labels=None):
        # images: array of shape (n, H, W, C)
        flat = images.reshape(len(images), -1)
        self.pca.fit(flat)

    def encode(self, images):
        flat = images.reshape(len(images), -1)
        return self.pca.transform(flat)

    def predict(self, images):
        # we don’t actually predict in isolation
        raise NotImplementedError("Image-only prediction not implemented")