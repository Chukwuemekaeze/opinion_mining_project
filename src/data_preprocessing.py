# src/data_preprocessing.py

import re
import numpy as np
import librosa
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

# ── Ensure NLTK resources are present ────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── BERT tokenizer for multimodal pipeline ───────────────────────────────────
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(texts):
    """
    Tokenize texts for BERT encoding.
    Used by the multimodal TextModel.encode().
    Returns a transformers BatchEncoding.
    """
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encodings


# ── VADER-style cleaning for live pipeline ────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_for_vader(text):
    """
    Clean a single text string: lowercase, strip URLs/mentions/hashtags,
    remove non-alpha, remove stop-words, lemmatize.
    Returns a cleaned string ready for VADER.
    """
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(tokens)


# ── Audio preprocessing (MFCC features) ───────────────────────────────────────
def preprocess_audio(filepaths, sr=16000, n_mfcc=13):
    """
    Given a list of audio file paths, load each, compute MFCCs,
    and return an (n_samples × n_mfcc) numpy array of mean MFCC features.
    """
    features = []
    for fp in filepaths:
        y, _ = librosa.load(fp, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features.append(np.mean(mfcc, axis=1))
    return np.vstack(features)


# ── Image preprocessing (resize + normalize) ─────────────────────────────────
def preprocess_image(filepaths, size=(224, 224)):
    """
    Given a list of image file paths, load each with OpenCV,
    resize to `size`, normalize pixel values to [0,1],
    and return an array (n_samples, H, W, C).
    """
    feats = []
    for fp in filepaths:
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(f"Could not read image file: {fp}")
        img = cv2.resize(img, size)
        feats.append(img.astype(np.float32) / 255.0)
    return np.stack(feats)