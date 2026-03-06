"""AdPulse - Data loading and preprocessing utilities."""

import os
import re
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUPERBOWL_CSV = os.path.join(DATA_DIR, "superbowl_ads.csv")
COMMENTS_CSV = os.path.join(DATA_DIR, "youtube_comments.csv")

# TidyTuesday version: includes YouTube engagement metrics + video descriptions
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2021/2021-03-02/youtube.csv"
)


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "sentiment/vader_lexicon": "vader_lexicon",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def download_superbowl_data():
    """Download the Super Bowl ads dataset if not already cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SUPERBOWL_CSV):
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(SUPERBOWL_CSV, "w") as f:
            f.write(response.text)
    return SUPERBOWL_CSV


def load_superbowl_data():
    """Load, clean, and enrich the Super Bowl ads dataset."""
    download_superbowl_data()
    df = pd.read_csv(SUPERBOWL_CSV)

    # Normalize boolean columns (handles R-style TRUE/FALSE)
    bool_cols = [
        "funny", "show_product_quickly", "patriotic",
        "celebrity", "danger", "animals", "use_sex",
    ]
    bool_map = {
        True: True, False: False,
        "TRUE": True, "FALSE": False,
        "True": True, "False": False,
    }
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(False)

    # Clean numeric columns
    num_cols = ["view_count", "like_count", "dislike_count",
                "comment_count", "favorite_count"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Feature engineering
    total_reactions = df["like_count"] + df["dislike_count"]
    df["engagement_rate"] = (
        (df["like_count"] + df["comment_count"])
        / df["view_count"].replace(0, 1)
    )
    df["like_ratio"] = df["like_count"] / total_reactions.replace(0, 1)

    # Display title
    df["title_display"] = df.apply(
        lambda r: r["title"] if pd.notna(r.get("title")) else f"{r['brand']} ({r['year']})",
        axis=1,
    )

    return df


def load_comments():
    """Load cached YouTube comments if available."""
    if os.path.exists(COMMENTS_CSV):
        return pd.read_csv(COMMENTS_CSV)
    return None


def preprocess_text(text):
    """Clean and preprocess text for NLP analysis."""
    ensure_nltk_data()
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)
