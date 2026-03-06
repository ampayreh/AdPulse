"""AdPulse v2 - Text Analysis & Clustering module.

Covers MSIS 521 course concepts:
  S2: TF-IDF vectorization, Word2Vec embeddings, t-SNE visualization
  S3: BERT sentiment, cosine similarity, Jaccard similarity,
      K-means clustering, hierarchical clustering
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from data_utils import preprocess_text


# ── TF-IDF Analysis ─────────────────────────────

def build_tfidf_matrix(texts, max_features=500):
    """Build a TF-IDF matrix from a list of texts.

    Returns (tfidf_matrix, feature_names, vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=0.90,
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer


def brand_tfidf_fingerprints(df, n_terms=10):
    """Get top TF-IDF terms per brand.

    Returns a DataFrame with columns: brand, term, tfidf_score.
    """
    rows = []
    brands = df["brand"].unique()
    for brand in brands:
        brand_texts = df[df["brand"] == brand]["description"].dropna().tolist()
        if len(brand_texts) < 2:
            continue
        processed = [preprocess_text(t) for t in brand_texts]
        processed = [t for t in processed if len(t.split()) >= 2]
        if len(processed) < 2:
            continue
        try:
            vec = TfidfVectorizer(max_features=200, max_df=0.95, min_df=1, stop_words="english")
            matrix = vec.fit_transform(processed)
            mean_tfidf = np.asarray(matrix.mean(axis=0)).flatten()
            top_idx = mean_tfidf.argsort()[-n_terms:][::-1]
            names = vec.get_feature_names_out()
            for idx in top_idx:
                rows.append({
                    "brand": brand,
                    "term": names[idx],
                    "tfidf_score": float(mean_tfidf[idx]),
                })
        except ValueError:
            continue
    return pd.DataFrame(rows)


# ── Cosine & Jaccard Similarity ──────────────────

def compute_cosine_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity matrix from TF-IDF vectors."""
    return cosine_similarity(tfidf_matrix)


def compute_brand_jaccard(df):
    """Compute Jaccard similarity between brands based on boolean feature profiles.

    Returns a DataFrame (brands x brands) of Jaccard coefficients.
    """
    features = ["funny", "show_product_quickly", "patriotic",
                 "celebrity", "danger", "animals", "use_sex"]
    available = [f for f in features if f in df.columns]
    if not available:
        return pd.DataFrame()

    brand_profiles = df.groupby("brand")[available].mean()
    brands = brand_profiles.index.tolist()
    n = len(brands)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = set(f for f in available if brand_profiles.iloc[i][f] > 0.5)
            b = set(f for f in available if brand_profiles.iloc[j][f] > 0.5)
            union = a | b
            if len(union) == 0:
                jaccard[i, j] = 0.0
            else:
                jaccard[i, j] = len(a & b) / len(union)

    return pd.DataFrame(jaccard, index=brands, columns=brands)


# ── K-Means Clustering ───────────────────────────

def cluster_ads_kmeans(tfidf_matrix, df, n_clusters=5):
    """Cluster ads using K-Means on TF-IDF features combined with boolean features.

    Returns (labels, cluster_profiles DataFrame).
    """
    features = ["funny", "show_product_quickly", "patriotic",
                 "celebrity", "danger", "animals", "use_sex"]
    available = [f for f in features if f in df.columns]

    # Combine TF-IDF with boolean features
    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else np.array(tfidf_matrix)
    if available:
        bool_features = df[available].values.astype(float)
        # Scale boolean features to match TF-IDF magnitude
        combined = np.hstack([tfidf_dense, bool_features * 0.5])
    else:
        combined = tfidf_dense

    n_clusters = min(n_clusters, len(combined) - 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(combined)

    # Build cluster profiles
    profile_rows = []
    temp_df = df.copy()
    temp_df["cluster"] = labels
    for cl in range(n_clusters):
        cl_data = temp_df[temp_df["cluster"] == cl]
        profile = {"cluster": cl + 1, "size": len(cl_data)}
        profile["avg_views"] = cl_data["view_count"].mean()
        profile["avg_engagement"] = cl_data["engagement_rate"].mean()
        for f in available:
            profile[f"pct_{f}"] = cl_data[f].mean()
        # Top brands in cluster
        top_brands = cl_data["brand"].value_counts().head(3)
        profile["top_brands"] = ", ".join(
            f"{b} ({c})" for b, c in top_brands.items()
        )
        profile_rows.append(profile)

    return labels, pd.DataFrame(profile_rows)


# ── Hierarchical Clustering ──────────────────────

def hierarchical_cluster_brands(df):
    """Hierarchical clustering of brands based on average feature + engagement profiles.

    Returns (linkage_matrix, brand_labels) for dendrogram plotting.
    """
    features = ["funny", "show_product_quickly", "patriotic",
                 "celebrity", "danger", "animals", "use_sex"]
    metrics = ["engagement_rate", "like_ratio"]
    all_cols = [c for c in features + metrics if c in df.columns]

    brand_profiles = df.groupby("brand")[all_cols].mean()
    # Normalize columns to 0-1
    for col in brand_profiles.columns:
        rng = brand_profiles[col].max() - brand_profiles[col].min()
        if rng > 0:
            brand_profiles[col] = (brand_profiles[col] - brand_profiles[col].min()) / rng

    Z = linkage(brand_profiles.values, method="ward")
    return Z, brand_profiles.index.tolist()


# ── t-SNE Visualization ─────────────────────────

def compute_tsne(tfidf_matrix, perplexity=15, random_state=42):
    """Reduce TF-IDF vectors to 2D using t-SNE.

    Returns array of shape (n_samples, 2).
    """
    dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else np.array(tfidf_matrix)
    n_samples = dense.shape[0]
    perplexity = min(perplexity, max(5, n_samples // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=800)
    return tsne.fit_transform(dense)


# ── Word2Vec Embeddings ──────────────────────────

def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    """Train a Word2Vec model on tokenized texts.

    Returns (model, document_embeddings) where document_embeddings is
    the average of word vectors per document.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        return None, None

    tokenized = [t.split() for t in texts]
    tokenized = [t for t in tokenized if len(t) >= 2]

    if len(tokenized) < 5:
        return None, None

    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=42,
        epochs=20,
    )

    # Compute document embeddings (average of word vectors)
    doc_embeddings = []
    for tokens in tokenized:
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if vecs:
            doc_embeddings.append(np.mean(vecs, axis=0))
        else:
            doc_embeddings.append(np.zeros(vector_size))

    return model, np.array(doc_embeddings)


def find_similar_words(model, word, topn=10):
    """Find most similar words to a query using the Word2Vec model."""
    if model is None or word not in model.wv:
        return []
    return model.wv.most_similar(word, topn=topn)


# ── BERT Sentiment ───────────────────────────────

def bert_sentiment_analysis(texts, batch_size=16):
    """Run BERT sentiment analysis (nlptown/bert-base-multilingual-uncased-sentiment).

    Returns DataFrame with columns: text, stars, bert_sentiment, bert_compound.
    Falls back gracefully if transformers is not installed.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return None

    model = hf_pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512,
        device=-1,  # CPU
    )

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:512] if isinstance(t, str) and t.strip() else "neutral text" for t in batch]
        preds = model(batch)
        for text, pred in zip(batch, preds):
            stars = int(pred["label"][0])
            sentiment = "positive" if stars >= 4 else ("negative" if stars <= 2 else "neutral")
            results.append({
                "text": text[:200],
                "stars": stars,
                "bert_sentiment": sentiment,
                "bert_compound": (stars - 3) / 2.0,
                "confidence": pred["score"],
            })

    return pd.DataFrame(results)


def compare_vader_bert(vader_results, bert_results):
    """Build a comparison DataFrame between VADER and BERT sentiment results.

    Returns DataFrame with vader_sentiment, bert_sentiment, agreement columns.
    """
    n = min(len(vader_results), len(bert_results))
    comparison = pd.DataFrame({
        "text": vader_results["text"].iloc[:n].values,
        "vader_sentiment": vader_results["sentiment"].iloc[:n].values,
        "vader_compound": vader_results["compound"].iloc[:n].values,
        "bert_sentiment": bert_results["bert_sentiment"].iloc[:n].values,
        "bert_compound": bert_results["bert_compound"].iloc[:n].values,
        "bert_stars": bert_results["stars"].iloc[:n].values,
    })
    comparison["agreement"] = comparison["vader_sentiment"] == comparison["bert_sentiment"]
    return comparison
