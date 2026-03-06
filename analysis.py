"""AdPulse - AI analysis engine for sentiment, topics, and engagement."""

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk


class SentimentAnalyzer:
    """Sentiment analysis using VADER (fast) or BERT (deep learning)."""

    def __init__(self, method="vader"):
        self.method = method
        if method == "bert":
            try:
                from transformers import pipeline as hf_pipeline
                self.model = hf_pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    truncation=True,
                    max_length=512,
                )
            except ImportError:
                self.method = "vader"
                self._init_vader()
        else:
            self._init_vader()

    def _init_vader(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        self.model = SentimentIntensityAnalyzer()

    def analyze(self, texts):
        """Analyze sentiment for a list of texts. Returns a DataFrame."""
        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                # Keep placeholder so output length matches input
                results.append({
                    "text": "", "sentiment": "neutral",
                    "compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0,
                })
                continue

            if self.method == "bert":
                out = self.model(text[:512])[0]
                stars = int(out["label"][0])
                sentiment = (
                    "positive" if stars >= 4
                    else ("negative" if stars <= 2 else "neutral")
                )
                results.append({
                    "text": text[:200], "sentiment": sentiment,
                    "compound": (stars - 3) / 2,  # normalize to ~[-1,1]
                    "pos": out["score"] if sentiment == "positive" else 0,
                    "neg": out["score"] if sentiment == "negative" else 0,
                    "neu": out["score"] if sentiment == "neutral" else 0,
                    "stars": stars,
                })
            else:
                scores = self.model.polarity_scores(text)
                compound = scores["compound"]
                sentiment = (
                    "positive" if compound >= 0.05
                    else ("negative" if compound <= -0.05 else "neutral")
                )
                results.append({
                    "text": text[:200], "sentiment": sentiment,
                    "compound": compound, "pos": scores["pos"],
                    "neg": scores["neg"], "neu": scores["neu"],
                })
        return pd.DataFrame(results)


class SentimentEvaluator:
    """Sanity-checking and evaluation utilities for sentiment results."""

    @staticmethod
    def distribution_check(results_df):
        """Check that sentiment distribution is reasonable (not all one class)."""
        counts = results_df["sentiment"].value_counts(normalize=True)
        dominant = counts.max()
        flags = []
        if dominant > 0.95:
            flags.append(f"Warning: {dominant:.0%} of texts share the same sentiment — results may be unreliable.")
        if len(counts) < 2:
            flags.append("Warning: Only one sentiment class detected — check input text quality.")
        return {"distribution": counts.to_dict(), "dominant_pct": dominant, "flags": flags}

    @staticmethod
    def known_sample_check(analyzer):
        """Validate the analyzer against known-sentiment examples."""
        test_cases = [
            ("This is absolutely amazing, best product ever!", "positive"),
            ("Terrible experience, total waste of money.", "negative"),
            ("The meeting is scheduled for Tuesday.", "neutral"),
            ("I love this incredible ad, so funny!", "positive"),
            ("Worst commercial I have ever seen, awful.", "negative"),
        ]
        results = analyzer.analyze([t for t, _ in test_cases])
        correct = sum(
            1 for (_, expected), (_, row) in zip(test_cases, results.iterrows())
            if row["sentiment"] == expected
        )
        accuracy = correct / len(test_cases)
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_cases),
            "details": [
                {"text": text, "expected": exp, "predicted": row["sentiment"],
                 "match": row["sentiment"] == exp}
                for (text, exp), (_, row) in zip(test_cases, results.iterrows())
            ],
        }


class TopicModeler:
    """LDA topic modeling for discovering themes in text."""

    def __init__(self, n_topics=5, max_features=1000):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_df=0.95, min_df=2,
            max_features=max_features,
            stop_words="english",
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20,
        )
        self.feature_names = None

    def fit_transform(self, texts):
        """Fit LDA model and return topic distributions."""
        dtm = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.lda.fit_transform(dtm)

    def get_top_words(self, n_words=10):
        """Get the top words and their weights for each topic."""
        if self.feature_names is None:
            return {}
        topics = {}
        for i, weights in enumerate(self.lda.components_):
            top_idx = weights.argsort()[-n_words:][::-1]
            topics[f"Topic {i + 1}"] = [
                (self.feature_names[j], float(weights[j])) for j in top_idx
            ]
        return topics

    def get_dominant_topic(self, topic_dist):
        """Return dominant topic index (1-based) for each document."""
        return np.argmax(topic_dist, axis=1) + 1

    def coherence_proxy(self, texts):
        """Compute a simple topic-distinctness proxy score (0-1).

        Measures how distinct topics are from each other using average
        pairwise cosine distance of topic-word distributions. Higher = better.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if self.lda.components_ is None or len(self.lda.components_) < 2:
            return None
        normed = self.lda.components_ / self.lda.components_.sum(axis=1, keepdims=True)
        sim_matrix = cosine_similarity(normed)
        n = len(sim_matrix)
        # Average off-diagonal similarity
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        distinctness = 1 - avg_sim  # higher = more distinct topics
        return round(float(distinctness), 3)


class EngagementAnalyzer:
    """Engagement metrics analysis and ad performance scoring."""

    def __init__(self, df):
        self.df = df.copy()

    def compute_composite_score(self):
        """Compute a normalized composite engagement score (0-100)."""
        df = self.df.copy()
        for m in ["view_count", "like_count", "comment_count"]:
            if m in df.columns:
                min_v, max_v = df[m].min(), df[m].max()
                df[f"{m}_norm"] = (df[m] - min_v) / (max_v - min_v + 1e-10)
        df["composite_score"] = (
            df.get("view_count_norm", 0) * 0.40
            + df.get("like_count_norm", 0) * 0.35
            + df.get("comment_count_norm", 0) * 0.25
        ) * 100
        return df

    def feature_impact(self):
        """Measure how each ad characteristic impacts engagement."""
        features = [
            "funny", "show_product_quickly", "patriotic",
            "celebrity", "danger", "animals", "use_sex",
        ]
        available = [f for f in features if f in self.df.columns]
        rows = []
        for feat in available:
            has = self.df[self.df[feat] == True]
            no = self.df[self.df[feat] == False]
            if len(has) == 0 or len(no) == 0:
                continue
            rows.append({
                "feature": feat.replace("_", " ").title(),
                "avg_views_with": has["view_count"].mean(),
                "avg_views_without": no["view_count"].mean(),
                "avg_engagement_with": has["engagement_rate"].mean(),
                "avg_engagement_without": no["engagement_rate"].mean(),
                "count_with": len(has),
                "count_without": len(no),
                "views_lift": (
                    has["view_count"].mean()
                    / max(no["view_count"].mean(), 1) - 1
                ) * 100,
            })
        return pd.DataFrame(rows)

    def brand_summary(self):
        """Summarize performance by brand."""
        return (
            self.df.groupby("brand")
            .agg(
                total_ads=("year", "count"),
                avg_views=("view_count", "mean"),
                total_views=("view_count", "sum"),
                avg_likes=("like_count", "mean"),
                avg_engagement=("engagement_rate", "mean"),
                avg_like_ratio=("like_ratio", "mean"),
            )
            .reset_index()
            .sort_values("avg_views", ascending=False)
        )


class TrendAnalyzer:
    """Google Trends analysis for brand search interest."""

    @staticmethod
    def get_search_trends(keywords, timeframe="today 3-m"):
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl="en-US")
            all_data = pd.DataFrame()
            for i in range(0, len(keywords), 5):
                batch = keywords[i : i + 5]
                pytrends.build_payload(batch, timeframe=timeframe)
                data = pytrends.interest_over_time()
                if not data.empty:
                    data = data.drop(columns=["isPartial"], errors="ignore")
                    all_data = pd.concat([all_data, data], axis=1)
            return all_data if not all_data.empty else None
        except Exception:
            return None

    @staticmethod
    def get_related_queries(keyword):
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl="en-US")
            pytrends.build_payload([keyword], timeframe="today 3-m")
            related = pytrends.related_queries()
            return related.get(keyword, {})
        except Exception:
            return None
