"""AdPulse v2 - SEM Performance Lens module.

Covers MSIS 521 course concepts:
  S5: CTR, CPC, CPA, Quality Score, ROA, keyword strategy
      (inspired by the Air France SEM case study)

Note: The Super Bowl ads dataset has no actual budget/impression data.
This module estimates SEM-equivalent metrics using industry benchmarks
to demonstrate understanding of SEM concepts in a creative context.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from data_utils import preprocess_text


# ── Industry Benchmark Estimates ─────────────────
# Average Super Bowl ad costs by year (30-second spot, in millions USD)
# Source: industry reports -- approximate figures for demonstration
SUPERBOWL_AD_COSTS = {
    2000: 2.1, 2001: 2.1, 2002: 2.2, 2003: 2.2, 2004: 2.3,
    2005: 2.4, 2006: 2.5, 2007: 2.6, 2008: 2.7, 2009: 3.0,
    2010: 2.8, 2011: 3.0, 2012: 3.5, 2013: 3.8, 2014: 4.0,
    2015: 4.5, 2016: 4.8, 2017: 5.0, 2018: 5.2, 2019: 5.3,
    2020: 5.6,
}

# Average Super Bowl viewership (millions) by year
SUPERBOWL_VIEWERSHIP = {
    2000: 88.5, 2001: 84.3, 2002: 86.8, 2003: 88.6, 2004: 89.8,
    2005: 86.1, 2006: 90.7, 2007: 93.2, 2008: 97.5, 2009: 98.7,
    2010: 106.5, 2011: 111.0, 2012: 111.3, 2013: 108.7, 2014: 112.2,
    2015: 114.4, 2016: 111.9, 2017: 111.3, 2018: 103.4, 2019: 98.2,
    2020: 99.9,
}


# ── SEM Metric Computation ───────────────────────

def compute_sem_metrics(df):
    """Compute SEM-equivalent metrics for each ad.

    Uses industry benchmarks to estimate cost, then derives:
      - estimated_cost: Cost estimate based on year
      - tv_impressions: Estimated TV impressions from Super Bowl viewership
      - cpv (Cost Per View): estimated_cost / youtube_view_count
      - cpe (Cost Per Engagement): estimated_cost / (likes + comments)
      - engagement_roi: composite engagement per dollar
      - quality_score: 0-10 proxy combining engagement rate, like ratio, sentiment

    Returns DataFrame with SEM columns added.
    """
    result = df.copy()

    # Estimate cost (in USD)
    result["estimated_cost"] = result["year"].map(SUPERBOWL_AD_COSTS).fillna(3.5) * 1_000_000

    # Estimate TV impressions
    result["tv_impressions"] = result["year"].map(SUPERBOWL_VIEWERSHIP).fillna(100) * 1_000_000

    # CPV - Cost Per (YouTube) View
    result["cpv"] = result["estimated_cost"] / result["view_count"].replace(0, 1)

    # CPE - Cost Per Engagement (likes + comments as "conversions")
    total_engagements = result["like_count"] + result["comment_count"]
    result["cpe"] = result["estimated_cost"] / total_engagements.replace(0, 1)

    # Engagement ROI (engagement score per million dollars)
    # First compute a simple engagement score if composite_score not present
    for m in ["view_count", "like_count", "comment_count"]:
        col = f"{m}_norm"
        if col not in result.columns:
            min_v, max_v = result[m].min(), result[m].max()
            result[col] = (result[m] - min_v) / (max_v - min_v + 1e-10)

    result["engagement_score"] = (
        result.get("view_count_norm", 0) * 0.40
        + result.get("like_count_norm", 0) * 0.35
        + result.get("comment_count_norm", 0) * 0.25
    ) * 100

    result["engagement_roi"] = result["engagement_score"] / (result["estimated_cost"] / 1_000_000)

    # Quality Score Proxy (0-10)
    er_norm = result["engagement_rate"].rank(pct=True)
    lr_norm = result["like_ratio"].rank(pct=True)
    view_norm = result["view_count"].rank(pct=True)
    result["quality_score"] = ((er_norm * 0.35 + lr_norm * 0.35 + view_norm * 0.30) * 10).round(1)

    # Digital CTR equivalent: YouTube views / TV impressions
    result["digital_ctr"] = result["view_count"] / result["tv_impressions"].replace(0, 1)

    return result


# ── Publisher-Style Brand Comparison ─────────────

def brand_publisher_comparison(df_with_sem):
    """Rank brands like publishers in the Air France case study.

    Computes average SEM metrics by brand and ranks them.
    Returns DataFrame sorted by engagement_roi descending.
    """
    brand_metrics = df_with_sem.groupby("brand").agg(
        num_ads=("year", "count"),
        avg_cost=("estimated_cost", "mean"),
        avg_views=("view_count", "mean"),
        avg_cpv=("cpv", "mean"),
        avg_cpe=("cpe", "mean"),
        avg_roi=("engagement_roi", "mean"),
        avg_quality_score=("quality_score", "mean"),
        avg_digital_ctr=("digital_ctr", "mean"),
        total_engagements=("like_count", "sum"),
    ).reset_index()

    # Add total comments
    comment_totals = df_with_sem.groupby("brand")["comment_count"].sum().reset_index()
    comment_totals.columns = ["brand", "total_comments"]
    brand_metrics = brand_metrics.merge(comment_totals, on="brand", how="left")
    brand_metrics["total_engagements"] = brand_metrics["total_engagements"] + brand_metrics["total_comments"].fillna(0)

    # Rank columns
    for col in ["avg_cpv", "avg_cpe"]:
        brand_metrics[f"{col}_rank"] = brand_metrics[col].rank(ascending=True).astype(int)
    for col in ["avg_roi", "avg_quality_score"]:
        brand_metrics[f"{col}_rank"] = brand_metrics[col].rank(ascending=False).astype(int)

    return brand_metrics.sort_values("avg_roi", ascending=False).reset_index(drop=True)


# ── Keyword Strategy Analysis ────────────────────

def keyword_strategy_analysis(df, n_terms=15):
    """Extract and categorize key terms from ad descriptions by brand.

    Categories:
      - Emotional: terms associated with feelings/reactions
      - Action: terms suggesting viewer action
      - Product: terms about product features
      - Brand: brand-specific terminology

    Returns (brand_keywords DataFrame, category_performance DataFrame).
    """
    # Emotional, action, and product term lists (seed words)
    emotional_seeds = {
        "love", "amazing", "funny", "hilarious", "heartwarming", "exciting",
        "beautiful", "touching", "inspiring", "powerful", "emotional", "happy",
        "sad", "scary", "thrilling", "epic", "awesome", "incredible", "cool",
        "great", "best", "worst", "terrible", "boring", "cute", "sweet",
    }
    action_seeds = {
        "buy", "try", "visit", "watch", "click", "share", "subscribe",
        "order", "call", "download", "join", "enter", "win", "save",
        "get", "check", "follow", "like", "comment", "vote",
    }
    product_seeds = {
        "new", "product", "feature", "model", "edition", "flavor",
        "design", "technology", "performance", "quality", "price",
        "value", "premium", "limited", "special", "exclusive",
    }

    # Get TF-IDF terms per brand
    brand_keywords = []
    texts_by_brand = {}

    for brand in df["brand"].unique():
        brand_texts = df[df["brand"] == brand]["description"].dropna().tolist()
        processed = [preprocess_text(t) for t in brand_texts if isinstance(t, str)]
        processed = [t for t in processed if len(t.split()) >= 2]
        if len(processed) < 2:
            continue

        texts_by_brand[brand] = processed
        try:
            vec = TfidfVectorizer(max_features=100, max_df=0.95, min_df=1, stop_words="english")
            matrix = vec.fit_transform(processed)
            mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
            top_idx = mean_scores.argsort()[-n_terms:][::-1]
            terms = vec.get_feature_names_out()

            for idx in top_idx:
                term = terms[idx]
                # Categorize
                if term.lower() in emotional_seeds:
                    category = "Emotional"
                elif term.lower() in action_seeds:
                    category = "Action"
                elif term.lower() in product_seeds:
                    category = "Product"
                else:
                    category = "Brand/Other"

                brand_keywords.append({
                    "brand": brand,
                    "term": term,
                    "tfidf_score": float(mean_scores[idx]),
                    "category": category,
                })
        except ValueError:
            continue

    brand_kw_df = pd.DataFrame(brand_keywords)

    # Category performance: avg engagement by keyword category
    if not brand_kw_df.empty:
        category_counts = brand_kw_df.groupby(["brand", "category"]).size().reset_index(name="term_count")
        brand_eng = df.groupby("brand").agg(
            avg_views=("view_count", "mean"),
            avg_engagement=("engagement_rate", "mean"),
        ).reset_index()
        category_perf = category_counts.merge(brand_eng, on="brand", how="left")
    else:
        category_perf = pd.DataFrame()

    return brand_kw_df, category_perf


# ── Budget Optimization Simulator ────────────────

def build_engagement_predictor(df):
    """Build a linear regression model predicting engagement from features + cost.

    Returns (model, feature_names, r2_score).
    """
    features = ["funny", "show_product_quickly", "patriotic",
                 "celebrity", "danger", "animals", "use_sex"]
    available = [f for f in features if f in df.columns]

    if not available or len(df) < 10:
        return None, [], 0

    X = df[available].values.astype(float)
    y = df["view_count"].values.astype(float)

    # Log-transform views for better regression
    y_log = np.log1p(y)

    model = LinearRegression()
    model.fit(X, y_log)
    r2 = model.score(X, y_log)

    return model, available, r2


def predict_engagement(model, feature_names, feature_values):
    """Predict expected views given a set of feature toggles.

    feature_values: dict of {feature_name: bool}
    Returns predicted view count.
    """
    if model is None:
        return 0

    X = np.array([[float(feature_values.get(f, 0)) for f in feature_names]])
    y_log_pred = model.predict(X)[0]
    return int(np.expm1(y_log_pred))


def optimal_feature_recommendation(df):
    """Find the feature combination that maximizes engagement ROI.

    Returns dict with recommended features and expected performance.
    """
    features = ["funny", "show_product_quickly", "patriotic",
                 "celebrity", "danger", "animals", "use_sex"]
    available = [f for f in features if f in df.columns]

    if not available:
        return {}

    # Find best single features
    feature_impact = {}
    for feat in available:
        with_feat = df[df[feat] == True]["view_count"].mean()
        without_feat = df[df[feat] == False]["view_count"].mean()
        if without_feat > 0:
            lift = (with_feat / without_feat - 1) * 100
            feature_impact[feat] = {
                "avg_views_with": with_feat,
                "lift_pct": lift,
            }

    # Sort by lift
    sorted_features = sorted(feature_impact.items(), key=lambda x: x[1]["lift_pct"], reverse=True)
    recommended = [f for f, _ in sorted_features if feature_impact[f]["lift_pct"] > 0]

    return {
        "recommended_features": recommended[:3],
        "feature_impact": feature_impact,
        "insight": (
            f"Top performing features: "
            + ", ".join(f.replace("_", " ").title() for f in recommended[:3])
            + f". These drive the highest view lift when present in ads."
            if recommended
            else "No features show consistent positive lift in this dataset."
        ),
    }
