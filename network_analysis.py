"""AdPulse v2 - Brand Influence Network & Wisdom of Crowds module.

Covers MSIS 521 course concepts:
  S6: Graph theory, NetworkX, degree/betweenness/closeness/PageRank
      centrality, Wisdom of Crowds (Surowiecki's 4 conditions)
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


FEATURES = [
    "funny", "show_product_quickly", "patriotic",
    "celebrity", "danger", "animals", "use_sex",
]


# ── Brand Strategy Network ───────────────────────

def build_brand_network(df, similarity_threshold=0.3):
    """Build a weighted network of brands based on strategy similarity.

    Nodes = brands (sized by total engagement).
    Edges = cosine similarity of average feature profiles (above threshold).

    Returns (networkx.Graph, brand_profiles DataFrame).
    """
    available = [f for f in FEATURES if f in df.columns]
    metrics = ["engagement_rate", "like_ratio"]
    cols = [c for c in available + metrics if c in df.columns]

    brand_profiles = df.groupby("brand")[cols].mean()

    # Normalize
    for col in brand_profiles.columns:
        rng = brand_profiles[col].max() - brand_profiles[col].min()
        if rng > 0:
            brand_profiles[col] = (brand_profiles[col] - brand_profiles[col].min()) / rng

    sim_matrix = cosine_similarity(brand_profiles.values)
    brands = brand_profiles.index.tolist()

    # Total engagement per brand for node sizing
    brand_engagement = df.groupby("brand")["view_count"].sum()

    G = nx.Graph()
    for brand in brands:
        G.add_node(
            brand,
            total_views=int(brand_engagement.get(brand, 0)),
            num_ads=int(len(df[df["brand"] == brand])),
        )

    for i in range(len(brands)):
        for j in range(i + 1, len(brands)):
            sim = sim_matrix[i, j]
            if sim >= similarity_threshold:
                G.add_edge(brands[i], brands[j], weight=round(float(sim), 3))

    return G, brand_profiles


def build_ad_network(df, tfidf_matrix, similarity_threshold=0.15, max_edges=500):
    """Build a content similarity network of individual ads.

    Nodes = ads. Edges = cosine similarity above threshold.
    Limited to max_edges to keep visualization manageable.

    Returns networkx.Graph.
    """
    sim_matrix = cosine_similarity(tfidf_matrix)
    n = sim_matrix.shape[0]

    G = nx.Graph()
    for i in range(n):
        title = df.iloc[i].get("title_display", f"Ad {i}")
        brand = df.iloc[i].get("brand", "Unknown")
        G.add_node(i, title=str(title)[:50], brand=brand)

    # Collect edges above threshold, sort by weight, keep top max_edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                edges.append((i, j, float(sim_matrix[i, j])))

    edges.sort(key=lambda x: x[2], reverse=True)
    for i, j, w in edges[:max_edges]:
        G.add_edge(i, j, weight=round(w, 3))

    return G


# ── Centrality Measures ──────────────────────────

def compute_centrality(G):
    """Compute all four centrality measures for a graph.

    Returns dict of {metric_name: {node: score}}.
    """
    results = {}

    results["degree"] = nx.degree_centrality(G)
    results["betweenness"] = nx.betweenness_centrality(G, weight="weight")
    results["closeness"] = nx.closeness_centrality(G)

    try:
        results["pagerank"] = nx.pagerank(G, weight="weight")
    except nx.PowerIterationFailedConvergence:
        results["pagerank"] = {n: 1.0 / len(G) for n in G.nodes()}

    return results


def centrality_summary(G, centrality_dict):
    """Summarize centrality measures into a DataFrame.

    Returns DataFrame with one row per node and columns for each metric.
    """
    rows = []
    for node in G.nodes():
        row = {"node": node}
        for metric, scores in centrality_dict.items():
            row[metric] = round(scores.get(node, 0), 4)
        # Add node attributes
        for attr_key, attr_val in G.nodes[node].items():
            row[attr_key] = attr_val
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_centrality_by_brand(ad_graph, centrality_dict):
    """Average centrality scores by brand for the ad network.

    Returns DataFrame with columns: brand, avg_degree, avg_betweenness, etc.
    """
    rows = []
    for node in ad_graph.nodes():
        brand = ad_graph.nodes[node].get("brand", "Unknown")
        row = {"brand": brand}
        for metric, scores in centrality_dict.items():
            row[metric] = scores.get(node, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("brand").mean(numeric_only=True).reset_index().round(4)


# ── Pyvis Network Visualization ──────────────────

def brand_network_html(G, height="500px"):
    """Generate an interactive Pyvis HTML visualization of the brand network.

    Returns HTML string suitable for st.components.v1.html().
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    net = Network(height=height, width="100%", bgcolor="#1A1F36", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Color palette for brands
    brand_colors = [
        "#0891B2", "#FF6B6B", "#10B981", "#F59E0B",
        "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
        "#F97316", "#6366F1",
    ]

    brands = list(G.nodes())
    max_views = max((G.nodes[b].get("total_views", 1) for b in brands), default=1)

    for i, brand in enumerate(brands):
        views = G.nodes[brand].get("total_views", 0)
        num_ads = G.nodes[brand].get("num_ads", 0)
        size = 15 + 35 * (views / max_views)
        color = brand_colors[i % len(brand_colors)]
        net.add_node(
            brand, label=brand, size=size, color=color,
            title=f"{brand}\nAds: {num_ads}\nTotal Views: {views:,}",
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0.5)
        width = 1 + weight * 4
        net.add_edge(u, v, value=width, title=f"Similarity: {weight:.2f}")

    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    return net.generate_html()


# ── Wisdom of Crowds ─────────────────────────────

def wisdom_of_crowds_analysis(df):
    """Evaluate the Super Bowl ad ecosystem against Surowiecki's 4 conditions.

    Returns dict with scores and interpretations for:
      - Diversity of opinion
      - Independence
      - Decentralization
      - Aggregation

    Each condition scored 0-100 with interpretation text.
    """
    available_features = [f for f in FEATURES if f in df.columns]
    results = {}

    # 1. DIVERSITY: Shannon entropy of feature combinations
    if available_features:
        combos = df[available_features].astype(str).agg("-".join, axis=1)
        combo_counts = combos.value_counts(normalize=True)
        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in combo_counts if p > 0)
        max_entropy = np.log2(len(combo_counts)) if len(combo_counts) > 1 else 1
        diversity_score = min(100, (entropy / max_entropy) * 100) if max_entropy > 0 else 0
        results["diversity"] = {
            "score": round(diversity_score, 1),
            "metric": f"Shannon entropy: {entropy:.2f} / {max_entropy:.2f}",
            "unique_strategies": len(combo_counts),
            "interpretation": (
                "High diversity -- ads employ many different feature combinations."
                if diversity_score >= 60
                else "Moderate diversity -- some convergence in ad strategies."
                if diversity_score >= 30
                else "Low diversity -- most ads follow similar strategies."
            ),
        }
    else:
        results["diversity"] = {"score": 0, "interpretation": "No feature data available."}

    # 2. INDEPENDENCE: Do brands copy each other year-over-year?
    # Measure: average correlation of feature usage between brand pairs
    if available_features and "year" in df.columns:
        yearly_brand = df.groupby(["year", "brand"])[available_features].mean().reset_index()
        brands = yearly_brand["brand"].unique()
        correlations = []
        for f in available_features:
            pivoted = yearly_brand.pivot(index="year", columns="brand", values=f).dropna()
            if pivoted.shape[1] >= 2:
                corr_matrix = pivoted.corr()
                # Average off-diagonal correlation
                n = len(corr_matrix)
                if n > 1:
                    off_diag = (corr_matrix.values.sum() - n) / (n * (n - 1))
                    correlations.append(abs(off_diag))

        avg_correlation = np.mean(correlations) if correlations else 0
        independence_score = max(0, (1 - avg_correlation) * 100)
        results["independence"] = {
            "score": round(independence_score, 1),
            "metric": f"Avg inter-brand correlation: {avg_correlation:.3f}",
            "interpretation": (
                "High independence -- brands make strategy choices independently."
                if independence_score >= 60
                else "Moderate independence -- some herding behavior in strategy choices."
                if independence_score >= 30
                else "Low independence -- brands tend to follow similar trends together."
            ),
        }
    else:
        results["independence"] = {"score": 0, "interpretation": "Insufficient data."}

    # 3. DECENTRALIZATION: Is engagement dominated by a few brands?
    # Measure: inverse Gini coefficient of engagement distribution
    if "view_count" in df.columns:
        brand_views = df.groupby("brand")["view_count"].sum().sort_values()
        values = brand_views.values.astype(float)
        n = len(values)
        if n > 1 and values.sum() > 0:
            # Gini coefficient
            cumulative = np.cumsum(values) / values.sum()
            gini = 1 - 2 * (cumulative.sum() / n)
            decentralization_score = max(0, (1 - abs(gini)) * 100)
        else:
            gini = 0
            decentralization_score = 50

        results["decentralization"] = {
            "score": round(decentralization_score, 1),
            "metric": f"Gini coefficient: {abs(gini):.3f}",
            "top_brand_share": f"{brand_views.iloc[-1] / brand_views.sum():.1%}" if brand_views.sum() > 0 else "N/A",
            "interpretation": (
                "High decentralization -- engagement is spread across many brands."
                if decentralization_score >= 60
                else "Moderate concentration -- a few brands capture disproportionate attention."
                if decentralization_score >= 30
                else "High concentration -- engagement is dominated by a small number of brands."
            ),
        }
    else:
        results["decentralization"] = {"score": 0, "interpretation": "No view data."}

    # 4. AGGREGATION: Does crowd engagement (views) reflect quality?
    # Measure: correlation between view count rank and composite score rank
    if "view_count" in df.columns:
        temp = df.copy()
        temp["view_rank"] = temp["view_count"].rank(ascending=False)
        # Compute a simple quality proxy from engagement + likes
        if "like_count" in temp.columns and "comment_count" in temp.columns:
            temp["quality"] = (
                temp["engagement_rate"].rank(pct=True) * 0.4
                + temp["like_ratio"].rank(pct=True) * 0.3
                + temp["comment_count"].rank(pct=True) * 0.3
            )
            temp["quality_rank"] = temp["quality"].rank(ascending=False)
            corr = temp["view_rank"].corr(temp["quality_rank"])
            aggregation_score = max(0, corr * 100)
        else:
            corr = 0
            aggregation_score = 50

        results["aggregation"] = {
            "score": round(aggregation_score, 1),
            "metric": f"Rank correlation (views vs quality): {corr:.3f}",
            "interpretation": (
                "Strong aggregation -- crowd views align well with multi-metric quality."
                if aggregation_score >= 60
                else "Moderate aggregation -- views partially reflect true ad quality."
                if aggregation_score >= 30
                else "Weak aggregation -- high view counts don't necessarily mean high quality."
            ),
        }
    else:
        results["aggregation"] = {"score": 0, "interpretation": "No engagement data."}

    # Overall Wisdom of Crowds score
    condition_scores = [results[k]["score"] for k in ["diversity", "independence", "decentralization", "aggregation"]]
    results["overall"] = {
        "score": round(np.mean(condition_scores), 1),
        "interpretation": (
            "The Super Bowl ad ecosystem exhibits strong crowd wisdom -- "
            "diverse, independent strategies are effectively aggregated by audience engagement."
            if np.mean(condition_scores) >= 60
            else "The Super Bowl ad ecosystem shows moderate crowd wisdom -- "
            "some conditions are met but there is room for improvement."
            if np.mean(condition_scores) >= 30
            else "The Super Bowl ad ecosystem shows limited crowd wisdom -- "
            "strategy herding and engagement concentration limit collective intelligence."
        ),
    }

    return results
