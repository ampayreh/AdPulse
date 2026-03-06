"""AdPulse - AI-Powered Ad Campaign Performance Analyzer

An extensible tool for analyzing ad campaign performance through
engagement metrics, audience sentiment, topic discovery, and search trends.
Demonstrated with Super Bowl ads data, but works with any campaign CSV.

Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx
from sklearn.manifold import TSNE

from analysis import (
    EngagementAnalyzer, SentimentAnalyzer, SentimentEvaluator,
    TopicModeler, TrendAnalyzer,
)
from data_utils import (
    DATA_DIR,
    load_comments,
    load_superbowl_data,
    preprocess_text,
)
from clustering import (
    build_tfidf_matrix,
    brand_tfidf_fingerprints,
    compute_cosine_similarity,
    compute_brand_jaccard,
    cluster_ads_kmeans,
    hierarchical_cluster_brands,
    compute_tsne,
    train_word2vec,
    find_similar_words,
    bert_sentiment_analysis,
    compare_vader_bert,
)
from network_analysis import (
    build_brand_network,
    build_ad_network,
    compute_centrality,
    centrality_summary,
    aggregate_centrality_by_brand,
    brand_network_html,
    wisdom_of_crowds_analysis,
)
from sem_analysis import (
    compute_sem_metrics,
    brand_publisher_comparison,
    keyword_strategy_analysis,
    build_engagement_predictor,
    predict_engagement,
    optimal_feature_recommendation,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AdPulse | AI Ad Campaign Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = px.colors.qualitative.Set2
SENTIMENT_COLORS = {
    "positive": "#2ca02c",
    "neutral": "#7f7f7f",
    "negative": "#d62728",
}

# ──────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_superbowl_data()


@st.cache_data
def get_comments():
    return load_comments()


@st.cache_resource
def get_sentiment_analyzer(method):
    return SentimentAnalyzer(method=method)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AdPulse")
    st.caption("AI-Powered Ad Campaign Analyzer")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Ad Rankings",
            "Feature Impact",
            "Sentiment Analysis",
            "Topic Discovery",
            "Text Analysis & Clustering",
            "Network & Crowd Intelligence",
            "SEM Performance Lab",
            "Search Trends",
            "Methodology",
            "Upload Data",
        ],
    )

    st.divider()
    st.markdown("**Filters**")

    raw_df = get_data()
    years = sorted(raw_df["year"].unique())
    year_range = st.slider(
        "Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years))),
    )

    all_brands = sorted(raw_df["brand"].unique().tolist())
    selected_brands = st.multiselect("Brands", all_brands, default=all_brands)

    st.divider()
    st.caption("Data: FiveThirtyEight / TidyTuesday")
    st.caption("Super Bowl Ads (2000-2020)")

# ──────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────
df = get_data()
df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
df = df[df["brand"].isin(selected_brands)]
comments_df = get_comments()

FEATURES = [
    "funny", "show_product_quickly", "patriotic",
    "celebrity", "danger", "animals", "use_sex",
]

# ══════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════
if page == "Dashboard":
    st.header("Campaign Dashboard")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ads", f"{len(df):,}")
    c2.metric("Brands", df["brand"].nunique())
    c3.metric("Avg Views", f"{df['view_count'].mean():,.0f}")
    c4.metric("Avg Engagement", f"{df['engagement_rate'].mean():.2%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        yearly = (
            df.groupby("year")
            .agg(avg_views=("view_count", "mean"), count=("year", "count"))
            .reset_index()
        )
        fig = px.line(
            yearly, x="year", y="avg_views",
            title="Average Views Per Ad Over Time",
            markers=True,
            labels={"avg_views": "Average Views", "year": "Year"},
        )
        fig.update_traces(line_color=COLORS[0])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        brand_views = (
            df.groupby("brand")["view_count"]
            .mean()
            .sort_values(ascending=True)
            .tail(10)
        )
        fig = px.bar(
            x=brand_views.values, y=brand_views.index,
            orientation="h",
            title="Top 10 Brands by Average Views",
            labels={"x": "Average Views", "y": ""},
            color_discrete_sequence=[COLORS[1]],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            df, x="view_count", y="like_count",
            color="brand",
            hover_data=["title_display", "year"],
            title="Views vs Likes (log scale)",
            log_x=True, log_y=True,
            labels={"view_count": "Views", "like_count": "Likes"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        feat_counts = df[FEATURES].sum().sort_values(ascending=True)
        feat_counts.index = [f.replace("_", " ").title() for f in feat_counts.index]
        fig = px.bar(
            x=feat_counts.values, y=feat_counts.index,
            orientation="h",
            title="Most Common Ad Features",
            labels={"x": "Count", "y": ""},
            color_discrete_sequence=[COLORS[2]],
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: Ad Rankings
# ══════════════════════════════════════════════
elif page == "Ad Rankings":
    st.header("Ad Performance Rankings")

    analyzer = EngagementAnalyzer(df)
    scored = analyzer.compute_composite_score()

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox(
            "Rank by",
            ["composite_score", "view_count", "like_count",
             "comment_count", "engagement_rate", "like_ratio"],
        )
    with col2:
        top_n = st.slider("Show top", 5, min(50, len(scored)), 15)

    ranked = scored.sort_values(metric, ascending=False).head(top_n)

    fig = px.bar(
        ranked, x="title_display", y=metric, color="brand",
        title=f"Top {top_n} Ads by {metric.replace('_', ' ').title()}",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "title_display", "brand", "year", "view_count",
        "like_count", "comment_count", "engagement_rate", "composite_score",
    ]
    display_cols = [c for c in display_cols if c in scored.columns]
    st.dataframe(
        ranked[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "title_display": "Ad Title",
            "engagement_rate": st.column_config.NumberColumn(
                "Engagement Rate", format="%.4f"
            ),
            "composite_score": st.column_config.NumberColumn("Score", format="%.1f"),
            "view_count": st.column_config.NumberColumn("Views", format="%d"),
            "like_count": st.column_config.NumberColumn("Likes", format="%d"),
            "comment_count": st.column_config.NumberColumn("Comments", format="%d"),
        },
    )


# ══════════════════════════════════════════════
# PAGE: Feature Impact
# ══════════════════════════════════════════════
elif page == "Feature Impact":
    st.header("Feature Impact Analysis")
    st.caption("How do ad characteristics influence engagement?")

    analyzer = EngagementAnalyzer(df)
    impact = analyzer.feature_impact()

    if impact.empty:
        st.warning("Not enough data for feature impact analysis.")
    else:
        # Grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="With Feature", x=impact["feature"],
            y=impact["avg_views_with"], marker_color=COLORS[0],
        ))
        fig.add_trace(go.Bar(
            name="Without Feature", x=impact["feature"],
            y=impact["avg_views_without"], marker_color=COLORS[3],
        ))
        fig.update_layout(
            title="Average Views: With vs Without Each Feature",
            barmode="group", xaxis_title="", yaxis_title="Average Views",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                impact, x="feature", y="views_lift",
                title="Views Lift % When Feature Present",
                color="views_lift", color_continuous_scale="RdYlGn",
                labels={"views_lift": "Lift (%)", "feature": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            feat_data = df[FEATURES].astype(int)
            corr = feat_data.corr()
            labels = [f.replace("_", " ").title() for f in FEATURES]
            fig = px.imshow(
                corr, text_auto=".2f", x=labels, y=labels,
                title="Feature Co-occurrence",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature trends over time
        st.subheader("Feature Trends Over Time")
        yearly_feat = df.groupby("year")[FEATURES].mean().reset_index()
        melted = yearly_feat.melt(
            id_vars="year", var_name="feature", value_name="proportion",
        )
        melted["feature"] = melted["feature"].str.replace("_", " ").str.title()
        fig = px.line(
            melted, x="year", y="proportion", color="feature",
            title="Proportion of Ads Using Each Feature Over Time",
            labels={"proportion": "Proportion", "year": "Year"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: Sentiment Analysis
# ══════════════════════════════════════════════
elif page == "Sentiment Analysis":
    st.header("Audience Sentiment Analysis")

    has_desc = (
        "description" in df.columns and df["description"].notna().sum() > 10
    )
    has_comments = comments_df is not None and len(comments_df) > 0

    sources = []
    if has_desc:
        sources.append("Ad Descriptions")
    if has_comments:
        sources.append("YouTube Comments")
    sources.append("Custom Text Input")

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Text Source", sources)
    with col2:
        method = st.selectbox("Method", ["VADER (Fast)", "BERT (Deep Learning)"])
    method_key = "vader" if "VADER" in method else "bert"

    results = None

    if source == "Ad Descriptions" and has_desc:
        desc_df = df[
            df["description"].notna() & (df["description"].str.strip() != "")
        ].copy()
        texts = desc_df["description"].tolist()
        brands_list = desc_df["brand"].tolist()
        with st.spinner(f"Analyzing {len(texts)} descriptions..."):
            analyzer = get_sentiment_analyzer(method_key)
            results = analyzer.analyze(texts)
            results["brand"] = brands_list[: len(results)]

    elif source == "YouTube Comments" and has_comments:
        texts = comments_df["text"].dropna().tolist()
        with st.spinner(f"Analyzing {len(texts)} comments..."):
            analyzer = get_sentiment_analyzer(method_key)
            results = analyzer.analyze(texts)
            if "brand" in comments_df.columns:
                brands = comments_df["brand"].dropna().tolist()
                results["brand"] = brands[: len(results)]

    elif source == "Custom Text Input":
        user_text = st.text_area(
            "Paste text to analyze (one entry per line)",
            height=200,
            placeholder="Enter text here...\nEach line is analyzed separately.",
        )
        if user_text and st.button("Analyze Sentiment"):
            texts = [t.strip() for t in user_text.strip().split("\n") if t.strip()]
            analyzer = get_sentiment_analyzer(method_key)
            results = analyzer.analyze(texts)

    # Display results
    if results is not None and len(results) > 0:
        st.divider()

        total = len(results)
        pos = (results["sentiment"] == "positive").sum()
        neg = (results["sentiment"] == "negative").sum()
        neu = (results["sentiment"] == "neutral").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{pos / total:.0%}", f"{pos} texts")
        c2.metric("Neutral", f"{neu / total:.0%}", f"{neu} texts")
        c3.metric("Negative", f"{neg / total:.0%}", f"{neg} texts")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                results, names="sentiment",
                title="Sentiment Distribution",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                results, x="compound", nbins=30,
                title="Sentiment Score Distribution",
                labels={"compound": "Compound Score"},
                color_discrete_sequence=[COLORS[0]],
            )
            st.plotly_chart(fig, use_container_width=True)

        if "brand" in results.columns:
            brand_sent = (
                results.groupby(["brand", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                brand_sent, x="brand", y="count", color="sentiment",
                title="Sentiment by Brand",
                color_discrete_map=SENTIMENT_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Results")
        tab_pos, tab_neu, tab_neg = st.tabs(["Positive", "Neutral", "Negative"])
        for tab, sent in [
            (tab_pos, "positive"), (tab_neu, "neutral"), (tab_neg, "negative"),
        ]:
            with tab:
                sample = results[results["sentiment"] == sent].head(5)
                if len(sample) > 0:
                    for _, row in sample.iterrows():
                        if sent == "positive":
                            st.success(row["text"])
                        elif sent == "negative":
                            st.error(row["text"])
                        else:
                            st.info(row["text"])
                else:
                    st.write(f"No {sent} results found.")

        # --- Evaluation / Sanity Checks ---
        st.divider()
        st.subheader("Output Validation")

        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            dist_check = SentimentEvaluator.distribution_check(results)
            st.markdown("**Distribution Check**")
            if dist_check["flags"]:
                for flag in dist_check["flags"]:
                    st.warning(flag)
            else:
                st.success(
                    f"Distribution looks healthy — dominant class is "
                    f"{dist_check['dominant_pct']:.0%} (below 95% threshold)."
                )

        with eval_col2:
            st.markdown("**Known-Sample Accuracy**")
            with st.spinner("Running sanity check on known examples..."):
                sample_check = SentimentEvaluator.known_sample_check(
                    get_sentiment_analyzer(method_key)
                )
            acc = sample_check["accuracy"]
            if acc >= 0.8:
                st.success(f"Accuracy on known samples: {sample_check['correct']}/{sample_check['total']} ({acc:.0%})")
            elif acc >= 0.6:
                st.warning(f"Accuracy on known samples: {sample_check['correct']}/{sample_check['total']} ({acc:.0%})")
            else:
                st.error(f"Accuracy on known samples: {sample_check['correct']}/{sample_check['total']} ({acc:.0%})")

            with st.expander("See validation details"):
                for d in sample_check["details"]:
                    icon = "+" if d["match"] else "-"
                    st.text(f"[{icon}] \"{d['text'][:50]}...\" expected={d['expected']} got={d['predicted']}")


# ══════════════════════════════════════════════
# PAGE: Topic Discovery
# ══════════════════════════════════════════════
elif page == "Topic Discovery":
    st.header("Topic Discovery")
    st.caption("Discover themes in ad content using LDA topic modeling")

    has_desc = (
        "description" in df.columns and df["description"].notna().sum() > 10
    )
    has_comments = comments_df is not None and len(comments_df) > 0

    sources = []
    if has_desc:
        sources.append("Ad Descriptions")
    if has_comments:
        sources.append("YouTube Comments")
    sources.append("Custom Text Input")

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Text Source", sources, key="topic_source")
    with col2:
        n_topics = st.slider("Number of Topics", 2, 10, 5)

    texts = None

    if source == "Ad Descriptions" and has_desc:
        raw = df[df["description"].notna()]["description"].tolist()
        texts = [preprocess_text(t) for t in raw]
        texts = [t for t in texts if len(t.split()) >= 3]
    elif source == "YouTube Comments" and has_comments:
        raw = comments_df["text"].dropna().tolist()
        texts = [preprocess_text(t) for t in raw]
        texts = [t for t in texts if len(t.split()) >= 3]
    elif source == "Custom Text Input":
        user_text = st.text_area(
            "Paste texts (one per line)", height=200, key="topic_text",
        )
        if user_text and st.button("Discover Topics"):
            raw = [t.strip() for t in user_text.strip().split("\n") if t.strip()]
            texts = [preprocess_text(t) for t in raw]
            texts = [t for t in texts if len(t.split()) >= 3]

    if texts and len(texts) >= 5:
        with st.spinner("Running topic model..."):
            modeler = TopicModeler(n_topics=n_topics)
            topic_dist = modeler.fit_transform(texts)
            top_words = modeler.get_top_words(n_words=10)

        # Top words per topic
        tabs = st.tabs(list(top_words.keys()))
        for tab, (topic_name, words) in zip(tabs, top_words.items()):
            with tab:
                word_df = pd.DataFrame(words, columns=["word", "weight"])
                word_df = word_df.sort_values("weight", ascending=True)
                topic_idx = int(topic_name.split()[-1]) % len(COLORS)
                fig = px.bar(
                    word_df, x="weight", y="word", orientation="h",
                    title=f"{topic_name} - Top Words",
                    color_discrete_sequence=[COLORS[topic_idx]],
                )
                fig.update_layout(yaxis_title="", xaxis_title="Weight")
                st.plotly_chart(fig, use_container_width=True)

        # Topic distribution
        st.subheader("Document Distribution Across Topics")
        dominant = modeler.get_dominant_topic(topic_dist)
        topic_counts = pd.Series(dominant).value_counts().sort_index()
        topic_labels = [f"Topic {i}" for i in topic_counts.index]
        fig = px.pie(
            values=topic_counts.values, names=topic_labels,
            title="How Documents Are Distributed Across Topics",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Topic quality evaluation
        st.subheader("Topic Quality Evaluation")
        distinctness = modeler.coherence_proxy(texts)
        if distinctness is not None:
            if distinctness >= 0.3:
                st.success(f"Topic distinctness score: {distinctness:.3f} — topics are well-separated.")
            elif distinctness >= 0.15:
                st.warning(f"Topic distinctness score: {distinctness:.3f} — moderate overlap between topics.")
            else:
                st.error(f"Topic distinctness score: {distinctness:.3f} — topics overlap heavily. Try fewer topics.")
            st.caption(
                "Distinctness = 1 minus average cosine similarity between topic-word distributions. "
                "Range 0-1; higher means topics are more distinct from each other."
            )

        # Word clouds
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            st.subheader("Word Clouds")
            wc_cols = st.columns(min(n_topics, 3))
            for i in range(n_topics):
                with wc_cols[i % len(wc_cols)]:
                    word_freq = {w: wt for w, wt in top_words[f"Topic {i + 1}"]}
                    wc = WordCloud(
                        width=400, height=250,
                        background_color="white", colormap="Set2",
                    ).generate_from_frequencies(word_freq)
                    fig_wc, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title(f"Topic {i + 1}", fontsize=12)
                    st.pyplot(fig_wc)
                    plt.close()
        except ImportError:
            st.info("Install `wordcloud` package for word cloud visualization.")
    elif texts is not None and len(texts) < 5:
        st.warning("Need at least 5 texts with 3+ words each for topic modeling.")


# ══════════════════════════════════════════════
# PAGE: Text Analysis & Clustering  (v2 - S2/S3)
# ══════════════════════════════════════════════
elif page == "Text Analysis & Clustering":
    st.header("Text Analysis & Clustering")
    st.caption(
        "TF-IDF, Word2Vec embeddings, BERT sentiment, cosine similarity, "
        "K-means clustering, hierarchical clustering (Sessions 2 & 3)"
    )

    # Prepare texts
    has_desc = "description" in df.columns and df["description"].notna().sum() > 10
    if not has_desc:
        st.warning("Not enough ad descriptions for text analysis.")
    else:
        desc_df = df[df["description"].notna() & (df["description"].str.strip() != "")].copy()
        raw_texts = desc_df["description"].tolist()
        processed_texts = [preprocess_text(t) for t in raw_texts]
        valid_mask = [len(t.split()) >= 2 for t in processed_texts]
        processed_texts = [t for t, v in zip(processed_texts, valid_mask) if v]
        desc_df = desc_df[valid_mask].reset_index(drop=True)

        tab_tfidf, tab_bert, tab_sim, tab_kmeans, tab_dendro = st.tabs([
            "TF-IDF Fingerprints", "BERT vs VADER", "Similarity Matrix",
            "K-Means Clusters", "Brand Dendrogram",
        ])

        # ── Tab 1: TF-IDF Brand Fingerprints ──
        with tab_tfidf:
            st.subheader("Brand TF-IDF Fingerprints")
            st.markdown(
                "TF-IDF (Term Frequency-Inverse Document Frequency) identifies the most "
                "distinctive words used by each brand in their ad descriptions."
            )
            with st.spinner("Computing TF-IDF..."):
                fingerprints = brand_tfidf_fingerprints(desc_df, n_terms=8)

            if not fingerprints.empty:
                # Pivot for heatmap
                pivot = fingerprints.pivot_table(
                    index="term", columns="brand", values="tfidf_score", fill_value=0,
                )
                # Keep top terms overall
                top_terms = fingerprints.groupby("term")["tfidf_score"].max().nlargest(20).index
                pivot = pivot.loc[pivot.index.isin(top_terms)]

                fig = px.imshow(
                    pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    title="Top TF-IDF Terms by Brand",
                    color_continuous_scale="Teal",
                    labels={"color": "TF-IDF Score"},
                    aspect="auto",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Word2Vec similar words explorer
                st.subheader("Word2Vec Similarity Explorer")
                st.markdown(
                    "Word2Vec learns word embeddings -- vector representations that capture "
                    "semantic relationships between words."
                )
                with st.spinner("Training Word2Vec model on ad descriptions..."):
                    w2v_model, doc_embeddings = train_word2vec(processed_texts)

                if w2v_model is not None:
                    query = st.text_input(
                        "Enter a word to find similar terms:",
                        value="funny",
                        key="w2v_query",
                    )
                    if query:
                        similar = find_similar_words(w2v_model, query.lower(), topn=10)
                        if similar:
                            sim_df = pd.DataFrame(similar, columns=["word", "similarity"])
                            fig = px.bar(
                                sim_df.sort_values("similarity"),
                                x="similarity", y="word", orientation="h",
                                title=f"Words Most Similar to '{query}'",
                                color="similarity", color_continuous_scale="Teal",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"'{query}' not found in vocabulary. Try another word.")

                    # t-SNE of document embeddings
                    if doc_embeddings is not None and len(doc_embeddings) >= 10:
                        st.subheader("t-SNE Document Map")
                        st.markdown(
                            "t-SNE reduces high-dimensional embeddings to 2D for visualization. "
                            "Each point is an ad, colored by brand."
                        )
                        perp = st.slider("t-SNE Perplexity", 5, 50, 15, key="tsne_w2v_perp")
                        tsne_coords = TSNE(
                            n_components=2, perplexity=min(perp, len(doc_embeddings) // 4),
                            random_state=42, max_iter=800,
                        ).fit_transform(doc_embeddings[:len(desc_df)])

                        tsne_df = pd.DataFrame({
                            "x": tsne_coords[:, 0],
                            "y": tsne_coords[:, 1],
                            "brand": desc_df["brand"].values[:len(tsne_coords)],
                            "title": desc_df["title_display"].values[:len(tsne_coords)],
                        })
                        fig = px.scatter(
                            tsne_df, x="x", y="y", color="brand",
                            hover_data=["title"],
                            title="t-SNE Map of Ad Descriptions (Word2Vec Embeddings)",
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough text data to train Word2Vec model.")
            else:
                st.warning("Not enough data for TF-IDF analysis.")

        # ── Tab 2: BERT vs VADER ──
        with tab_bert:
            st.subheader("BERT vs VADER Sentiment Comparison")
            st.markdown(
                "VADER is a rule-based sentiment analyzer. BERT is a deep learning model "
                "that understands context. Here we compare both on the same ad descriptions."
            )

            if st.button("Run BERT Analysis", key="run_bert"):
                with st.spinner("Running VADER analysis..."):
                    vader_analyzer = SentimentAnalyzer(method="vader")
                    vader_results = vader_analyzer.analyze(raw_texts[:len(desc_df)])

                with st.spinner("Running BERT analysis (this may take a minute)..."):
                    bert_results = bert_sentiment_analysis(raw_texts[:len(desc_df)])

                if bert_results is not None:
                    comparison = compare_vader_bert(vader_results, bert_results)

                    # Agreement rate
                    agree_pct = comparison["agreement"].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Agreement Rate", f"{agree_pct:.0%}")
                    c2.metric("VADER Positive %", f"{(comparison['vader_sentiment'] == 'positive').mean():.0%}")
                    c3.metric("BERT Positive %", f"{(comparison['bert_sentiment'] == 'positive').mean():.0%}")

                    # Scatter: VADER compound vs BERT compound
                    fig = px.scatter(
                        comparison, x="vader_compound", y="bert_compound",
                        color="agreement",
                        color_discrete_map={True: "#10B981", False: "#FF6B6B"},
                        hover_data=["text"],
                        title="VADER vs BERT Sentiment Scores",
                        labels={
                            "vader_compound": "VADER Compound Score",
                            "bert_compound": "BERT Compound Score",
                        },
                    )
                    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                                  line=dict(dash="dash", color="gray"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Side-by-side distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            comparison, names="vader_sentiment",
                            title="VADER Distribution",
                            color="vader_sentiment", color_discrete_map=SENTIMENT_COLORS,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.pie(
                            comparison, names="bert_sentiment",
                            title="BERT Distribution",
                            color="bert_sentiment", color_discrete_map=SENTIMENT_COLORS,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # BERT star rating distribution
                    fig = px.histogram(
                        comparison, x="bert_stars", nbins=5,
                        title="BERT Star Rating Distribution (1-5)",
                        labels={"bert_stars": "Stars"},
                        color_discrete_sequence=["#0891B2"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(
                        "BERT analysis requires the `transformers` and `torch` packages. "
                        "Install them with: `pip install transformers torch`"
                    )
            else:
                st.info("Click 'Run BERT Analysis' to compare VADER and BERT sentiment.")

        # ── Tab 3: Similarity Matrix ──
        with tab_sim:
            st.subheader("Ad Similarity Analysis")
            st.markdown(
                "Cosine similarity measures how similar two documents are based on their "
                "TF-IDF vectors. Jaccard similarity compares brands by shared features."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Cosine Similarity (Content-Based)**")
                with st.spinner("Computing TF-IDF similarity..."):
                    tfidf_mat, feat_names, _ = build_tfidf_matrix(processed_texts)
                    cos_sim = compute_cosine_similarity(tfidf_mat)

                # Show brand-level average similarity
                brands_for_sim = desc_df["brand"].values
                brand_list = sorted(desc_df["brand"].unique())
                brand_sim = np.zeros((len(brand_list), len(brand_list)))
                for i, b1 in enumerate(brand_list):
                    for j, b2 in enumerate(brand_list):
                        mask1 = brands_for_sim == b1
                        mask2 = brands_for_sim == b2
                        if mask1.any() and mask2.any():
                            brand_sim[i, j] = cos_sim[np.ix_(mask1, mask2)].mean()

                fig = px.imshow(
                    brand_sim, x=brand_list, y=brand_list,
                    title="Avg Cosine Similarity Between Brands",
                    color_continuous_scale="Blues",
                    text_auto=".2f",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Jaccard Similarity (Feature-Based)**")
                with st.spinner("Computing Jaccard similarity..."):
                    jaccard_df = compute_brand_jaccard(df)

                if not jaccard_df.empty:
                    fig = px.imshow(
                        jaccard_df.values,
                        x=jaccard_df.columns.tolist(),
                        y=jaccard_df.index.tolist(),
                        title="Jaccard Similarity of Brand Feature Profiles",
                        color_continuous_scale="Purples",
                        text_auto=".2f",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Tab 4: K-Means Clusters ──
        with tab_kmeans:
            st.subheader("K-Means Ad Clustering")
            st.markdown(
                "K-Means groups ads into clusters based on their text content and features. "
                "Ads in the same cluster share similar characteristics."
            )

            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5, key="kmeans_k")

            with st.spinner("Running K-Means clustering..."):
                tfidf_mat, _, _ = build_tfidf_matrix(processed_texts)
                labels, profiles = cluster_ads_kmeans(tfidf_mat, desc_df, n_clusters)

            # t-SNE visualization colored by cluster
            with st.spinner("Computing t-SNE projection..."):
                tsne_coords = compute_tsne(tfidf_mat, perplexity=15)

            cluster_df = pd.DataFrame({
                "x": tsne_coords[:, 0],
                "y": tsne_coords[:, 1],
                "cluster": [f"Cluster {l + 1}" for l in labels],
                "brand": desc_df["brand"].values[:len(labels)],
                "title": desc_df["title_display"].values[:len(labels)],
            })

            fig = px.scatter(
                cluster_df, x="x", y="y", color="cluster",
                hover_data=["brand", "title"],
                title="Ad Clusters (t-SNE Visualization)",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Cluster profiles
            st.subheader("Cluster Profiles")
            st.dataframe(
                profiles,
                use_container_width=True,
                column_config={
                    "cluster": "Cluster",
                    "size": st.column_config.NumberColumn("Ads", format="%d"),
                    "avg_views": st.column_config.NumberColumn("Avg Views", format="%.0f"),
                    "avg_engagement": st.column_config.NumberColumn("Avg Engagement", format="%.4f"),
                    "top_brands": "Top Brands",
                },
            )

        # ── Tab 5: Brand Dendrogram ──
        with tab_dendro:
            st.subheader("Hierarchical Brand Clustering")
            st.markdown(
                "Hierarchical clustering reveals which brands have the most similar "
                "advertising strategies. The dendrogram shows the merging order."
            )

            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

            with st.spinner("Computing hierarchical clustering..."):
                Z, brand_labels = hierarchical_cluster_brands(df)

            fig_dendro, ax = plt.subplots(figsize=(10, 5))
            scipy_dendrogram(
                Z, labels=brand_labels, ax=ax,
                leaf_rotation=45, leaf_font_size=11,
                color_threshold=0.7 * max(Z[:, 2]),
            )
            ax.set_title("Brand Strategy Dendrogram (Ward Linkage)", fontsize=14)
            ax.set_ylabel("Distance")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig_dendro)
            plt.close()

            st.caption(
                "Brands that merge at lower distances have more similar ad strategies. "
                "Ward linkage minimizes within-cluster variance at each merge step."
            )


# ══════════════════════════════════════════════
# PAGE: Network & Crowd Intelligence  (v2 - S6)
# ══════════════════════════════════════════════
elif page == "Network & Crowd Intelligence":
    st.header("Network & Crowd Intelligence")
    st.caption(
        "Brand influence networks, centrality analysis, and Wisdom of Crowds "
        "(Session 6)"
    )

    tab_network, tab_centrality, tab_woc = st.tabs([
        "Brand Strategy Network", "Centrality Analysis", "Wisdom of Crowds",
    ])

    # ── Tab 1: Brand Network ──
    with tab_network:
        st.subheader("Brand Strategy Similarity Network")
        st.markdown(
            "Each node is a brand. Edges connect brands with similar advertising strategies. "
            "Node size reflects total engagement (views). Edge thickness reflects similarity."
        )

        sim_threshold = st.slider(
            "Similarity Threshold", 0.1, 0.9, 0.3, 0.05,
            help="Only show edges between brands with similarity above this threshold.",
            key="brand_net_threshold",
        )

        with st.spinner("Building brand network..."):
            G_brand, brand_profiles = build_brand_network(df, sim_threshold)

        if len(G_brand.nodes()) > 0:
            html = brand_network_html(G_brand, height="500px")
            if html:
                import streamlit.components.v1 as components
                components.html(html, height=520, scrolling=False)
            else:
                st.info("Install `pyvis` for interactive network visualization: `pip install pyvis`")

            # Network stats
            st.markdown("**Network Statistics**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nodes", len(G_brand.nodes()))
            c2.metric("Edges", len(G_brand.edges()))
            density = nx.density(G_brand) if len(G_brand.nodes()) > 1 else 0
            c3.metric("Density", f"{density:.2f}")
            components_count = nx.number_connected_components(G_brand)
            c4.metric("Components", components_count)
        else:
            st.warning("No brands in filtered data to build network.")

    # ── Tab 2: Centrality Analysis ──
    with tab_centrality:
        st.subheader("Centrality Analysis")
        st.markdown(
            "Centrality measures identify the most important or influential nodes in a network. "
            "We compute four key metrics from the ad content similarity network."
        )

        # Build ad network for centrality
        has_desc = "description" in df.columns and df["description"].notna().sum() > 10
        if has_desc:
            desc_df_cent = df[df["description"].notna() & (df["description"].str.strip() != "")].copy()
            proc_texts = [preprocess_text(t) for t in desc_df_cent["description"].tolist()]
            valid = [len(t.split()) >= 2 for t in proc_texts]
            proc_texts = [t for t, v in zip(proc_texts, valid) if v]
            desc_df_cent = desc_df_cent[valid].reset_index(drop=True)

            if len(proc_texts) >= 10:
                with st.spinner("Building ad similarity network and computing centrality..."):
                    tfidf_mat, _, _ = build_tfidf_matrix(proc_texts)
                    G_ads = build_ad_network(desc_df_cent, tfidf_mat, similarity_threshold=0.15)
                    cent = compute_centrality(G_ads)
                    brand_cent = aggregate_centrality_by_brand(G_ads, cent)

                if not brand_cent.empty:
                    st.markdown("**Average Centrality by Brand**")

                    metric_names = {
                        "degree": "Degree Centrality",
                        "betweenness": "Betweenness Centrality",
                        "closeness": "Closeness Centrality",
                        "pagerank": "PageRank",
                    }
                    metric_descriptions = {
                        "degree": "How many connections a brand's ads have (popularity).",
                        "betweenness": "How often a brand's ads lie on shortest paths (bridge role).",
                        "closeness": "How close a brand's ads are to all others (accessibility).",
                        "pagerank": "Influence score based on the quality of connections.",
                    }

                    for metric_key, metric_label in metric_names.items():
                        if metric_key in brand_cent.columns:
                            sorted_df = brand_cent.sort_values(metric_key, ascending=True)
                            fig = px.bar(
                                sorted_df, x=metric_key, y="brand",
                                orientation="h",
                                title=metric_label,
                                color=metric_key, color_continuous_scale="Teal",
                            )
                            fig.update_layout(yaxis_title="", xaxis_title=metric_label)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(metric_descriptions[metric_key])

                    # Top bridge and influencer ads
                    cent_summary = centrality_summary(G_ads, cent)
                    st.subheader("Notable Ads")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Bridge Ads** (highest betweenness)")
                        top_bridges = cent_summary.nlargest(5, "betweenness")
                        for _, row in top_bridges.iterrows():
                            st.write(f"- {row.get('title', 'N/A')} ({row.get('brand', '')}) -- {row['betweenness']:.4f}")
                    with col2:
                        st.markdown("**Influencer Ads** (highest PageRank)")
                        top_pr = cent_summary.nlargest(5, "pagerank")
                        for _, row in top_pr.iterrows():
                            st.write(f"- {row.get('title', 'N/A')} ({row.get('brand', '')}) -- {row['pagerank']:.4f}")
            else:
                st.warning("Need at least 10 ads with descriptions for centrality analysis.")
        else:
            st.warning("No descriptions available for network analysis.")

    # ── Tab 3: Wisdom of Crowds ──
    with tab_woc:
        st.subheader("Wisdom of Crowds Analysis")
        st.markdown(
            "James Surowiecki identified four conditions for crowd wisdom: **Diversity**, "
            "**Independence**, **Decentralization**, and **Aggregation**. We evaluate whether "
            "the Super Bowl ad ecosystem meets these conditions."
        )

        with st.spinner("Evaluating crowd wisdom conditions..."):
            woc = wisdom_of_crowds_analysis(df)

        # Overall score
        overall = woc.get("overall", {})
        st.metric("Overall Wisdom Score", f"{overall.get('score', 0):.0f} / 100")
        st.markdown(f"*{overall.get('interpretation', '')}*")

        st.divider()

        # Four condition cards
        conditions = [
            ("diversity", "Diversity of Opinion", "Are ad strategies diverse?"),
            ("independence", "Independence", "Do brands make choices independently?"),
            ("decentralization", "Decentralization", "Is engagement spread across brands?"),
            ("aggregation", "Aggregation", "Does crowd engagement reflect quality?"),
        ]

        cols = st.columns(2)
        for i, (key, title, question) in enumerate(conditions):
            with cols[i % 2]:
                data = woc.get(key, {})
                score = data.get("score", 0)
                color = "#10B981" if score >= 60 else ("#F59E0B" if score >= 30 else "#FF6B6B")

                st.markdown(f"### {title}")
                st.markdown(f"*{question}*")
                st.metric("Score", f"{score:.0f} / 100")
                if "metric" in data:
                    st.caption(data["metric"])
                if "top_brand_share" in data:
                    st.caption(f"Top brand share: {data['top_brand_share']}")
                if "unique_strategies" in data:
                    st.caption(f"Unique strategies: {data['unique_strategies']}")
                st.markdown(f"**{data.get('interpretation', '')}**")
                st.divider()


# ══════════════════════════════════════════════
# PAGE: SEM Performance Lab  (v2 - S5)
# ══════════════════════════════════════════════
elif page == "SEM Performance Lab":
    st.header("SEM Performance Lab")
    st.caption(
        "Search Engine Marketing metrics applied to Super Bowl ads "
        "(Session 5 - inspired by the Air France case study)"
    )
    st.info(
        "This module re-frames Super Bowl ad performance through an SEM lens. "
        "Since the dataset lacks actual ad spend data, we use industry benchmark "
        "estimates for Super Bowl ad costs to compute digital marketing equivalence metrics."
    )

    tab_metrics, tab_publisher, tab_keywords = st.tabs([
        "SEM Metrics Dashboard", "Brand Publisher Comparison", "Keyword Strategy",
    ])

    # Compute SEM metrics
    with st.spinner("Computing SEM metrics..."):
        sem_df = compute_sem_metrics(df)

    # ── Tab 1: SEM Metrics Dashboard ──
    with tab_metrics:
        st.subheader("SEM-Equivalent Metrics")

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg CPV", f"${sem_df['cpv'].mean():.2f}", help="Cost Per View")
        c2.metric("Avg CPE", f"${sem_df['cpe'].mean():.0f}", help="Cost Per Engagement")
        c3.metric("Avg Quality Score", f"{sem_df['quality_score'].mean():.1f}/10")
        c4.metric("Avg Digital CTR", f"{sem_df['digital_ctr'].mean():.4%}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            # CPV by brand
            brand_cpv = sem_df.groupby("brand")["cpv"].mean().sort_values()
            fig = px.bar(
                x=brand_cpv.values, y=brand_cpv.index,
                orientation="h",
                title="Average Cost Per View by Brand (Lower = More Efficient)",
                labels={"x": "Cost Per View ($)", "y": ""},
                color_discrete_sequence=["#0891B2"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Quality Score by brand
            brand_qs = sem_df.groupby("brand")["quality_score"].mean().sort_values()
            fig = px.bar(
                x=brand_qs.values, y=brand_qs.index,
                orientation="h",
                title="Average Quality Score by Brand (Higher = Better)",
                labels={"x": "Quality Score (0-10)", "y": ""},
                color_discrete_sequence=["#10B981"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # ROI over time
        yearly_roi = sem_df.groupby("year")["engagement_roi"].mean().reset_index()
        fig = px.line(
            yearly_roi, x="year", y="engagement_roi",
            title="Engagement ROI Over Time (Engagement Score per $M Spent)",
            markers=True,
            labels={"engagement_roi": "ROI", "year": "Year"},
        )
        fig.update_traces(line_color="#0891B2")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Publisher Comparison ──
    with tab_publisher:
        st.subheader("Brand Publisher Comparison")
        st.markdown(
            "Inspired by the **Air France case study**, this table ranks brands like "
            "SEM publishers (Google, Yahoo, MSN) by cost efficiency and quality metrics."
        )

        with st.spinner("Computing publisher comparison..."):
            pub_df = brand_publisher_comparison(sem_df)

        if not pub_df.empty:
            display_cols = ["brand", "num_ads", "avg_cpv", "avg_cpe",
                            "avg_roi", "avg_quality_score", "avg_digital_ctr"]
            available_cols = [c for c in display_cols if c in pub_df.columns]

            st.dataframe(
                pub_df[available_cols].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "brand": "Brand (Publisher)",
                    "num_ads": st.column_config.NumberColumn("Ads", format="%d"),
                    "avg_cpv": st.column_config.NumberColumn("Avg CPV ($)", format="%.2f"),
                    "avg_cpe": st.column_config.NumberColumn("Avg CPE ($)", format="%.0f"),
                    "avg_roi": st.column_config.NumberColumn("Avg ROI", format="%.2f"),
                    "avg_quality_score": st.column_config.NumberColumn("Quality Score", format="%.1f"),
                    "avg_digital_ctr": st.column_config.NumberColumn("Digital CTR", format="%.4f"),
                },
            )

            # Scatter: CPV vs Quality Score
            fig = px.scatter(
                pub_df, x="avg_cpv", y="avg_quality_score",
                size="num_ads", color="avg_roi",
                text="brand",
                title="Brand Efficiency Map: CPV vs Quality Score",
                labels={
                    "avg_cpv": "Avg Cost Per View ($)",
                    "avg_quality_score": "Avg Quality Score",
                    "avg_roi": "ROI",
                },
                color_continuous_scale="RdYlGn",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Ideal position: bottom-right (low cost, high quality). "
                "Bubble size = number of ads. Color = engagement ROI."
            )

            # Recommendation
            rec = optimal_feature_recommendation(df)
            if rec.get("recommended_features"):
                st.divider()
                st.subheader("Strategy Recommendation")
                st.success(rec["insight"])

    # ── Tab 3: Keyword Strategy ──
    with tab_keywords:
        st.subheader("Keyword Strategy Analysis")
        st.markdown(
            "This analysis extracts key terms from ad descriptions and categorizes them "
            "into keyword buckets -- a core SEM concept for campaign optimization."
        )

        with st.spinner("Analyzing keyword strategies..."):
            kw_df, cat_perf = keyword_strategy_analysis(df)

        if not kw_df.empty:
            # Top keywords by brand
            brand_sel = st.selectbox(
                "Select brand",
                sorted(kw_df["brand"].unique()),
                key="kw_brand",
            )
            brand_kw = kw_df[kw_df["brand"] == brand_sel].sort_values("tfidf_score", ascending=True).tail(15)
            fig = px.bar(
                brand_kw, x="tfidf_score", y="term",
                color="category", orientation="h",
                title=f"Top Keywords for {brand_sel}",
                labels={"tfidf_score": "TF-IDF Score", "term": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Category distribution across brands
            if not cat_perf.empty:
                cat_summary = cat_perf.groupby("category")["term_count"].sum().reset_index()
                fig = px.pie(
                    cat_summary, names="category", values="term_count",
                    title="Keyword Category Distribution Across All Brands",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough description data for keyword analysis.")


# ══════════════════════════════════════════════
# PAGE: Methodology  (v2)
# ══════════════════════════════════════════════
elif page == "Methodology":
    st.header("Methodology & Course Concept Mapping")
    st.caption("How each analysis technique maps to MSIS 521 course sessions")

    st.markdown("""
    AdPulse demonstrates the practical application of AI/ML techniques taught in
    MSIS 521: IT and Marketing in the New Economy. Below is a mapping of every
    technique used in this project to its corresponding course session.
    """)

    methodology_data = [
        {"Session": "S1", "Topic": "Web Analytics", "Technique": "Google Trends Integration",
         "AdPulse Page": "Search Trends", "Description": "Real-time search interest tracking for ad brands using Google Trends API"},
        {"Session": "S2", "Topic": "Word Embedding", "Technique": "TF-IDF Vectorization",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Transform ad descriptions into weighted term vectors for comparison"},
        {"Session": "S2", "Topic": "Word Embedding", "Technique": "Word2Vec (Skip-gram)",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Learn semantic word embeddings from ad corpus; find similar terms"},
        {"Session": "S2", "Topic": "Word Embedding", "Technique": "t-SNE Visualization",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Reduce high-dimensional embeddings to 2D scatter plots"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "VADER Sentiment",
         "AdPulse Page": "Sentiment Analysis", "Description": "Rule-based sentiment scoring using lexicon approach"},
        {"Session": "S3", "Topic": "Text Analysis / LLMs", "Technique": "BERT Sentiment",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Deep learning contextual sentiment (nlptown model from A2)"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "Cosine Similarity",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Measure content similarity between ads using TF-IDF vectors"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "Jaccard Similarity",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Compare brand feature profiles using set overlap"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "K-Means Clustering",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Group ads into clusters by content + features"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "Hierarchical Clustering",
         "AdPulse Page": "Text Analysis & Clustering", "Description": "Ward linkage dendrogram showing brand strategy relationships"},
        {"Session": "S3", "Topic": "Text Analysis", "Technique": "LDA Topic Modeling",
         "AdPulse Page": "Topic Discovery", "Description": "Discover latent themes in ad descriptions"},
        {"Session": "S5", "Topic": "SEM/SEO", "Technique": "CPV / CPE / ROA Metrics",
         "AdPulse Page": "SEM Performance Lab", "Description": "SEM-equivalent cost efficiency metrics using industry benchmarks"},
        {"Session": "S5", "Topic": "SEM/SEO", "Technique": "Quality Score Proxy",
         "AdPulse Page": "SEM Performance Lab", "Description": "Composite quality metric inspired by Google Ads Quality Score"},
        {"Session": "S5", "Topic": "SEM/SEO", "Technique": "Keyword Strategy",
         "AdPulse Page": "SEM Performance Lab", "Description": "TF-IDF keyword extraction and categorization by brand"},
        {"Session": "S5", "Topic": "SEM/SEO", "Technique": "Publisher Comparison",
         "AdPulse Page": "SEM Performance Lab", "Description": "Rank brands like publishers (Air France case study approach)"},
        {"Session": "S6", "Topic": "Social Networks", "Technique": "NetworkX Graph Analysis",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Build and analyze brand/ad similarity networks"},
        {"Session": "S6", "Topic": "Social Networks", "Technique": "Degree Centrality",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Measure how connected each brand/ad is"},
        {"Session": "S6", "Topic": "Social Networks", "Technique": "Betweenness Centrality",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Identify bridge ads connecting different clusters"},
        {"Session": "S6", "Topic": "Social Networks", "Technique": "Closeness Centrality",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Measure proximity to all other ads in the network"},
        {"Session": "S6", "Topic": "Social Networks", "Technique": "PageRank",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Identify influential ads based on connection quality"},
        {"Session": "S6", "Topic": "Wisdom of Crowds", "Technique": "Surowiecki's 4 Conditions",
         "AdPulse Page": "Network & Crowd Intelligence", "Description": "Evaluate diversity, independence, decentralization, aggregation"},
    ]

    method_df = pd.DataFrame(methodology_data)

    # Summary stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Techniques Used", len(method_df))
    c2.metric("Course Sessions Covered", method_df["Session"].nunique())
    c3.metric("Dashboard Pages", method_df["AdPulse Page"].nunique())

    st.divider()

    # Filter by session
    session_filter = st.multiselect(
        "Filter by Session",
        sorted(method_df["Session"].unique()),
        default=sorted(method_df["Session"].unique()),
        key="method_session_filter",
    )
    filtered = method_df[method_df["Session"].isin(session_filter)]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Session": st.column_config.TextColumn("Session", width="small"),
            "Topic": st.column_config.TextColumn("Topic", width="medium"),
            "Technique": st.column_config.TextColumn("Technique", width="medium"),
            "AdPulse Page": st.column_config.TextColumn("Page", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    # Output validation summary
    st.divider()
    st.subheader("Output Validation Methods")
    st.markdown("""
    | Validation | Method | Page |
    |-----------|--------|------|
    | Sentiment distribution check | Flag if >95% single class | Sentiment Analysis |
    | Known-sample accuracy | 5 test cases with expected labels | Sentiment Analysis |
    | Topic distinctness score | Cosine distance between topic distributions | Topic Discovery |
    | BERT vs VADER comparison | Agreement rate + score correlation | Text Analysis & Clustering |
    | Wisdom of Crowds scoring | 4 condition metrics with interpretations | Network & Crowd Intelligence |
    | SEM metric benchmarking | Industry cost estimates + ROI computation | SEM Performance Lab |
    """)


# ══════════════════════════════════════════════
# PAGE: Search Trends
# ══════════════════════════════════════════════
elif page == "Search Trends":
    st.header("Search Trends Analysis")
    st.caption("Explore Google search interest for ad brands")

    available_brands = sorted(df["brand"].unique().tolist())
    default_sel = available_brands[:3] if len(available_brands) >= 3 else available_brands
    selected = st.multiselect(
        "Select brands to compare", available_brands, default=default_sel,
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["Past 12 months", "Past 3 months", "Past 5 years"],
    )
    tf_map = {
        "Past 12 months": "today 12-m",
        "Past 3 months": "today 3-m",
        "Past 5 years": "today 5-y",
    }

    if selected and st.button("Fetch Trends"):
        with st.spinner("Fetching Google Trends data..."):
            trends = TrendAnalyzer.get_search_trends(
                selected, timeframe=tf_map[timeframe],
            )

        if trends is not None and not trends.empty:
            fig = px.line(
                trends, title="Google Search Interest Over Time",
                labels={"value": "Interest (0-100)", "date": "Date"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Related Queries")
            for brand in selected[:3]:
                related = TrendAnalyzer.get_related_queries(brand)
                if related:
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        top_q = related.get("top")
                        if top_q is not None:
                            st.markdown(f"**{brand} - Top Related**")
                            st.dataframe(
                                top_q.head(10), use_container_width=True,
                            )
                    with rcol2:
                        rising_q = related.get("rising")
                        if rising_q is not None:
                            st.markdown(f"**{brand} - Rising**")
                            st.dataframe(
                                rising_q.head(10), use_container_width=True,
                            )
        else:
            st.warning(
                "Could not fetch trends. Google may be rate-limiting. "
                "Try again in a few minutes."
            )
    elif not selected:
        st.info("Select at least one brand above.")


# ══════════════════════════════════════════════
# PAGE: Upload Data
# ══════════════════════════════════════════════
elif page == "Upload Data":
    st.header("Upload Custom Campaign Data")
    st.caption("Analyze any ad campaign — not just Super Bowl")

    st.markdown(
        """
        Upload a CSV with your campaign data. Useful columns include:
        - **brand** — Advertiser name
        - **title** or **name** — Ad title
        - **view_count**, **like_count**, **comment_count** — Engagement metrics
        - **description** or **text** — Text content for NLP analysis

        Any additional columns will be preserved.
        """
    )

    uploaded = st.file_uploader("Upload Campaign CSV", type=["csv"])

    if uploaded:
        try:
            custom_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(custom_df)} rows, {len(custom_df.columns)} columns")
            st.dataframe(custom_df.head(10), use_container_width=True)

            if st.button("Run Quick Analysis"):
                # Numeric distributions
                numeric_cols = custom_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.subheader("Metric Distributions")
                    chart_cols = st.columns(min(len(numeric_cols), 3))
                    for i, col in enumerate(numeric_cols[:6]):
                        with chart_cols[i % len(chart_cols)]:
                            fig = px.histogram(
                                custom_df, x=col,
                                title=f"Distribution: {col}",
                                color_discrete_sequence=[COLORS[i % len(COLORS)]],
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # Text analysis
                text_cols = [
                    c for c in custom_df.columns
                    if custom_df[c].dtype == "object"
                    and custom_df[c].dropna().str.len().mean() > 50
                ]
                if text_cols:
                    st.subheader("Sentiment Analysis")
                    text_col = st.selectbox("Select text column", text_cols)
                    texts = custom_df[text_col].dropna().tolist()
                    sa = get_sentiment_analyzer("vader")
                    sent_results = sa.analyze(texts)
                    fig = px.pie(
                        sent_results, names="sentiment",
                        title=f"Sentiment Distribution ({text_col})",
                        color="sentiment",
                        color_discrete_map=SENTIMENT_COLORS,
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Upload comments
    st.divider()
    st.subheader("Upload YouTube Comments (Optional)")
    st.markdown(
        "Upload a CSV with columns: `text`, `brand` (optional), `video_id` (optional)"
    )
    comments_upload = st.file_uploader(
        "Upload Comments CSV", type=["csv"], key="comments_upload",
    )
    if comments_upload:
        try:
            comments_custom = pd.read_csv(comments_upload)
            out_path = os.path.join(DATA_DIR, "youtube_comments.csv")
            comments_custom.to_csv(out_path, index=False)
            st.success(
                f"Saved {len(comments_custom)} comments. "
                "Reload the app to use them in Sentiment/Topic pages."
            )
        except Exception as e:
            st.error(f"Error: {e}")
