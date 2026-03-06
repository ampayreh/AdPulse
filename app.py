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
            "Search Trends",
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
