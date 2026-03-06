"""Generate a team explanation PDF for AdPulse v2.

Creates a document with both technical and layman's explanations
of the AdPulse project for team members.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "AdPulse_v2_Team_Guide.pdf")

# Colors
NAVY = HexColor("#1A1F36")
TEAL = HexColor("#0891B2")
CORAL = HexColor("#FF6B6B")
PURPLE = HexColor("#8B5CF6")
GREEN = HexColor("#059669")
DARK = HexColor("#1E293B")
MID = HexColor("#64748B")
LIGHT_BG = HexColor("#F1F5F9")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=26,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=13,
        textColor=MID, alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        "SectionHeader", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=18,
        textColor=TEAL, spaceBefore=20, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "SubHeader", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=14,
        textColor=NAVY, spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11,
        textColor=DARK, alignment=TA_JUSTIFY,
        spaceAfter=8, leading=15,
    ))
    styles.add(ParagraphStyle(
        "BulletText", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11,
        textColor=DARK, alignment=TA_LEFT,
        leftIndent=20, spaceAfter=4, leading=14,
        bulletIndent=8, bulletFontSize=11,
    ))
    styles.add(ParagraphStyle(
        "TechLabel", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=11,
        textColor=PURPLE, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=10,
        textColor=MID, alignment=TA_CENTER, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=DARK, leading=12,
    ))
    styles.add(ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9,
        textColor=HexColor("#FFFFFF"), leading=12,
    ))
    return styles


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = build_styles()
    story = []

    # ── TITLE PAGE ──────────────────────────────
    story.append(Spacer(1, 60))
    story.append(Paragraph("AdPulse v2", styles["DocTitle"]))
    story.append(Paragraph("AI-Powered Ad Campaign Performance Analyzer", styles["DocSubtitle"]))
    story.append(HRFlowable(width="40%", thickness=2, color=TEAL, spaceAfter=12))
    story.append(Paragraph("Team Explanation Guide", styles["DocSubtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Team 6 -- Purple Cohort<br/>"
        "Graeme Ampeire | Soumya Thareja | Tiffany Nakamitsu | Akanksha Singh",
        styles["Note"],
    ))
    story.append(Paragraph(
        "MSIS 521 -- IT and Marketing in the New Economy<br/>"
        "Winter 2025 | University of Washington",
        styles["Note"],
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "This guide explains the AdPulse project in two ways: a plain-language overview "
        "that any team member can reference for the presentation, and a deeper technical "
        "breakdown of every module, algorithm, and course concept we implemented.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════
    # PART 1: LAYMAN'S EXPLANATION
    # ══════════════════════════════════════════════
    story.append(Paragraph("PART 1: Plain-Language Overview", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=12))

    story.append(Paragraph("What Is AdPulse?", styles["SubHeader"]))
    story.append(Paragraph(
        "AdPulse is an interactive web dashboard that analyzes Super Bowl commercials "
        "using artificial intelligence. Think of it as a smart assistant that watches "
        "all 247 Super Bowl ads from 2000 to 2020 and tells you which ones worked best, "
        "why they worked, and what patterns brands follow.",
        styles["Body"],
    ))

    story.append(Paragraph("What Problem Does It Solve?", styles["SubHeader"]))
    story.append(Paragraph(
        "Super Bowl ads cost over $5 million for 30 seconds. Brands spend enormous "
        "budgets but often rely on gut feeling to decide what makes a good ad. AdPulse "
        "replaces guesswork with data. It looks at real YouTube engagement (views, likes, "
        "comments) and uses AI to find patterns that humans might miss.",
        styles["Body"],
    ))

    story.append(Paragraph("What Does It Actually Do?", styles["SubHeader"]))
    items = [
        "<b>Ranks every ad</b> by a composite engagement score combining views, likes, and comments",
        "<b>Analyzes sentiment</b> of ad descriptions using two different AI methods (VADER and BERT) to determine if the tone is positive, negative, or neutral",
        "<b>Discovers hidden topics</b> in ad copy using topic modeling, showing what themes brands talk about",
        "<b>Groups similar ads together</b> using clustering algorithms, revealing which brands have similar ad strategies (for example, beer brands all advertise similarly)",
        "<b>Maps brand relationships</b> as a visual network graph where connected brands share similar approaches",
        "<b>Evaluates crowd wisdom</b> by testing whether the Super Bowl ad ecosystem meets the scientific conditions for collective intelligence",
        "<b>Computes marketing efficiency metrics</b> like cost per view and return on investment, ranking brands the same way the Air France SEM case study ranked search publishers",
        "<b>Tracks search trends</b> by pulling Google Trends data to see how ad buzz translates to real search interest",
    ]
    for item in items:
        story.append(Paragraph(f"&#8226; {item}", styles["BulletText"]))

    story.append(Paragraph("How Do You Use It?", styles["SubHeader"]))
    story.append(Paragraph(
        "Run one command in the terminal (<font face='Courier'>streamlit run app.py</font>) and a web "
        "browser opens with an 11-page interactive dashboard. You can filter by brand, "
        "year, or ad features. Every chart is interactive -- you can hover for details, zoom, "
        "and pan. No coding required to explore the data once the app is running.",
        styles["Body"],
    ))

    story.append(Paragraph("Key Findings (Good for the Presentation)", styles["SubHeader"]))
    findings = [
        "<b>Show your product early.</b> Ads that display the product quickly get 70% more views than those that don't. Clarity beats cleverness.",
        "<b>Celebrities are overrated.</b> Despite costing more, celebrity ads actually average 54% fewer views. The story and product matter more than the star.",
        "<b>Brands copy each other.</b> Our clustering analysis shows beer brands (Bud Light, Budweiser) and car brands (Toyota, Kia, Hyundai) follow nearly identical ad strategies year after year.",
        "<b>The crowd is only somewhat wise.</b> We scored the Super Bowl ad ecosystem at 41 out of 100 on the Wisdom of Crowds framework. Brands show herding behavior rather than true strategic diversity.",
        "<b>Efficiency varies wildly.</b> The cost per YouTube view ranges from under $1 to over $50 across brands, revealing massive differences in how well each brand converts TV exposure into digital engagement.",
    ]
    for f in findings:
        story.append(Paragraph(f"&#8226; {f}", styles["BulletText"]))

    story.append(Paragraph("How Many Course Concepts Does It Cover?", styles["SubHeader"]))
    story.append(Paragraph(
        "AdPulse implements 21 techniques from 5 of our 6 course sessions. These range "
        "from Google Trends (Session 1) and TF-IDF word embeddings (Session 2) to BERT "
        "deep learning and K-Means clustering (Session 3), SEM metrics like CPV and Quality "
        "Score (Session 5), and network centrality analysis with Wisdom of Crowds (Session 6). "
        "The Methodology page in the app has a full table mapping each technique to its course session.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════
    # PART 2: TECHNICAL EXPLANATION
    # ══════════════════════════════════════════════
    story.append(Paragraph("PART 2: Technical Deep Dive", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=12))

    # --- Project Architecture ---
    story.append(Paragraph("Project Architecture", styles["SubHeader"]))
    story.append(Paragraph(
        "AdPulse is a Python application built on the Streamlit framework. The codebase "
        "is organized into modular Python files, each handling a specific analysis domain. "
        "The main entry point (<font face='Courier'>app.py</font>, ~1,500 lines) orchestrates "
        "an 11-page dashboard that imports from five analysis modules.",
        styles["Body"],
    ))

    # File structure table
    file_data = [
        [Paragraph("<b>File</b>", styles["TableHeader"]),
         Paragraph("<b>Lines</b>", styles["TableHeader"]),
         Paragraph("<b>Purpose</b>", styles["TableHeader"])],
        [Paragraph("app.py", styles["TableCell"]),
         Paragraph("~1,530", styles["TableCell"]),
         Paragraph("Streamlit dashboard, 11 pages, all visualizations", styles["TableCell"])],
        [Paragraph("analysis.py", styles["TableCell"]),
         Paragraph("~270", styles["TableCell"]),
         Paragraph("VADER sentiment, LDA topics, engagement scoring, Google Trends", styles["TableCell"])],
        [Paragraph("clustering.py", styles["TableCell"]),
         Paragraph("~290", styles["TableCell"]),
         Paragraph("TF-IDF, Word2Vec, t-SNE, cosine/Jaccard similarity, K-Means, hierarchical clustering, BERT", styles["TableCell"])],
        [Paragraph("network_analysis.py", styles["TableCell"]),
         Paragraph("~365", styles["TableCell"]),
         Paragraph("Brand/ad networks, 4 centrality measures, Pyvis visualization, Wisdom of Crowds", styles["TableCell"])],
        [Paragraph("sem_analysis.py", styles["TableCell"]),
         Paragraph("~300", styles["TableCell"]),
         Paragraph("SEM metrics (CPV, CPE, ROI), Quality Score proxy, keyword strategy, publisher comparison", styles["TableCell"])],
        [Paragraph("data_utils.py", styles["TableCell"]),
         Paragraph("~115", styles["TableCell"]),
         Paragraph("Data loading, cleaning, text preprocessing with NLTK", styles["TableCell"])],
    ]
    t = Table(file_data, colWidths=[1.3 * inch, 0.7 * inch, 4.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#F8FAFC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # --- Dataset ---
    story.append(Paragraph("Dataset", styles["SubHeader"]))
    story.append(Paragraph(
        "We use the FiveThirtyEight/TidyTuesday Super Bowl Ads dataset: 247 ads from "
        "10 major brands spanning 2000-2020. Each row contains 25 columns including 7 boolean "
        "ad characteristics (funny, celebrity, patriotic, danger, animals, use_sex, "
        "show_product_quickly), YouTube engagement metrics (views, likes, dislikes, comments, "
        "favorites), and video metadata (title, description, thumbnail URL, publish date). "
        "Of the 247 ads, 207 have non-null descriptions suitable for NLP analysis.",
        styles["Body"],
    ))

    # --- Module 1: Engagement & Feature Impact ---
    story.append(Paragraph("Module: Engagement & Feature Impact (v1)", styles["SubHeader"]))
    story.append(Paragraph(
        "The EngagementAnalyzer class normalizes view_count, like_count, and comment_count "
        "to 0-1 range, then computes a weighted composite score (40% views, 35% likes, "
        "25% comments). Feature impact analysis compares mean views for ads with vs. without "
        "each boolean feature, computing lift percentages. This reveals that "
        "show_product_quickly delivers +70% views lift while celebrity usage shows -54%.",
        styles["Body"],
    ))

    # --- Module 2: Sentiment Analysis ---
    story.append(Paragraph("Module: Sentiment Analysis (v1 + v2 BERT)", styles["SubHeader"]))
    story.append(Paragraph(
        "v1 uses NLTK's VADER (Valence Aware Dictionary for sEntiment Reasoning), a "
        "rule-based sentiment analyzer that outputs compound, positive, negative, and neutral "
        "scores. v2 adds optional BERT sentiment using the "
        "<font face='Courier'>nlptown/bert-base-multilingual-uncased-sentiment</font> model, "
        "which classifies text into 1-5 star ratings. The compare_vader_bert() function "
        "aligns both scales and computes agreement rate. A SentimentEvaluator validates "
        "outputs: it checks distribution skew (flags if >95% single class) and tests 5 "
        "known-sentiment samples for 80%+ accuracy.",
        styles["Body"],
    ))

    # --- Module 3: Topic Modeling ---
    story.append(Paragraph("Module: LDA Topic Modeling (v1)", styles["SubHeader"]))
    story.append(Paragraph(
        "The TopicModeler uses gensim's LDA (Latent Dirichlet Allocation) to discover "
        "latent themes in ad descriptions. Text preprocessing includes lowercasing, "
        "tokenization, stopword removal, and lemmatization via NLTK. The model extracts "
        "5 topics, each represented by its top 10 weighted terms. Topic distinctness is "
        "measured by average pairwise cosine distance between topic vectors (score: 0.312). "
        "Interactive word clouds and topic-document distributions are rendered in Plotly.",
        styles["Body"],
    ))

    # --- Module 4: Text Analysis & Clustering ---
    story.append(Paragraph("Module: Text Analysis & Clustering (v2 -- Sessions 2 & 3)", styles["SubHeader"]))

    story.append(Paragraph("TF-IDF Vectorization", styles["TechLabel"]))
    story.append(Paragraph(
        "scikit-learn's TfidfVectorizer transforms 207 ad descriptions into a "
        "term-frequency inverse-document-frequency matrix (193 docs x 500 features after "
        "filtering). Per-brand fingerprints show each brand's most distinctive vocabulary. "
        "Top terms are extracted by averaging TF-IDF scores across each brand's ads.",
        styles["Body"],
    ))

    story.append(Paragraph("Word2Vec Embeddings", styles["TechLabel"]))
    story.append(Paragraph(
        "gensim's Word2Vec trains on the tokenized ad corpus (vocab ~800 words, "
        "100-dimensional vectors, window=5, min_count=2). Document embeddings are computed "
        "by averaging constituent word vectors. The dashboard includes a word similarity "
        "explorer where users enter a term and see semantically related words.",
        styles["Body"],
    ))

    story.append(Paragraph("t-SNE Visualization", styles["TechLabel"]))
    story.append(Paragraph(
        "scikit-learn's t-SNE (t-distributed Stochastic Neighbor Embedding) projects "
        "the high-dimensional Word2Vec document embeddings down to 2D for visualization. "
        "The resulting scatter plot colors points by brand, revealing spatial clusters "
        "of similar ad content.",
        styles["Body"],
    ))

    story.append(Paragraph("Cosine & Jaccard Similarity", styles["TechLabel"]))
    story.append(Paragraph(
        "Cosine similarity measures textual similarity between all ad pairs using TF-IDF "
        "vectors (rendered as an interactive heatmap). Jaccard similarity compares brands "
        "by their boolean feature overlap: |intersection| / |union| of feature usage patterns.",
        styles["Body"],
    ))

    story.append(Paragraph("K-Means Clustering", styles["TechLabel"]))
    story.append(Paragraph(
        "K-Means groups ads into k clusters based on combined feature vectors (TF-IDF + "
        "boolean features + normalized engagement). The dashboard lets users select k (2-10) "
        "and displays cluster profiles showing the dominant features and average engagement "
        "per cluster. Results are visualized on the t-SNE scatter plot.",
        styles["Body"],
    ))

    story.append(Paragraph("Hierarchical Clustering", styles["TechLabel"]))
    story.append(Paragraph(
        "Ward-linkage hierarchical clustering is applied to brand-level average feature "
        "profiles. The resulting dendrogram reveals which brands have the most similar ad "
        "strategies. For example, Bud Light and Budweiser cluster tightly, as do Toyota, "
        "Kia, and Hyundai.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # --- Module 5: Network & Crowd Intelligence ---
    story.append(Paragraph("Module: Network & Crowd Intelligence (v2 -- Session 6)", styles["SubHeader"]))

    story.append(Paragraph("Brand Strategy Network", styles["TechLabel"]))
    story.append(Paragraph(
        "A NetworkX graph with 10 nodes (brands) and edges weighted by cosine similarity "
        "of their average feature profiles + engagement patterns. Edges are only drawn "
        "above a configurable similarity threshold (default 0.3). Node size represents "
        "total engagement. The graph is rendered as an interactive force-directed "
        "visualization using Pyvis, embedded directly in the Streamlit dashboard via "
        "an HTML component.",
        styles["Body"],
    ))

    story.append(Paragraph("Ad Content Network", styles["TechLabel"]))
    story.append(Paragraph(
        "A second network connects individual ads (not brands) based on TF-IDF cosine "
        "similarity. With 193 nodes and up to 500 edges (capped for performance), this "
        "reveals content clusters that cross brand boundaries.",
        styles["Body"],
    ))

    story.append(Paragraph("Centrality Analysis", styles["TechLabel"]))
    story.append(Paragraph(
        "Four centrality measures are computed on the ad network: degree centrality "
        "(how connected an ad is), betweenness centrality (how often it bridges different "
        "clusters), closeness centrality (how accessible it is to all other ads), and "
        "PageRank (Google's algorithm measuring influence based on connection quality). "
        "Results are aggregated by brand and displayed as comparative bar charts.",
        styles["Body"],
    ))

    story.append(Paragraph("Wisdom of Crowds", styles["TechLabel"]))
    story.append(Paragraph(
        "We evaluate the Super Bowl ad ecosystem against James Surowiecki's four conditions "
        "for collective intelligence, each scored 0-100:",
        styles["Body"],
    ))
    woc_items = [
        "<b>Diversity of Opinion</b> -- Shannon entropy of feature combinations across all ads. Higher entropy = more diverse strategies.",
        "<b>Independence</b> -- Average inter-brand correlation of feature usage over time. Low correlation = brands making independent choices.",
        "<b>Decentralization</b> -- Inverse Gini coefficient of engagement distribution. Low Gini = engagement spread across many brands.",
        "<b>Aggregation</b> -- Rank correlation between crowd-voted engagement (views) and a multi-metric quality proxy (engagement rate + like ratio + comments). High correlation = crowd effectively aggregates quality signals.",
    ]
    for item in woc_items:
        story.append(Paragraph(f"&#8226; {item}", styles["BulletText"]))
    story.append(Paragraph(
        "The overall score (~41/100) suggests moderate crowd wisdom with room for "
        "improvement, particularly in strategic diversity and independence.",
        styles["Body"],
    ))

    # --- Module 6: SEM Performance Lab ---
    story.append(Paragraph("Module: SEM Performance Lab (v2 -- Session 5)", styles["SubHeader"]))
    story.append(Paragraph(
        "This module reframes Super Bowl ads through the lens of search engine marketing "
        "metrics, inspired by the Air France SEM case study. Since the dataset lacks actual "
        "budget data, we estimate costs using industry benchmarks for Super Bowl ad pricing "
        "($2.1M in 2000 scaling to $5.6M in 2020) and viewership (88M to 114M).",
        styles["Body"],
    ))

    sem_metrics = [
        "<b>CPV (Cost Per View)</b> -- Estimated ad cost divided by YouTube view count. Measures how efficiently a brand converts TV spend into digital views.",
        "<b>CPE (Cost Per Engagement)</b> -- Estimated cost divided by total engagements (likes + comments). Measures cost efficiency of driving meaningful interaction.",
        "<b>Quality Score Proxy (0-10)</b> -- Weighted combination of engagement rate rank (35%), like ratio rank (35%), and view count rank (30%). Analogous to Google Ads Quality Score.",
        "<b>Engagement ROI</b> -- Composite engagement score per million dollars spent. The primary efficiency metric.",
        "<b>Digital CTR Equivalent</b> -- YouTube views divided by estimated TV impressions. Measures how well TV exposure converts to digital interest.",
    ]
    for item in sem_metrics:
        story.append(Paragraph(f"&#8226; {item}", styles["BulletText"]))

    story.append(Paragraph(
        "The Publisher-Style Brand Comparison ranks all 10 brands in a table by CPV, CPE, "
        "ROI, and Quality Score, mirroring how the Air France case ranked Google, Yahoo, "
        "MSN, and other search publishers. Keyword Strategy Analysis uses TF-IDF to extract "
        "top terms per brand and categorizes them into Emotional, Action, Product, and Brand "
        "buckets, then correlates category usage with engagement outcomes.",
        styles["Body"],
    ))

    # --- Module 7: Google Trends ---
    story.append(Paragraph("Module: Google Trends Integration (v1 -- Session 1)", styles["SubHeader"]))
    story.append(Paragraph(
        "The TrendAnalyzer uses the pytrends library to pull Google Trends data for "
        "selected brands. This connects ad performance to real-world search behavior, "
        "showing whether Super Bowl ad buzz translates into sustained search interest. "
        "Interactive time series charts display relative search volume over configurable "
        "date ranges.",
        styles["Body"],
    ))

    story.append(PageBreak())

    # --- Course Concept Mapping ---
    story.append(Paragraph("Course Concept Mapping", styles["SubHeader"]))
    story.append(Paragraph(
        "The table below maps every technique implemented in AdPulse to its corresponding "
        "MSIS 521 course session and the module where it appears in the app.",
        styles["Body"],
    ))

    concepts = [
        ["Google Trends API", "S1: Web Analytics", "Search Trends page"],
        ["TF-IDF Vectorization", "S2: Word Embedding", "Text Analysis & Clustering"],
        ["Word2Vec Embeddings", "S2: Word Embedding", "Text Analysis & Clustering"],
        ["t-SNE Visualization", "S2: Word Embedding", "Text Analysis & Clustering"],
        ["VADER Sentiment", "S3: Text Analysis", "Sentiment Analysis page"],
        ["BERT Sentiment", "S3: Text Analysis / LLMs", "Text Analysis & Clustering"],
        ["LDA Topic Modeling", "S3: Text Analysis", "Topic Discovery page"],
        ["Cosine Similarity", "S3: Text Analysis", "Text Analysis & Clustering"],
        ["Jaccard Similarity", "S3: Text Analysis", "Text Analysis & Clustering"],
        ["K-Means Clustering", "S3: Text Analysis", "Text Analysis & Clustering"],
        ["Hierarchical Clustering", "S3: Text Analysis", "Text Analysis & Clustering"],
        ["Dendrogram", "S3: Text Analysis", "Text Analysis & Clustering"],
        ["CPV / CPE Metrics", "S5: SEM/SEO", "SEM Performance Lab"],
        ["Quality Score Proxy", "S5: SEM/SEO", "SEM Performance Lab"],
        ["ROI / ROA Analysis", "S5: SEM/SEO", "SEM Performance Lab"],
        ["Keyword Strategy", "S5: SEM/SEO", "SEM Performance Lab"],
        ["Publisher Comparison", "S5: SEM/SEO", "SEM Performance Lab"],
        ["NetworkX Graph Analysis", "S6: Social Networks", "Network & Crowd Intel"],
        ["Degree Centrality", "S6: Social Networks", "Network & Crowd Intel"],
        ["Betweenness Centrality", "S6: Social Networks", "Network & Crowd Intel"],
        ["Closeness / PageRank", "S6: Social Networks", "Network & Crowd Intel"],
        ["Wisdom of Crowds", "S6: Social Networks", "Network & Crowd Intel"],
    ]

    header = [
        Paragraph("<b>Technique</b>", styles["TableHeader"]),
        Paragraph("<b>Course Session</b>", styles["TableHeader"]),
        Paragraph("<b>App Module</b>", styles["TableHeader"]),
    ]
    table_data = [header]
    for row in concepts:
        table_data.append([Paragraph(cell, styles["TableCell"]) for cell in row])

    t2 = Table(table_data, colWidths=[2.2 * inch, 1.8 * inch, 2.7 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 16))

    # --- How to Run ---
    story.append(Paragraph("How to Run the App", styles["SubHeader"]))
    story.append(Paragraph(
        "Prerequisites: Python 3.9+ with pip. All dependencies are listed in "
        "<font face='Courier'>requirements.txt</font>.",
        styles["Body"],
    ))
    steps = [
        "Open a terminal and navigate to the AdPulse project directory",
        "Install dependencies: <font face='Courier'>pip install -r requirements.txt</font>",
        "Run the app: <font face='Courier'>streamlit run app.py</font>",
        "The dashboard opens automatically in your default browser at localhost:8501",
        "Use the sidebar to navigate between all 11 pages",
        "Optional: Install transformers and torch for BERT sentiment (large download, ~1GB)",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", styles["BulletText"]))

    story.append(Spacer(1, 16))

    # --- Presentation Tips ---
    story.append(Paragraph("Presentation Tips", styles["SubHeader"]))
    tips = [
        "Start with the problem slide to set context -- $5M+ per ad, brands need data not guesswork",
        "When showing the dashboard live, start on the Dashboard page for the overview, then jump to Feature Impact (the 70% lift stat is a crowd-pleaser)",
        "The dendrogram and network graph are the most visually impressive -- great for the technical demo portion",
        "Wisdom of Crowds is unique and shows deep course engagement; explain Surowiecki's 4 conditions briefly",
        "The SEM Performance Lab directly connects to the Air France case study from Session 5 -- mention this link explicitly",
        "End with the Methodology page showing 21 techniques across 5 sessions to reinforce breadth",
    ]
    for tip in tips:
        story.append(Paragraph(f"&#8226; {tip}", styles["BulletText"]))

    # Build
    doc.build(story)
    print(f"Team guide saved to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
