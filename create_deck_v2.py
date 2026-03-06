"""Generate the AdPulse v2 presentation deck (.pptx).

Updated for v2 with Text Analysis & Clustering, Network & Crowd Intelligence,
and SEM Performance Lab modules.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- Paths ---
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "deck_assets")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUT = os.path.join(BASE, "AdPulse_v2_Presentation.pptx")

# --- Colors ---
NAVY = RGBColor(0x1A, 0x1F, 0x36)
TEAL = RGBColor(0x08, 0x91, 0xB2)
CORAL = RGBColor(0xFF, 0x6B, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF8, 0xFA, 0xFC)
DARK_TEXT = RGBColor(0x1E, 0x29, 0x3B)
MID_GRAY = RGBColor(0x64, 0x74, 0x8B)
ICE_BLUE = RGBColor(0xCA, 0xDC, 0xFC)
PURPLE = RGBColor(0x8B, 0x5C, 0xF6)
GREEN = RGBColor(0x05, 0x96, 0x69)

HEX_NAVY = "#1A1F36"
HEX_TEAL = "#0891B2"
HEX_CORAL = "#FF6B6B"
HEX_WHITE = "#FFFFFF"
HEX_DARK = "#1E293B"
HEX_MID = "#64748B"

# --- Chart generation ---

def load_data():
    sys.path.insert(0, BASE)
    from data_utils import load_superbowl_data
    return load_superbowl_data()

def make_feature_chart(df):
    from analysis import EngagementAnalyzer
    eng = EngagementAnalyzer(df)
    impact = eng.feature_impact()
    impact = impact.sort_values("views_lift")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [HEX_TEAL if v >= 0 else HEX_CORAL for v in impact["views_lift"]]
    bars = ax.barh(impact["feature"], impact["views_lift"], color=colors, height=0.6)
    ax.set_xlabel("Views Lift (%)", fontsize=11, color=HEX_MID)
    ax.set_title("Impact of Ad Features on Views", fontsize=14, fontweight="bold", color=HEX_DARK, pad=15)
    ax.axvline(0, color="#CBD5E1", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.spines["left"].set_color("#E2E8F0")
    ax.tick_params(colors=HEX_MID)
    for bar, val in zip(bars, impact["views_lift"]):
        ax.text(bar.get_width() + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                f"{val:+.0f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=10, color=HEX_DARK)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "feature_impact.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def make_sentiment_chart(df):
    from analysis import SentimentAnalyzer
    sa = SentimentAnalyzer("vader")
    descs = df[df["description"].notna()]["description"].tolist()
    results = sa.analyze(descs)
    counts = results["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    colors_map = {"positive": "#2ca02c", "neutral": "#94a3b8", "negative": "#ef4444"}
    labels = counts.index.tolist()
    vals = counts.values.tolist()
    cols = [colors_map.get(l, "#999") for l in labels]
    wedges, texts, autotexts = ax.pie(
        vals, labels=[l.title() for l in labels], autopct="%1.0f%%",
        colors=cols, startangle=90, textprops={"fontsize": 12}
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("Sentiment of Ad Descriptions", fontsize=14, fontweight="bold", color=HEX_DARK, pad=15)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "sentiment_pie.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def make_brand_chart(df):
    from analysis import EngagementAnalyzer
    brand_sum = EngagementAnalyzer(df).brand_summary().head(6)
    brand_sum = brand_sum.sort_values("avg_views")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(brand_sum["brand"], brand_sum["avg_views"] / 1e6, color=HEX_TEAL, height=0.55)
    ax.set_xlabel("Average Views (Millions)", fontsize=11, color=HEX_MID)
    ax.set_title("Top Brands by Average Views", fontsize=14, fontweight="bold", color=HEX_DARK, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.spines["left"].set_color("#E2E8F0")
    ax.tick_params(colors=HEX_MID)
    for i, (_, row) in enumerate(brand_sum.iterrows()):
        ax.text(row["avg_views"]/1e6 + 0.1, i, f"{row['avg_views']/1e6:.1f}M",
                va="center", fontsize=10, color=HEX_DARK)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "brand_views.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def make_dendrogram_chart(df):
    from clustering import hierarchical_cluster_brands
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    Z, brand_labels = hierarchical_cluster_brands(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    scipy_dendrogram(Z, labels=brand_labels, ax=ax, leaf_rotation=35, leaf_font_size=10,
                     color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Brand Strategy Dendrogram", fontsize=14, fontweight="bold", color=HEX_DARK, pad=15)
    ax.set_ylabel("Distance", color=HEX_MID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=HEX_MID)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "dendrogram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def make_woc_chart(df):
    from network_analysis import wisdom_of_crowds_analysis
    woc = wisdom_of_crowds_analysis(df)

    conditions = ["Diversity", "Independence", "Decentralization", "Aggregation"]
    scores = [woc[k.lower()]["score"] for k in conditions]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [HEX_TEAL if s >= 60 else ("#F59E0B" if s >= 30 else HEX_CORAL) for s in scores]
    bars = ax.barh(conditions, scores, color=colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (0-100)", fontsize=11, color=HEX_MID)
    ax.set_title("Wisdom of Crowds: 4 Conditions", fontsize=14, fontweight="bold", color=HEX_DARK, pad=15)
    ax.axvline(50, color="#CBD5E1", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.spines["left"].set_color("#E2E8F0")
    ax.tick_params(colors=HEX_MID)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=11, fontweight="bold", color=HEX_DARK)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "woc_scores.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1, "Data\nCollection", HEX_TEAL),
        (2.3, 1, "Preprocessing\n& Cleaning", "#0E7490"),
        (4.3, 1, "AI Analysis\n(NLP + ML)", HEX_CORAL),
        (6.3, 1, "Network &\nSEM Analysis", "#8B5CF6"),
        (8.3, 1, "Interactive\nDashboard", "#059669"),
    ]
    for bx, by, label, color in boxes:
        rect = plt.Rectangle((bx, by), 1.6, 1.2, facecolor=color, edgecolor="none", alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(bx + 0.8, by + 0.6, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.6
        x2 = boxes[i+1][0]
        y = 1.6
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=2))

    sub_items = [
        (0.3, 0.3, "FiveThirtyEight\n247 ads, 10 brands"),
        (2.3, 0.3, "NLTK tokenize\nTF-IDF, Word2Vec"),
        (4.3, 0.3, "VADER + BERT\nLDA, K-Means, t-SNE"),
        (6.3, 0.3, "NetworkX, Pyvis\nSEM metrics, ROI"),
        (8.3, 0.3, "Streamlit 11-page\ninteractive app"),
    ]
    for sx, sy, label in sub_items:
        ax.text(sx + 0.8, sy + 0.2, label, ha="center", va="center",
                fontsize=7, color=HEX_MID, style="italic")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "pipeline_v2.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --- Slide helpers ---

def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, prs, x, y, w, h, fill_color, transparency=0):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape

def add_text_box(slide, text, x, y, w, h, font_size=14, color=DARK_TEXT,
                 bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri",
                 anchor=MSO_ANCHOR.TOP):
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    tf.auto_size = None
    return txBox

def add_bullet_list(slide, items, x, y, w, h, font_size=14, color=DARK_TEXT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(6)
        p.level = 0
        pPr = p._pPr
        if pPr is None:
            from pptx.oxml.ns import qn
            pPr = p._p.get_or_add_pPr()
        from pptx.oxml.ns import qn
        buNone = pPr.findall(qn("a:buNone"))
        for bn in buNone:
            pPr.remove(bn)
        from lxml import etree
        buChar = etree.SubElement(pPr, qn("a:buChar"))
        buChar.set("char", "\u2022")
    return txBox

def add_stat_card(slide, number, label, x, y, w=Inches(2), h=Inches(1.2),
                  num_color=TEAL, label_color=MID_GRAY):
    add_shape(slide, None, x, y, w, h, LIGHT_GRAY)
    add_text_box(slide, number, x + Inches(0.15), y + Inches(0.1),
                 w - Inches(0.3), Inches(0.65),
                 font_size=28, color=num_color, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, label, x + Inches(0.15), y + Inches(0.7),
                 w - Inches(0.3), Inches(0.4),
                 font_size=11, color=label_color, alignment=PP_ALIGN.CENTER)


# --- Build presentation ---

def build_deck():
    df = load_data()

    # Generate chart images
    feat_chart = make_feature_chart(df)
    sent_chart = make_sentiment_chart(df)
    brand_chart = make_brand_chart(df)
    pipeline_img = make_pipeline_diagram()
    dendro_chart = make_dendrogram_chart(df)
    woc_chart = make_woc_chart(df)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ═══════════════════════════════════════
    # SLIDE 1: Title
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, NAVY)
    add_shape(slide, prs, Inches(0), Inches(0), prs.slide_width, Inches(0.06), TEAL)

    add_text_box(slide, "AdPulse", Inches(1), Inches(1.5), Inches(11), Inches(1.0),
                 font_size=54, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "AI-Powered Ad Campaign Performance Analyzer",
                 Inches(1), Inches(2.6), Inches(11), Inches(0.8),
                 font_size=22, color=ICE_BLUE, alignment=PP_ALIGN.CENTER)

    add_shape(slide, prs, Inches(4.5), Inches(3.6), Inches(4.3), Inches(0.02), TEAL)

    add_text_box(slide, "MSIS 521 -- IT and Marketing in the New Economy",
                 Inches(1), Inches(4.0), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "Winter Quarter 2025  |  University of Washington",
                 Inches(1), Inches(4.5), Inches(11), Inches(0.5),
                 font_size=13, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, "Team 6 -- Purple Cohort",
                 Inches(1), Inches(5.4), Inches(11), Inches(0.5),
                 font_size=15, color=ICE_BLUE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "Graeme Ampeire  |  Soumya Thareja  |  Tiffany Nakamitsu  |  Akanksha Singh",
                 Inches(1), Inches(5.9), Inches(11), Inches(0.5),
                 font_size=13, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # ═══════════════════════════════════════
    # SLIDE 2: The Problem
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "The Problem", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "Why brands need better ad performance analytics",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    add_stat_card(slide, "$7M+", "Cost per 30-sec\nSuper Bowl ad",
                  Inches(0.8), Inches(2.2), Inches(3.5), Inches(1.4), CORAL)
    add_stat_card(slide, "113M+", "Average Super Bowl\nviewers",
                  Inches(0.8), Inches(3.9), Inches(3.5), Inches(1.4), TEAL)
    add_stat_card(slide, "50+", "Ads per game,\nlimited attention",
                  Inches(0.8), Inches(5.6), Inches(3.5), Inches(1.4), PURPLE)

    add_bullet_list(slide, [
        "Brands invest millions but rely on subjective measures of ad success",
        "Traditional metrics (TV ratings, reach) miss the digital conversation",
        "No unified tool to analyze engagement, sentiment, and themes at scale",
        "Marketers need data-driven insights to optimize future campaigns",
    ], Inches(5.0), Inches(2.3), Inches(7.5), Inches(4.5), font_size=16, color=DARK_TEXT)

    # ═══════════════════════════════════════
    # SLIDE 3: Our Solution
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "Our Solution: AdPulse", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "An 11-page AI dashboard covering 21 course techniques across 5 sessions",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    features = [
        ("Dashboard & Rankings", "KPI metrics, composite scoring,\nbrand comparison"),
        ("Feature Impact", "Which ad characteristics\ndrive engagement (lift %)"),
        ("Sentiment Analysis", "VADER & BERT NLP on\nad descriptions"),
        ("Text Analysis & Clustering", "TF-IDF, Word2Vec, K-Means,\nhierarchical clustering, t-SNE"),
        ("Network & Crowd Intel", "Brand networks, centrality,\nWisdom of Crowds scoring"),
        ("SEM Performance Lab", "CPV, CPE, Quality Score, ROI\n(Air France case approach)"),
    ]
    accents = [TEAL, CORAL, TEAL, PURPLE, GREEN, CORAL]
    for i, ((title, desc), accent) in enumerate(zip(features, accents)):
        col = i % 3
        row = i // 3
        fx = Inches(0.8) + col * Inches(4.1)
        fy = Inches(2.0) + row * Inches(2.4)
        add_shape(slide, prs, fx, fy, Inches(3.8), Inches(2.1), LIGHT_GRAY)
        add_shape(slide, prs, fx, fy, Inches(0.06), Inches(2.1), accent)
        add_text_box(slide, title, fx + Inches(0.25), fy + Inches(0.2),
                     Inches(3.3), Inches(0.45), font_size=15, color=DARK_TEXT, bold=True)
        add_text_box(slide, desc, fx + Inches(0.25), fy + Inches(0.7),
                     Inches(3.3), Inches(1.2), font_size=12, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 4: The Data
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "The Data", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "Real-world dataset from FiveThirtyEight / TidyTuesday",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    data_stats = [
        ("247", "Super Bowl Ads"),
        ("10", "Major Brands"),
        ("21", "Years (2000-2020)"),
        ("207", "Video Descriptions"),
        ("25", "Data Columns"),
    ]
    for i, (num, label) in enumerate(data_stats):
        sx = Inches(0.6) + i * Inches(2.5)
        add_stat_card(slide, num, label, sx, Inches(2.0), Inches(2.2), Inches(1.3))

    add_bullet_list(slide, [
        "Ad characteristics: funny, celebrity, patriotic, danger, animals, use_sex, show_product_quickly",
        "YouTube engagement: views, likes, dislikes, comments, favorites",
        "Video metadata: title, description, publish date, channel, thumbnail URL",
    ], Inches(0.8), Inches(3.8), Inches(7), Inches(2.5), font_size=14, color=DARK_TEXT)

    slide.shapes.add_picture(brand_chart, Inches(8.2), Inches(3.6), Inches(4.8), Inches(2.8))

    # ═══════════════════════════════════════
    # SLIDE 5: Technical Architecture
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "Technical Architecture", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "End-to-end AI pipeline built in Python",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    slide.shapes.add_picture(pipeline_img, Inches(0.5), Inches(2.0), Inches(12.3), Inches(3.2))

    stack_items = [
        ("AI/NLP", "VADER + BERT sentiment, LDA topics, Word2Vec embeddings, TF-IDF, K-Means, t-SNE"),
        ("Networks", "NetworkX graph analysis, Pyvis visualization, PageRank, 4 centrality measures"),
        ("SEM", "CPV/CPE/ROI metrics, Quality Score proxy, keyword strategy, publisher comparison"),
        ("Viz + App", "Plotly interactive charts, Streamlit 11-page dashboard, real-time filtering"),
    ]
    for i, (cat, desc) in enumerate(stack_items):
        ty = Inches(5.6) + i * Inches(0.38)
        add_text_box(slide, f"{cat}:  {desc}", Inches(0.8), ty, Inches(12), Inches(0.35),
                     font_size=12, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 6: Feature Impact
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, CORAL)

    add_text_box(slide, "What Makes an Ad Successful?", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "Feature impact analysis from 247 Super Bowl ads",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    slide.shapes.add_picture(feat_chart, Inches(0.5), Inches(2.0), Inches(8.5), Inches(4.5))

    add_shape(slide, prs, Inches(9.3), Inches(2.5), Inches(3.5), Inches(3.5), LIGHT_GRAY)
    add_shape(slide, prs, Inches(9.3), Inches(2.5), Inches(0.06), Inches(3.5), TEAL)
    add_text_box(slide, "Key Insights", Inches(9.55), Inches(2.7), Inches(3.1), Inches(0.4),
                 font_size=16, color=DARK_TEXT, bold=True)
    add_bullet_list(slide, [
        "Show Product Quickly: +70% views lift",
        "Patriotic themes: +33% lift",
        "Funny ads: +12% lift",
        "Celebrity ads actually get fewer views (-54%)",
    ], Inches(9.55), Inches(3.2), Inches(3.1), Inches(2.5), font_size=12, color=DARK_TEXT)

    # ═══════════════════════════════════════
    # SLIDE 7: NLP - Sentiment & Topics
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, CORAL)

    add_text_box(slide, "NLP Analysis: Sentiment & Topics", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "VADER + BERT sentiment, LDA topic modeling",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    slide.shapes.add_picture(sent_chart, Inches(0.5), Inches(2.0), Inches(5.0), Inches(4.5))

    add_shape(slide, prs, Inches(6.0), Inches(2.0), Inches(6.8), Inches(4.5), LIGHT_GRAY)
    add_shape(slide, prs, Inches(6.0), Inches(2.0), Inches(0.06), Inches(4.5), PURPLE)

    add_text_box(slide, "Dual Sentiment + Topic Modeling", Inches(6.3), Inches(2.2), Inches(6.2), Inches(0.4),
                 font_size=18, color=DARK_TEXT, bold=True)
    add_bullet_list(slide, [
        "VADER: Rule-based lexicon scoring (fast, interpretable)",
        "BERT: Deep learning contextual analysis (nlptown model)",
        "Side-by-side comparison reveals agreement and divergence",
        "LDA topic modeling discovers 5 latent themes in ad copy",
        "Word clouds and topic quality scoring (distinctness metric)",
    ], Inches(6.3), Inches(2.8), Inches(6.2), Inches(3.2), font_size=13, color=DARK_TEXT)

    # ═══════════════════════════════════════
    # SLIDE 8: Text Analysis & Clustering (NEW)
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, PURPLE)

    add_text_box(slide, "Text Analysis & Clustering", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "TF-IDF, Word2Vec, cosine similarity, K-Means, hierarchical clustering (Sessions 2 & 3)",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    # Left: dendrogram image
    slide.shapes.add_picture(dendro_chart, Inches(0.5), Inches(2.0), Inches(6.5), Inches(4.0))

    # Right: technique descriptions
    add_shape(slide, prs, Inches(7.3), Inches(2.0), Inches(5.5), Inches(5.0), LIGHT_GRAY)
    add_shape(slide, prs, Inches(7.3), Inches(2.0), Inches(0.06), Inches(5.0), PURPLE)

    techniques = [
        ("TF-IDF Vectorization", "Identifies each brand's most distinctive ad vocabulary"),
        ("Word2Vec Embeddings", "Learns semantic word relationships from ad corpus"),
        ("t-SNE Visualization", "Projects high-dim embeddings to interactive 2D scatter"),
        ("Cosine Similarity", "Measures content similarity between ad descriptions"),
        ("Jaccard Similarity", "Compares brands by shared feature profiles"),
        ("K-Means Clustering", "Groups ads by content + features into clusters"),
        ("Hierarchical Clustering", "Dendrogram reveals brand strategy relationships"),
    ]
    for i, (tech, desc) in enumerate(techniques):
        ty = Inches(2.3) + i * Inches(0.65)
        add_text_box(slide, tech, Inches(7.55), ty, Inches(5.0), Inches(0.3),
                     font_size=13, color=DARK_TEXT, bold=True)
        add_text_box(slide, desc, Inches(7.55), ty + Inches(0.25), Inches(5.0), Inches(0.3),
                     font_size=11, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 9: Network & Crowd Intelligence (NEW)
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, GREEN)

    add_text_box(slide, "Network & Crowd Intelligence", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "Brand influence networks, centrality analysis, Wisdom of Crowds (Session 6)",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    # Left: network description
    add_shape(slide, prs, Inches(0.8), Inches(2.0), Inches(5.5), Inches(4.8), LIGHT_GRAY)
    add_shape(slide, prs, Inches(0.8), Inches(2.0), Inches(0.06), Inches(4.8), GREEN)

    add_text_box(slide, "Brand Strategy Network", Inches(1.1), Inches(2.2), Inches(5.0), Inches(0.4),
                 font_size=18, color=DARK_TEXT, bold=True)
    add_bullet_list(slide, [
        "10 brand nodes, edges weighted by strategy similarity",
        "Interactive Pyvis force-directed graph in dashboard",
        "Node size = total engagement, edge width = similarity",
    ], Inches(1.1), Inches(2.8), Inches(5.0), Inches(1.8), font_size=13, color=DARK_TEXT)

    add_text_box(slide, "Centrality Measures", Inches(1.1), Inches(4.5), Inches(5.0), Inches(0.4),
                 font_size=16, color=DARK_TEXT, bold=True)
    add_bullet_list(slide, [
        "Degree: connectivity / popularity",
        "Betweenness: bridge role between clusters",
        "Closeness: accessibility to all other ads",
        "PageRank: influence based on connection quality",
    ], Inches(1.1), Inches(5.0), Inches(5.0), Inches(1.5), font_size=12, color=DARK_TEXT)

    # Right: WoC chart
    slide.shapes.add_picture(woc_chart, Inches(6.8), Inches(2.0), Inches(6.0), Inches(3.8))

    add_text_box(slide, "Surowiecki's 4 conditions evaluated against the Super Bowl ad ecosystem",
                 Inches(6.8), Inches(5.9), Inches(6.0), Inches(0.5),
                 font_size=12, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 10: SEM Performance Lab (NEW)
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, CORAL)

    add_text_box(slide, "SEM Performance Lab", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "Super Bowl ads through an SEM lens (Session 5 -- Air France case study approach)",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    # SEM metric cards
    sem_cards = [
        ("CPV", "Cost Per View", "Estimated ad cost /\nYouTube view count", TEAL),
        ("CPE", "Cost Per Engagement", "Estimated ad cost /\n(likes + comments)", CORAL),
        ("Quality Score", "0-10 Proxy", "Engagement rate +\nlike ratio + views rank", PURPLE),
        ("ROI", "Return on Ad Spend", "Engagement score /\nestimated $ spent", GREEN),
    ]
    for i, (metric, subtitle, desc, accent) in enumerate(sem_cards):
        sx = Inches(0.6) + i * Inches(3.2)
        add_shape(slide, prs, sx, Inches(2.2), Inches(2.9), Inches(2.2), LIGHT_GRAY)
        add_shape(slide, prs, sx, Inches(2.2), Inches(2.9), Inches(0.06), accent)
        add_text_box(slide, metric, sx + Inches(0.2), Inches(2.35), Inches(2.5), Inches(0.45),
                     font_size=22, color=accent, bold=True)
        add_text_box(slide, subtitle, sx + Inches(0.2), Inches(2.8), Inches(2.5), Inches(0.35),
                     font_size=12, color=DARK_TEXT, bold=True)
        add_text_box(slide, desc, sx + Inches(0.2), Inches(3.2), Inches(2.5), Inches(0.9),
                     font_size=11, color=MID_GRAY)

    # Bottom section
    add_text_box(slide, "Publisher-Style Brand Comparison", Inches(0.8), Inches(4.8), Inches(5), Inches(0.4),
                 font_size=18, color=DARK_TEXT, bold=True)
    add_bullet_list(slide, [
        "Ranks 10 brands like SEM publishers (mirroring Air France case)",
        "Brands compared by CPV, CPE, ROI, and Quality Score",
        "Keyword strategy analysis with TF-IDF term categorization",
        "Feature-based strategy recommendation engine",
    ], Inches(0.8), Inches(5.3), Inches(12), Inches(2.0), font_size=13, color=DARK_TEXT)

    # ═══════════════════════════════════════
    # SLIDE 11: Course Concept Mapping
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "Course Concept Coverage", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "21 techniques spanning 5 of 6 course sessions",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    # Stats
    add_stat_card(slide, "21", "Techniques\nImplemented", Inches(0.6), Inches(2.0), Inches(2.4), Inches(1.3), TEAL)
    add_stat_card(slide, "5/6", "Course Sessions\nCovered", Inches(3.2), Inches(2.0), Inches(2.4), Inches(1.3), PURPLE)
    add_stat_card(slide, "11", "Dashboard\nPages", Inches(5.8), Inches(2.0), Inches(2.4), Inches(1.3), GREEN)
    add_stat_card(slide, "6", "Validation\nMethods", Inches(8.4), Inches(2.0), Inches(2.4), Inches(1.3), CORAL)

    # Session mapping
    sessions = [
        ("S1: Web Analytics", "Google Trends integration", TEAL),
        ("S2: Word Embedding", "TF-IDF, Word2Vec, t-SNE", TEAL),
        ("S3: Text Analysis & LLMs", "VADER, BERT, cosine similarity, Jaccard, K-Means, hierarchical clustering, LDA", PURPLE),
        ("S5: SEM/SEO", "CPV, CPE, ROI, Quality Score, keyword strategy, publisher comparison", CORAL),
        ("S6: Social Networks", "NetworkX, degree/betweenness/closeness/PageRank centrality, Wisdom of Crowds", GREEN),
    ]
    for i, (session, techniques, accent) in enumerate(sessions):
        sy = Inches(3.8) + i * Inches(0.72)
        add_shape(slide, prs, Inches(0.8), sy, Inches(0.06), Inches(0.6), accent)
        add_text_box(slide, session, Inches(1.1), sy + Inches(0.05), Inches(3.0), Inches(0.35),
                     font_size=14, color=DARK_TEXT, bold=True)
        add_text_box(slide, techniques, Inches(4.3), sy + Inches(0.05), Inches(8.5), Inches(0.5),
                     font_size=12, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 12: Output Validation
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "Output Validation", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)
    add_text_box(slide, "6 validation methods ensuring AI outputs are reliable",
                 Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY)

    val_items = [
        ("Sentiment Distribution", "Flags if >95% single class", TEAL),
        ("Known-Sample Accuracy", "5 test cases, 80% accuracy", GREEN),
        ("Topic Distinctness", "Cosine distance score: 0.312", PURPLE),
        ("BERT vs VADER", "Agreement rate comparison", CORAL),
        ("Wisdom of Crowds", "4-condition scoring system", GREEN),
        ("SEM Benchmarking", "Industry cost validation", TEAL),
    ]
    for i, (title, result, accent) in enumerate(val_items):
        col = i % 3
        row = i // 3
        vx = Inches(0.6) + col * Inches(4.2)
        vy = Inches(2.0) + row * Inches(2.5)
        add_shape(slide, prs, vx, vy, Inches(3.9), Inches(2.2), LIGHT_GRAY)
        add_shape(slide, prs, vx, vy, Inches(0.06), Inches(2.2), accent)
        add_text_box(slide, title, vx + Inches(0.25), vy + Inches(0.2),
                     Inches(3.4), Inches(0.5), font_size=16, color=DARK_TEXT, bold=True)
        add_shape(slide, prs, vx + Inches(0.15), vy + Inches(1.0), Inches(3.6), Inches(0.8), WHITE)
        add_text_box(slide, result, vx + Inches(0.3), vy + Inches(1.1),
                     Inches(3.3), Inches(0.6), font_size=13, color=accent, bold=True)

    # ═══════════════════════════════════════
    # SLIDE 13: Key Findings
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_shape(slide, prs, Inches(0), Inches(0), Inches(0.06), prs.slide_height, TEAL)

    add_text_box(slide, "Key Findings", Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
                 font_size=36, color=DARK_TEXT, bold=True)

    findings = [
        ("Show Product Quickly wins",
         "Ads that show the product early get 70% more views. Clarity beats cleverness."),
        ("Celebrity doesn't guarantee success",
         "Celebrity ads averaged 54% fewer views. The product and story matter more."),
        ("Brands cluster into strategy groups",
         "Hierarchical clustering reveals beer brands (Bud Light, Budweiser) and tech brands (Hyundai, Kia, Toyota) use similar strategies."),
        ("Network analysis reveals bridge brands",
         "Some brands' ads bridge different content clusters, connecting humor-based and product-focused strategies."),
        ("Crowd wisdom is moderate",
         "The Super Bowl ad ecosystem scores ~41/100 on Wisdom of Crowds -- showing herding behavior among brands."),
        ("SEM efficiency varies 10x across brands",
         "Cost per view ranges from $0.50 to $50+ across brands, revealing massive efficiency gaps."),
    ]
    for i, (title, desc) in enumerate(findings):
        fy = Inches(1.4) + i * Inches(0.95)
        accents = [TEAL, CORAL, PURPLE, GREEN, TEAL, CORAL]
        add_shape(slide, prs, Inches(0.8), fy, Inches(0.06), Inches(0.78), accents[i])
        add_text_box(slide, title, Inches(1.1), fy + Inches(0.02), Inches(3.8), Inches(0.35),
                     font_size=14, color=DARK_TEXT, bold=True)
        add_text_box(slide, desc, Inches(5.0), fy + Inches(0.02), Inches(7.8), Inches(0.7),
                     font_size=12, color=MID_GRAY)

    # ═══════════════════════════════════════
    # SLIDE 14: Live Demo
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, NAVY)
    add_shape(slide, prs, Inches(0), Inches(0), prs.slide_width, Inches(0.06), TEAL)

    add_text_box(slide, "Live Demo", Inches(1), Inches(2.2), Inches(11), Inches(1.0),
                 font_size=48, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "AdPulse v2 -- 11-Page Interactive Streamlit Dashboard",
                 Inches(1), Inches(3.4), Inches(11), Inches(0.6),
                 font_size=20, color=ICE_BLUE, alignment=PP_ALIGN.CENTER)

    add_shape(slide, prs, Inches(4.5), Inches(4.3), Inches(4.3), Inches(0.02), TEAL)

    add_text_box(slide, "streamlit run app.py",
                 Inches(1), Inches(4.8), Inches(11), Inches(0.5),
                 font_size=16, color=MID_GRAY, alignment=PP_ALIGN.CENTER, font_name="Consolas")

    # ═══════════════════════════════════════
    # SLIDE 15: Thank You
    # ═══════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, NAVY)
    add_shape(slide, prs, Inches(0), Inches(0), prs.slide_width, Inches(0.06), TEAL)

    add_text_box(slide, "Thank You", Inches(1), Inches(1.8), Inches(11), Inches(1.0),
                 font_size=48, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "Questions?",
                 Inches(1), Inches(3.0), Inches(11), Inches(0.6),
                 font_size=22, color=ICE_BLUE, alignment=PP_ALIGN.CENTER)

    add_shape(slide, prs, Inches(4.5), Inches(3.9), Inches(4.3), Inches(0.02), TEAL)

    add_text_box(slide, "Team 6 -- Purple Cohort",
                 Inches(1), Inches(4.4), Inches(11), Inches(0.5),
                 font_size=15, color=ICE_BLUE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "Graeme Ampeire  |  Soumya Thareja  |  Tiffany Nakamitsu  |  Akanksha Singh",
                 Inches(1), Inches(4.9), Inches(11), Inches(0.5),
                 font_size=13, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "AdPulse  |  AI-Powered Ad Campaign Performance Analyzer",
                 Inches(1), Inches(5.6), Inches(11), Inches(0.5),
                 font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, "MSIS 521 -- Winter 2025 -- University of Washington",
                 Inches(1), Inches(6.1), Inches(11), Inches(0.5),
                 font_size=13, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Save
    prs.save(OUTPUT)
    print(f"Presentation saved to: {OUTPUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    build_deck()
