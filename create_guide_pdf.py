"""Generate a condensed 2-page AdPulse Project Guide PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    HRFlowable, Frame, PageTemplate, BaseDocTemplate,
)

OUTPUT = "/Users/graemetobiasampeire/Library/Mobile Documents/com~apple~CloudDocs/Career/Education/Masters Degree/UW/Winter Quarter/MSIS 521 - Information Technology And Marketing In The New Economy/AdPulse/AdPulse_Project_Guide.pdf"

NAVY = HexColor("#1A1F36")
TEAL = HexColor("#0891B2")
DARK = HexColor("#1E293B")
MID = HexColor("#64748B")
LIGHT_BG = HexColor("#F1F5F9")
WHITE = white


def S():
    """Build compact styles."""
    ss = getSampleStyleSheet()
    d = {}
    d["title"] = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=16, leading=18,
                                 textColor=NAVY, alignment=TA_CENTER, spaceAfter=1)
    d["subtitle"] = ParagraphStyle("ST", fontName="Helvetica", fontSize=8, leading=10,
                                    textColor=MID, alignment=TA_CENTER, spaceAfter=4)
    d["h1"] = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=11, leading=13,
                              textColor=NAVY, spaceBefore=6, spaceAfter=2)
    d["h2"] = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5,
                              textColor=TEAL, spaceBefore=5, spaceAfter=1)
    d["h3"] = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=7.5, leading=9.5,
                              textColor=DARK, spaceBefore=3, spaceAfter=1)
    d["body"] = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=7,
                                leading=9, textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=2)
    d["bullet"] = ParagraphStyle("BL", parent=d["body"], leftIndent=10, bulletIndent=3,
                                  spaceAfter=1)
    d["italic"] = ParagraphStyle("IT", parent=d["body"], fontName="Helvetica-Oblique",
                                  textColor=MID, spaceAfter=1.5)
    d["th"] = ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=6.5, leading=8,
                              textColor=WHITE, alignment=TA_CENTER)
    d["tc"] = ParagraphStyle("TC", fontName="Helvetica", fontSize=6.5, leading=8, textColor=DARK)
    d["tcc"] = ParagraphStyle("TCC", parent=d["tc"], alignment=TA_CENTER)
    return d


def hr():
    return HRFlowable(width="100%", thickness=0.4, color=HexColor("#CBD5E1"),
                       spaceBefore=2, spaceAfter=3)


def build_pdf():
    s = S()
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
                            leftMargin=0.55*inch, rightMargin=0.55*inch,
                            topMargin=0.45*inch, bottomMargin=0.4*inch)
    story = []

    # ════════════════════════════════════════
    # PAGE 1: DETAILED TECHNICAL EXPLANATION
    # ════════════════════════════════════════
    story.append(Paragraph("AdPulse - Project Guide", s["title"]))
    story.append(Paragraph("AI-Powered Ad Campaign Performance Analyzer  |  MSIS 521  |  University of Washington  |  Winter 2025", s["subtitle"]))
    story.append(hr())

    story.append(Paragraph("Part 1: Detailed Technical Explanation", s["h1"]))

    story.append(Paragraph("What We Built", s["h2"]))
    story.append(Paragraph(
        "AdPulse is a web-based tool that helps marketers understand <b>why some ads perform better than others</b>. "
        "It takes real advertising data (247 Super Bowl ads, 2000-2020) and runs AI and statistical analyses, "
        "presenting results through an interactive Streamlit dashboard. The tool is extensible: anyone can upload "
        "their own campaign CSV and get the same analyses on any set of ads.", s["body"]))

    story.append(Paragraph("The Data", s["h2"]))
    story.append(Paragraph(
        "We used a real dataset from <b>FiveThirtyEight</b>, enhanced by <b>TidyTuesday</b>. FiveThirtyEight staff "
        "watched every Super Bowl ad (2000-2020) and tagged each with characteristics (funny, celebrity, patriotic, "
        "danger, animals, use_sex, show_product_quickly). The TidyTuesday version added <b>YouTube engagement data</b> "
        "(views, likes, dislikes, comments, favorites) plus video titles and descriptions. This gives us <b>247 ads "
        "across 10 major brands</b> (Bud Light, Budweiser, Coca-Cola, Doritos, Hyundai, Kia, NFL, Pepsi, Toyota, "
        "E-Trade) with <b>25 columns</b> combining human-coded characteristics and real engagement numbers.", s["body"]))

    story.append(Paragraph("The Six Core Analyses", s["h2"]))

    story.append(Paragraph("1. Campaign Dashboard", s["h3"]))
    story.append(Paragraph(
        "Overview page with four KPIs (total ads, total views, average views, brand count), a views-over-time line "
        "chart, and brand comparison. Built with pandas aggregation and Plotly interactive charts. Streamlit sidebar "
        "provides year-range and brand filters across all pages.", s["body"]))

    story.append(Paragraph("2. Ad Performance Ranking", s["h3"]))
    story.append(Paragraph(
        "A <b>composite engagement score</b> (0-100) weights multiple metrics: Views 40%, Likes 35%, Comments 25%. "
        "Each metric is min-max normalized to 0-1, multiplied by its weight, summed, then scaled to 100. This "
        "single number balances reach with active audience engagement for fair cross-ad comparison.", s["body"]))

    story.append(Paragraph("3. Feature Impact Analysis", s["h3"]))
    story.append(Paragraph(
        "For each of the 7 boolean features, we split ads into WITH vs. WITHOUT groups and compute: "
        "views_lift = (avg_with / avg_without - 1) x 100. <b>Key finding:</b> \"Show Product Quickly\" gives +70% "
        "lift; celebrity appearances show -54% lift (counterintuitive but real in this dataset).", s["body"]))

    story.append(Paragraph("4. Sentiment Analysis", s["h3"]))
    story.append(Paragraph(
        "We use <b>VADER</b> (Valence Aware Dictionary and sEntiment Reasoner) from NLTK on ad descriptions. "
        "VADER assigns compound scores (-1 to +1); we classify >=0.05 as positive, <=-0.05 as negative, rest "
        "neutral. An optional BERT transformer model is available for higher accuracy. "
        "<b>Key finding:</b> ~75% of descriptions are positive; brands deliberately craft optimistic narratives.", s["body"]))

    story.append(Paragraph("5. Topic Modeling (LDA)", s["h3"]))
    story.append(Paragraph(
        "We use <b>Latent Dirichlet Allocation</b> to discover themes across 207 descriptions. The pipeline: "
        "(a) preprocess text (remove URLs/punctuation, lowercase, tokenize, remove stopwords, lemmatize); "
        "(b) vectorize with CountVectorizer (max_df=0.95, min_df=2, max 1000 features); "
        "(c) fit LDA with 5 topics, extracting top 10 words per topic; "
        "(d) assign each ad its dominant topic. "
        "<b>Key finding:</b> Topics cluster around brands (Hyundai, Bud Light, Doritos) and themes (NFL/football).", s["body"]))

    story.append(Paragraph("6. Search Trends", s["h3"]))
    story.append(Paragraph(
        "Integrates <b>Google Trends</b> via pytrends to show how brand search interest changes around Super Bowl "
        "time, serving as a proxy for ad effectiveness beyond YouTube views.", s["body"]))

    story.append(Paragraph("Output Validation", s["h2"]))
    story.append(Paragraph(
        "<b>Sentiment Distribution Check:</b> Flags results where a single class exceeds 95% (suggests a broken model). "
        "<b>Known-Sample Accuracy:</b> Runs VADER on 5 hand-picked test sentences with known answers and reports "
        "accuracy (e.g., 4/5 = 80%). "
        "<b>Topic Distinctness Score:</b> Computes pairwise cosine similarity between topic-word distributions; "
        "distinctness = 1 - avg_similarity. Higher means more distinct topics; alerts if LDA produces redundant output.", s["body"]))

    # Tech stack + file structure side by side
    story.append(Paragraph("Technology Stack and File Structure", s["h2"]))

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])

    tech_data = [
        [Paragraph("<b>Layer</b>", s["th"]), Paragraph("<b>Tech</b>", s["th"]), Paragraph("<b>Purpose</b>", s["th"])],
        [Paragraph("UI", s["tcc"]), Paragraph("Streamlit", s["tc"]), Paragraph("Interactive web dashboard", s["tc"])],
        [Paragraph("Data", s["tcc"]), Paragraph("pandas, NumPy", s["tc"]), Paragraph("Data manipulation", s["tc"])],
        [Paragraph("NLP", s["tcc"]), Paragraph("NLTK (VADER)", s["tc"]), Paragraph("Sentiment analysis, tokenization", s["tc"])],
        [Paragraph("ML", s["tcc"]), Paragraph("scikit-learn", s["tc"]), Paragraph("LDA topic modeling", s["tc"])],
        [Paragraph("Charts", s["tcc"]), Paragraph("Plotly", s["tc"]), Paragraph("Interactive visualizations", s["tc"])],
        [Paragraph("Trends", s["tcc"]), Paragraph("pytrends", s["tc"]), Paragraph("Google Trends API", s["tc"])],
    ]
    file_data = [
        [Paragraph("<b>File</b>", s["th"]), Paragraph("<b>Purpose</b>", s["th"])],
        [Paragraph("app.py", s["tc"]), Paragraph("Streamlit dashboard (7 pages, ~700 lines)", s["tc"])],
        [Paragraph("analysis.py", s["tc"]), Paragraph("Sentiment, topics, engagement, validation (~240 lines)", s["tc"])],
        [Paragraph("data_utils.py", s["tc"]), Paragraph("Data loading, cleaning, NLP preprocessing (~114 lines)", s["tc"])],
        [Paragraph("fetch_comments.py", s["tc"]), Paragraph("YouTube comment fetcher (~150 lines)", s["tc"])],
        [Paragraph("create_deck.py", s["tc"]), Paragraph("Generates 12-slide .pptx from data (~660 lines)", s["tc"])],
        [Paragraph("data/superbowl_ads.csv", s["tc"]), Paragraph("TidyTuesday dataset (247 rows, 25 cols)", s["tc"])],
    ]

    t1 = Table(tech_data, colWidths=[0.5*inch, 0.9*inch, 1.7*inch])
    t1.setStyle(ts)
    t2 = Table(file_data, colWidths=[1.15*inch, 2.65*inch])
    t2.setStyle(ts)

    combo = Table([[t1, t2]], colWidths=[3.25*inch, 3.95*inch])
    combo.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    story.append(combo)

    story.append(PageBreak())

    # ════════════════════════════════════════
    # PAGE 2: LAYMAN'S EXPLANATION
    # ════════════════════════════════════════
    story.append(Paragraph("Part 2: The Layman's Explanation", s["h1"]))
    story.append(Paragraph(
        "This section explains the project in plain English for anyone new to marketing analytics or AI.", s["italic"]))
    story.append(hr())

    story.append(Paragraph("What is the big picture?", s["h2"]))
    story.append(Paragraph(
        "Imagine you are a marketing manager at Doritos. You just spent <b>$7 million</b> on a 30-second Super Bowl "
        "ad. Your boss asks: \"Did it work? Should we do the same thing next year?\" Traditionally, you would look "
        "at TV ratings (\"113 million people watched\") or run a focus group. But that is vague. Did people actually "
        "<i>engage</i> with your ad? Did they <i>like</i> it? What did they <i>talk about</i>? Was it the humor "
        "that worked, or showing the chips in the first 5 seconds? <b>AdPulse answers these questions using data "
        "and AI.</b>", s["body"]))

    story.append(Paragraph("How does it work?", s["h2"]))
    story.append(Paragraph(
        "Think of AdPulse as doing five things a human analyst would do, but faster and at scale:", s["body"]))

    story.append(Paragraph("1. \"How popular was each ad?\" - Engagement Scoring", s["h3"]))
    story.append(Paragraph(
        "YouTube tells us views, likes, and comments for each ad, but raw numbers are hard to compare. An ad with "
        "5 million views and 100,000 likes is not obviously better than one with 2 million views and 200,000 likes. "
        "So we create a <b>single score from 0 to 100</b> that blends all three metrics. Think of it like a GPA: "
        "instead of individual grades, you get one number summarizing overall performance.", s["body"]))

    story.append(Paragraph("2. \"What makes an ad work?\" - Feature Impact", s["h3"]))
    story.append(Paragraph(
        "Every ad was tagged by reviewers: funny, celebrity, patriotic, shows product quickly, etc. We split all "
        "247 ads into groups (e.g., funny vs. not funny) and compare average views. The surprising answer: "
        "<b>showing your product quickly</b> matters far more than hiring a celebrity. Ads that show the product "
        "early get 70% more views. Celebrity ads actually get <i>fewer</i> views on average.", s["body"]))

    story.append(Paragraph("3. \"What is the vibe?\" - Sentiment Analysis", s["h3"]))
    story.append(Paragraph(
        "We feed each ad's text description into <b>VADER</b>, a program that reads words and determines the overall "
        "tone (positive, negative, or neutral). How? Imagine a huge dictionary where every word has a \"mood score\": "
        "\"amazing\" scores positive, \"terrible\" scores negative, \"table\" is neutral. VADER reads all words, "
        "accounts for negation (\"NOT amazing\") and intensifiers (\"VERY amazing\"), and outputs an overall score. "
        "It is not perfect (it does not truly <i>understand</i> language like a human), which is why we include "
        "sanity checks.", s["body"]))

    story.append(Paragraph("4. \"What are people talking about?\" - Topic Discovery", s["h3"]))
    story.append(Paragraph(
        "Reading 207 ad descriptions and summarizing themes manually would take hours. <b>LDA (Latent Dirichlet "
        "Allocation)</b> does this automatically. Think of it like sorting newspaper articles into piles by topic. "
        "You do not tell the algorithm what the topics are; it figures them out by looking at which words appear "
        "together. Sports articles cluster because they share \"game,\" \"team,\" \"score.\" LDA found 5 themes: "
        "Hyundai campaigns, Bud Light/Budweiser, recent ads, Doritos contests, and NFL/football.", s["body"]))

    story.append(Paragraph("5. \"Can we trust these results?\" - Output Validation", s["h3"]))
    story.append(Paragraph(
        "AI can give confident-sounding but wrong answers, so we built in three automatic checks:", s["body"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Distribution check:</b> If the tool says 98% of ads are \"positive,\" something "
        "is likely broken. We flag any result where one category dominates too heavily (>95%).", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Known-answer test:</b> We run the tool on sentences where we already know the "
        "answer (\"This is amazing\" = positive, \"This is terrible\" = negative). If it gets those wrong, we know "
        "something is off.", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Topic overlap check:</b> If all 5 \"topics\" contain the same words, they are "
        "not really different. We measure how distinct each topic is from the others.", s["bullet"]))

    story.append(Paragraph("Why does this matter?", s["h2"]))
    story.append(Paragraph(
        "Marketing has traditionally been a gut-feeling industry: \"I <i>think</i> humor works,\" \"I <i>feel</i> "
        "like we need a celebrity.\" AdPulse brings <b>data to the conversation</b>. Instead of guessing, you can "
        "see that across 20 years of Super Bowl data, showing your product quickly consistently outperforms celebrity "
        "appearances. And because the tool accepts any CSV of ad data, it is not limited to Super Bowl. A startup "
        "analyzing Instagram campaigns could upload their data and get the same analyses.", s["body"]))

    story.append(Paragraph("The Presentation", s["h2"]))
    story.append(Paragraph(
        "We also generated a 12-slide PowerPoint deck summarizing the problem, solution, data, technical approach, "
        "key findings, and output validation. It is designed for a 15-minute class presentation with a live demo "
        "of the interactive dashboard.", s["body"]))

    # Build
    doc.build(story)
    print(f"PDF saved to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
