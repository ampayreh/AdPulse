# AdPulse — AI-Powered Ad Campaign Performance Analyzer

Quantitative analysis of Super Bowl advertisements (247 ads, 10 brands, 2000–2020) with VADER sentiment scoring, LDA topic modeling, composite engagement metrics, and a Claude-powered insight agent that produces grounded narrative briefs with automated fact-checking.

MSIS 521 Course Project | University of Washington | Winter 2025

## Setup

```bash
git clone https://github.com/ampayreh/AdPulse.git
cd AdPulse
pip install -r requirements.txt
```

## What's Inside

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (7 pages) |
| `analysis.py` | Sentiment analysis (VADER/BERT), LDA topic modeling, engagement scoring |
| `insight_agent.py` | Claude-powered narrator + critic insight agent |
| `data_utils.py` | Data loading, cleaning, NLP preprocessing |
| `fetch_comments.py` | YouTube comment fetcher (optional) |
| `data/superbowl_ads.csv` | TidyTuesday dataset (247 ads, 10 brands, 2000–2020) |

## Insight Agent

The `insight_agent.py` adds a Claude-powered analysis layer on top of the computed artifacts. It uses a **narrator + critic** pattern:

1. **Compute** — runs the full analysis pipeline (VADER sentiment, LDA topics, engagement composites, feature impact, brand summaries)
2. **Narrator** — Claude reads the computed artifacts and produces a 600–800 word insight brief, grounding every claim in specific numbers
3. **Critic** — a second Claude call fact-checks the narrator's claims against the underlying data, flagging unsupported, misleading, or fabricated assertions

```bash
# See what data the agent will analyze (no API call)
python insight_agent.py --dry-run

# Full narrator + critic analysis
export ANTHROPIC_API_KEY=sk-ant-...
python insight_agent.py

# Narrator only (skip fact-checking)
python insight_agent.py --narrator-only

# JSON output with metrics and artifacts
python insight_agent.py --format json --output analysis.json
```

The critic returns a structured verdict:
- **claims_checked / claims_supported** — how many factual claims were verified
- **issues** — each unsupported or misleading claim with severity and explanation
- **overall_verdict** — `grounded`, `mostly_grounded`, `partially_grounded`, or `unreliable`

### Why narrator + critic?

The quantitative analysis (VADER, LDA, engagement scores) produces numbers. Numbers need interpretation to be actionable — but LLM-generated interpretation can hallucinate patterns that aren't in the data. The critic pass catches this:

- A narrator claim of "nearly 50%" when the actual figure is 43% → flagged as misleading
- A claim about a time trend with no supporting time-series data → flagged as unsupported
- A number that doesn't appear anywhere in the artifacts → flagged as fabricated

This is the same adversarial-verification pattern used in the [ScopingAgent](https://github.com/ampayreh/ScopingAgent) eval harness and the [LMMSmartClinicAI](https://github.com/ampayreh/LMMSmartClinicAI) clinical safety evals.

## Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with 7 pages covering sentiment analysis, topic modeling, engagement scoring, brand comparison, feature impact, clustering, and network analysis.

## Dataset

[TidyTuesday Super Bowl Ads](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-03-02) — 247 ads from 10 brands (2000–2020) with YouTube engagement metrics, video descriptions, and 7 boolean ad characteristics (funny, celebrity, animals, danger, patriotic, show_product_quickly, use_sex).

## Presentation

To regenerate the presentation deck from the latest data:

```bash
pip install python-pptx matplotlib
python create_deck.py
```

## Author

**Graeme Tobias Ampeire** — Applied AI Architect
MSIS, University of Washington
