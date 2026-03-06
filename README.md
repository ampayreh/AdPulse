# AdPulse — AI-Powered Ad Campaign Performance Analyzer

MSIS 521 Course Project | University of Washington | Winter 2025

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/ampayreh/AdPulse.git
cd AdPulse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What's Inside

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (7 pages) |
| `analysis.py` | Sentiment analysis, topic modeling, engagement scoring |
| `data_utils.py` | Data loading, cleaning, NLP preprocessing |
| `fetch_comments.py` | YouTube comment fetcher (optional) |
| `create_deck.py` | Generates `AdPulse_Presentation.pptx` from data |
| `data/superbowl_ads.csv` | TidyTuesday dataset (247 ads, 10 brands, 2000–2020) |
| `AdPulse_Presentation.pptx` | Presentation deck (12 slides) |

## Presentation

Open `AdPulse_Presentation.pptx` directly in PowerPoint or Keynote. To regenerate it from the latest data:

```bash
pip install python-pptx matplotlib
python create_deck.py
```
