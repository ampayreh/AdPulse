# Prompt for Gemini in Google Colab

Paste the following into Gemini in a Google Colab notebook to replicate the AdPulse project as a single notebook.

---

Create a Google Colab notebook called **"AdPulse — AI-Powered Ad Campaign Performance Analyzer"** for a university course project (MSIS 521: IT and Marketing in the New Economy, University of Washington, Winter 2025).

## What the project does

AdPulse analyzes ad campaign performance using AI/NLP techniques. It is demonstrated with Super Bowl ads but designed to work with any ad campaign data. The core analyses are:

1. **Engagement scoring** — composite score (0–100) weighted: views 40%, likes 35%, comments 25%
2. **Feature impact analysis** — compares average views for ads with vs. without characteristics (funny, celebrity, patriotic, danger, animals, use_sex, show_product_quickly) and computes percentage lift
3. **Sentiment analysis** — VADER (NLTK) on ad descriptions, classifying as positive/neutral/negative using compound score thresholds (≥0.05 positive, ≤-0.05 negative)
4. **Topic modeling** — LDA (sklearn) with 5 topics on preprocessed ad descriptions
5. **Output validation** — sentiment distribution check (flags if >95% one class), known-sample accuracy check (5 test cases), topic distinctness score (cosine distance between topic-word distributions)
6. **Brand performance summary** — aggregated stats by brand

## Dataset

Load the TidyTuesday Super Bowl ads dataset directly from this URL:
```
https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-02/youtube.csv
```

This CSV has 247 rows and 25 columns including: year, brand, youtube_url, funny, show_product_quickly, patriotic, celebrity, danger, animals, use_sex, view_count, like_count, dislike_count, favorite_count, comment_count, title, description, thumbnail, channel_title.

The boolean columns (funny, show_product_quickly, patriotic, celebrity, danger, animals, use_sex) come as R-style TRUE/FALSE strings and need to be mapped to Python booleans.

## Notebook structure

Create the notebook with these sections, each in its own cell(s):

### Cell 1: Setup & Installs
```python
!pip install nltk scikit-learn wordcloud matplotlib plotly pandas
```

### Cell 2: Imports & Data Loading
- Import pandas, numpy, matplotlib, plotly, sklearn, nltk, wordcloud
- Download NLTK data: punkt_tab, stopwords, wordnet, vader_lexicon
- Load the CSV from the URL above
- Clean boolean columns (map TRUE/FALSE strings to Python booleans)
- Clean numeric columns (coerce to numeric, fill NaN with 0)
- Engineer features: `engagement_rate = (like_count + comment_count) / view_count`, `like_ratio = like_count / (like_count + dislike_count)`

### Cell 3: Text Preprocessing
- Function to preprocess text: remove URLs, remove punctuation, lowercase, tokenize with NLTK word_tokenize, remove stopwords, lemmatize with WordNetLemmatizer, keep tokens with length > 2

### Cell 4: Exploratory Data Analysis
- Show dataset shape, column names, dtypes
- Display first few rows
- Show summary statistics for numeric columns
- Plot: Ads per year (bar chart)
- Plot: Top 10 brands by number of ads

### Cell 5: Engagement Analysis
- Compute composite engagement score (normalize view_count, like_count, comment_count to 0-1, then weighted sum × 100)
- Show top 10 ads by composite score
- Plot: Brand performance summary (average views by brand, horizontal bar chart)

### Cell 6: Feature Impact Analysis
- For each boolean feature (funny, celebrity, patriotic, etc.), compute average views WITH the feature vs WITHOUT
- Compute views_lift percentage = (avg_with / avg_without - 1) × 100
- Display as a horizontal bar chart sorted by lift, with positive bars in teal and negative in coral
- Print key insight: which features help and which hurt

### Cell 7: Sentiment Analysis
- Use NLTK VADER SentimentIntensityAnalyzer on the "description" column (filter non-null descriptions)
- Classify: compound ≥ 0.05 → positive, ≤ -0.05 → negative, else neutral
- Show sentiment distribution (pie chart)
- Show average compound score by brand
- Show example texts for each sentiment class

### Cell 8: Sentiment Output Validation
- **Distribution check**: flag if any single sentiment class is >95% of results
- **Known-sample check**: run VADER on 5 known-sentiment test texts:
  - "This is absolutely amazing, best product ever!" → positive
  - "Terrible experience, total waste of money." → negative
  - "The meeting is scheduled for Tuesday." → neutral
  - "I love this incredible ad, so funny!" → positive
  - "Worst commercial I have ever seen, awful." → negative
- Report accuracy (correct/total)

### Cell 9: Topic Modeling (LDA)
- Preprocess descriptions using the text preprocessing function from Cell 3
- Use CountVectorizer (max_df=0.95, min_df=2, max_features=1000, stop_words="english")
- Fit LDA with 5 topics (sklearn LatentDirichletAllocation, random_state=42, max_iter=20)
- Display top 10 words per topic
- Assign dominant topic to each ad
- Show topic distribution across ads (bar chart)

### Cell 10: Topic Quality Evaluation
- Compute topic distinctness: normalize LDA components, compute pairwise cosine similarity, average off-diagonal similarity, distinctness = 1 - avg_similarity
- Report the score (higher = more distinct topics)
- Generate word cloud for each topic

### Cell 11: Key Findings Summary
- Print a markdown summary of the key findings:
  - Which ad features drive the most views
  - Sentiment breakdown
  - Top brands by engagement
  - Topic themes discovered
  - Output validation results

## Important notes

- Use plotly for interactive charts where possible, matplotlib for word clouds
- Include markdown cells between code cells explaining each analysis step
- Every chart should have a title and axis labels
- Use a consistent color scheme: teal (#0891B2) for positive/primary, coral (#FF6B6B) for negative/accent, navy (#1A1F36) for dark backgrounds
- The notebook should run end-to-end without errors in Google Colab
- Do NOT use Streamlit — this is a pure notebook implementation of the same analyses
- Include output validation / sanity checking as this is a grading requirement
