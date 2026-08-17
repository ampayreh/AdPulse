"""
AdPulse Insight Agent — narrator + critic pattern for ad campaign analysis.

Reads the computed artifacts from analysis.py (VADER sentiment distribution,
LDA topics, engagement composites, feature impact, brand summaries) and
produces a grounded narrative insight brief. A critic pass then checks every
claim the narrator makes against the underlying numbers.

Usage:
    python insight_agent.py                    # full analysis + critique
    python insight_agent.py --narrator-only    # skip the critic pass
    python insight_agent.py --dry-run          # print data summary, no API call
    python insight_agent.py --format json      # structured output
    python insight_agent.py --output brief.md  # write to file
"""

import argparse
import json
import sys
import time

from data_utils import load_superbowl_data, preprocess_text
from analysis import (
    SentimentAnalyzer,
    SentimentEvaluator,
    TopicModeler,
    EngagementAnalyzer,
)


def compute_artifacts(df):
    """Run the full analysis pipeline and return structured artifacts."""

    # ── Sentiment ────────────────────────────────────────────────────
    sa = SentimentAnalyzer(method="vader")
    descriptions = df["description"].fillna("").tolist()
    sentiment_df = sa.analyze(descriptions)
    dist_check = SentimentEvaluator.distribution_check(sentiment_df)

    sentiment_by_brand = {}
    for brand in df["brand"].unique():
        mask = df["brand"] == brand
        brand_descs = df.loc[mask, "description"].fillna("").tolist()
        brand_sent = sa.analyze(brand_descs)
        sentiment_by_brand[brand] = {
            "mean_compound": round(brand_sent["compound"].mean(), 3),
            "positive_pct": round(
                (brand_sent["sentiment"] == "positive").mean() * 100, 1
            ),
            "negative_pct": round(
                (brand_sent["sentiment"] == "negative").mean() * 100, 1
            ),
            "neutral_pct": round(
                (brand_sent["sentiment"] == "neutral").mean() * 100, 1
            ),
            "count": len(brand_sent),
        }

    # ── Topics ───────────────────────────────────────────────────────
    processed = [preprocess_text(d) for d in descriptions]
    non_empty = [t for t in processed if t.strip()]
    tm = TopicModeler(n_topics=5, max_features=500)
    if len(non_empty) >= 10:
        tm.fit_transform(non_empty)
        topics = tm.get_top_words(n_words=8)
        distinctness = tm.coherence_proxy(non_empty)
    else:
        topics = {}
        distinctness = None

    # ── Engagement ───────────────────────────────────────────────────
    ea = EngagementAnalyzer(df)
    scored_df = ea.compute_composite_score()
    feature_impact = ea.feature_impact()
    brand_summary = ea.brand_summary()

    # Top and bottom ads by composite score
    if "composite_score" in scored_df.columns:
        top_5 = (
            scored_df.nlargest(5, "composite_score")[
                ["brand", "year", "title_display", "composite_score",
                 "view_count", "like_count"]
            ]
            .to_dict("records")
        )
        bottom_5 = (
            scored_df.nsmallest(5, "composite_score")[
                ["brand", "year", "title_display", "composite_score",
                 "view_count", "like_count"]
            ]
            .to_dict("records")
        )
    else:
        top_5, bottom_5 = [], []

    return {
        "dataset": {
            "total_ads": len(df),
            "brands": sorted(df["brand"].unique().tolist()),
            "year_range": [int(df["year"].min()), int(df["year"].max())],
        },
        "sentiment": {
            "overall_distribution": dist_check["distribution"],
            "dominant_pct": dist_check["dominant_pct"],
            "flags": dist_check["flags"],
            "mean_compound": round(sentiment_df["compound"].mean(), 3),
            "by_brand": sentiment_by_brand,
        },
        "topics": {
            name: [(word, round(weight, 2)) for word, weight in words]
            for name, words in topics.items()
        },
        "topic_distinctness": distinctness,
        "engagement": {
            "top_5_ads": top_5,
            "bottom_5_ads": bottom_5,
            "feature_impact": feature_impact.to_dict("records")
            if not feature_impact.empty else [],
            "brand_summary": brand_summary.to_dict("records")
            if not brand_summary.empty else [],
        },
    }


# ── Narrator prompt ──────────────────────────────────────────────────

NARRATOR_SYSTEM = """\
You are an advertising analytics narrator. You have been given the computed
results of a quantitative analysis of {total_ads} Super Bowl advertisements
from {brands} brands, spanning {year_min}–{year_max}.

The analysis includes:
- VADER sentiment scores on ad descriptions
- LDA topic modeling (5 topics)
- Composite engagement scores (40% views + 35% likes + 25% comments)
- Feature impact analysis (funny, celebrity, animals, danger, etc.)
- Brand-level performance summaries

Your job: produce a concise, insight-rich narrative brief (600–800 words) that
a marketing executive could read in 3 minutes. Structure it as:

1. **Headline finding** — the single most interesting pattern in the data
2. **What drives engagement** — which ad characteristics correlate with higher
   performance, grounded in the feature impact numbers
3. **Brand positioning** — how the top brands differ in strategy (tone,
   features, engagement) based on the brand summary data
4. **Sentiment landscape** — what the sentiment distribution reveals (and its
   limitations — VADER on short ad descriptions has known blind spots)
5. **Topic themes** — what the LDA topics suggest about ad content strategies

Rules:
- Every claim must reference a specific number from the data provided.
- Do not speculate beyond what the numbers show.
- Flag limitations: VADER is a lexicon (not deep learning), LDA topics may
  be noisy on short text, engagement metrics are YouTube-only, the dataset
  covers 2000–2020 (pre-2021 trends may not reflect current patterns).
- Do not fabricate statistics or claim patterns not present in the data.
"""

NARRATOR_USER = """\
Here are the computed analysis artifacts:

{artifacts_json}

Produce the narrative insight brief.
"""


# ── Critic prompt ────────────────────────────────────────────────────

CRITIC_SYSTEM = """\
You are a quantitative fact-checker reviewing a narrative brief about Super
Bowl ad campaign performance. You have been given:

1. The narrative brief (produced by a narrator)
2. The underlying computed data artifacts

Your job: check every factual claim in the narrative against the data.
For each claim that references a number, percentage, ranking, or comparison:

- Verify it matches the data exactly
- Flag any claim that exaggerates, rounds misleadingly, or draws a conclusion
  the data does not support
- Flag any claim made without supporting data

Output a structured JSON object:
{
  "claims_checked": <int>,
  "claims_supported": <int>,
  "claims_unsupported": <int>,
  "claims_misleading": <int>,
  "issues": [
    {
      "claim": "<the exact text from the narrative>",
      "problem": "unsupported | misleading | fabricated | rounding_error",
      "explanation": "<what the data actually shows>",
      "severity": "minor | moderate | major"
    }
  ],
  "overall_verdict": "grounded | mostly_grounded | partially_grounded | unreliable",
  "suggested_corrections": ["<correction 1>", ...]
}

Be strict. A claim that says "nearly 50%" when the actual figure is 43% is
misleading. A claim about a trend that has no supporting time-series data is
unsupported. A claim that names a number not present in the artifacts is
fabricated.
"""

CRITIC_USER = """\
## Narrative Brief

{narrative}

## Underlying Data Artifacts

{artifacts_json}

Check every factual claim in the narrative against the data artifacts.
Return your analysis as the JSON structure described in your instructions.
"""


def run_narrator(artifacts: dict) -> tuple[str, dict]:
    """Run the narrator to produce a narrative brief. Returns (text, metrics)."""
    import anthropic

    client = anthropic.Anthropic()
    ds = artifacts["dataset"]

    system = NARRATOR_SYSTEM.format(
        total_ads=ds["total_ads"],
        brands=len(ds["brands"]),
        year_min=ds["year_range"][0],
        year_max=ds["year_range"][1],
    )
    user_msg = NARRATOR_USER.format(
        artifacts_json=json.dumps(artifacts, indent=2, default=str)
    )

    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = "".join(b.text for b in response.content if b.type == "text")
    elapsed = time.time() - start

    metrics = {
        "role": "narrator",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": round(elapsed, 1),
    }
    return text, metrics


def run_critic(narrative: str, artifacts: dict) -> tuple[dict, dict]:
    """Run the critic to fact-check the narrative. Returns (verdict, metrics)."""
    import anthropic

    client = anthropic.Anthropic()
    user_msg = CRITIC_USER.format(
        narrative=narrative,
        artifacts_json=json.dumps(artifacts, indent=2, default=str),
    )

    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = "".join(b.text for b in response.content if b.type == "text")
    elapsed = time.time() - start

    # Parse the JSON from the critic's response
    try:
        # Find JSON block in the response
        json_start = text.index("{")
        json_end = text.rindex("}") + 1
        verdict = json.loads(text[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        verdict = {"error": "Failed to parse critic output", "raw": text}

    metrics = {
        "role": "critic",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": round(elapsed, 1),
    }
    return verdict, metrics


def format_markdown(narrative: str, verdict: dict | None, metrics: list) -> str:
    """Format the full output as markdown."""
    parts = ["# AdPulse Insight Brief\n", narrative]

    if verdict and "error" not in verdict:
        parts.append("\n\n---\n\n## Fact-Check Results\n")
        parts.append(
            f"**Verdict:** {verdict.get('overall_verdict', 'unknown')} "
            f"({verdict.get('claims_supported', '?')}/{verdict.get('claims_checked', '?')} "
            f"claims supported)\n"
        )
        issues = verdict.get("issues", [])
        if issues:
            parts.append(f"\n**Issues found ({len(issues)}):**\n")
            for issue in issues:
                sev = issue.get("severity", "?")
                parts.append(
                    f"- [{sev}] \"{issue.get('claim', '')}\" — "
                    f"{issue.get('explanation', '')}\n"
                )
        corrections = verdict.get("suggested_corrections", [])
        if corrections:
            parts.append("\n**Suggested corrections:**\n")
            for c in corrections:
                parts.append(f"- {c}\n")

    # Metrics footer
    total_tokens = sum(m.get("input_tokens", 0) + m.get("output_tokens", 0) for m in metrics)
    total_time = sum(m.get("latency_seconds", 0) for m in metrics)
    cost = sum(
        m.get("input_tokens", 0) * 3 / 1_000_000
        + m.get("output_tokens", 0) * 15 / 1_000_000
        for m in metrics
    )
    parts.append(
        f"\n\n---\n*Generated by AdPulse Insight Agent "
        f"(narrator{'+ critic' if verdict else ' only'}) | "
        f"{total_time:.1f}s | {total_tokens} tokens | ~${cost:.4f}*\n"
    )
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="AdPulse Insight Agent — narrator + critic analysis"
    )
    parser.add_argument(
        "--narrator-only",
        action="store_true",
        help="Skip the critic pass",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute artifacts and print summary, no API call",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    parser.add_argument("--output", type=str, help="Write to file")
    args = parser.parse_args()

    # Load data and compute artifacts
    print("Loading Super Bowl ads dataset...", file=sys.stderr)
    df = load_superbowl_data()
    print(f"  {len(df)} ads, {df['brand'].nunique()} brands, "
          f"{df['year'].min()}–{df['year'].max()}", file=sys.stderr)

    print("Computing analysis artifacts...", file=sys.stderr)
    artifacts = compute_artifacts(df)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Dataset: {artifacts['dataset']}")
        print(f"\nSentiment distribution: {artifacts['sentiment']['overall_distribution']}")
        print(f"Mean compound: {artifacts['sentiment']['mean_compound']}")
        if artifacts['sentiment']['flags']:
            for flag in artifacts['sentiment']['flags']:
                print(f"  ⚠ {flag}")
        print(f"\nTopics: {len(artifacts['topics'])} discovered")
        for name, words in artifacts['topics'].items():
            print(f"  {name}: {', '.join(w for w, _ in words[:5])}")
        print(f"  Distinctness: {artifacts['topic_distinctness']}")
        print(f"\nEngagement: {len(artifacts['engagement']['feature_impact'])} features analyzed")
        for fi in artifacts['engagement']['feature_impact']:
            print(f"  {fi['feature']}: {fi['views_lift']:+.1f}% views lift")
        print(f"\nBrand summary: {len(artifacts['engagement']['brand_summary'])} brands")
        for bs in artifacts['engagement']['brand_summary'][:5]:
            print(f"  {bs['brand']}: {bs['total_ads']} ads, "
                  f"avg views {bs['avg_views']:,.0f}")
        return

    # Narrator
    print("Running narrator...", file=sys.stderr)
    narrative, narrator_metrics = run_narrator(artifacts)
    print(f"  {narrator_metrics['output_tokens']} tokens, "
          f"{narrator_metrics['latency_seconds']}s", file=sys.stderr)
    all_metrics = [narrator_metrics]

    # Critic
    verdict = None
    if not args.narrator_only:
        print("Running critic...", file=sys.stderr)
        verdict, critic_metrics = run_critic(narrative, artifacts)
        print(f"  {critic_metrics['output_tokens']} tokens, "
              f"{critic_metrics['latency_seconds']}s", file=sys.stderr)
        all_metrics.append(critic_metrics)

        v = verdict.get("overall_verdict", "unknown")
        supported = verdict.get("claims_supported", "?")
        checked = verdict.get("claims_checked", "?")
        issues = len(verdict.get("issues", []))
        print(f"  Verdict: {v} ({supported}/{checked} supported, "
              f"{issues} issues)", file=sys.stderr)

    # Output
    if args.format == "json":
        output = json.dumps({
            "narrative": narrative,
            "fact_check": verdict,
            "artifacts": artifacts,
            "metrics": all_metrics,
        }, indent=2, default=str)
    else:
        output = format_markdown(narrative, verdict, all_metrics)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
