"""
Report generation — JSON and Markdown output for senator profiles.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .fusion import SenatorProfile

log = logging.getLogger(__name__)


def generate_json_report(
    profile: SenatorProfile,
    output_path: Optional[str] = None,
) -> dict:
    """Generate a JSON report from a SenatorProfile."""
    report = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "0.1.0",
            "model_stack": {
                "embeddings": "Qwen/Qwen3-Embedding-0.6B",
                "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "emotion": "cardiffnlp/twitter-roberta-base-emotion",
                "offensive": "cardiffnlp/twitter-roberta-base-offensive",
                "irony": "cardiffnlp/twitter-roberta-base-irony",
                "hate": "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
                "toxicity": "s-nlp/roberta_toxicity_classifier",
            },
        },
        "senator": {
            "name": profile.senator_name,
            "twitter_handle": profile.twitter_handle,
            "party": profile.party,
            "state": profile.state,
        },
        "summary": {
            "n_tweets_analyzed": profile.n_tweets_analyzed,
            "date_range": profile.date_range,
            "compulsion_score": profile.compulsion_score,
            "virulence_score": profile.virulence_score,
            "overall_risk_score": profile.overall_risk_score,
        },
        "compulsion": {
            "score": profile.compulsion_score,
            "subscores": profile.compulsion_subscores,
        },
        "virulence": {
            "score": profile.virulence_score,
            "subscores": profile.virulence_subscores,
            "distribution": profile.virulence_distribution,
        },
        "classification_detail": {
            "sentiment_distribution": profile.sentiment_distribution,
            "emotion_distribution": profile.emotion_distribution,
            "toxicity_stats": profile.toxicity_stats,
        },
        "top_rage_tweets": profile.top_rage_tweets,
        "disclaimers": [
            "This analysis does not constitute a clinical diagnosis of addiction, "
            "compulsion, or mental health condition.",
            "Scores are derived from automated classifiers with known error rates "
            "and should not be treated as ground truth.",
            "Temporal analysis uses UTC timestamps which may not reflect the "
            "poster's local timezone.",
            "Classifier models were trained on general Twitter data, not "
            "specifically on political speech.",
        ],
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("JSON report saved to %s", p)

    return report


def generate_markdown_report(
    profile: SenatorProfile,
    output_path: Optional[str] = None,
) -> str:
    """Generate a Markdown report from a SenatorProfile."""

    lines = []
    lines.append(f"# X-Box Analysis: {profile.senator_name}")
    lines.append(f"**@{profile.twitter_handle}** | {profile.party} | {profile.state}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- **Tweets analyzed**: {profile.n_tweets_analyzed:,}")
    lines.append(f"- **Date range**: {profile.date_range}")
    lines.append(f"- **Compulsion score**: {profile.compulsion_score}/100")
    lines.append(f"- **Virulence score**: {profile.virulence_score}/100")
    lines.append(f"- **Overall risk score**: {profile.overall_risk_score}/100")
    lines.append("")

    # Compulsion breakdown
    lines.append("## Compulsion-Like Behavior")
    lines.append("| Dimension | Score |")
    lines.append("| --- | ---: |")
    for k, v in profile.compulsion_subscores.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
    lines.append("")

    # Virulence breakdown
    lines.append("## Virulence Analysis")
    lines.append("| Dimension | Score |")
    lines.append("| --- | ---: |")
    for k, v in profile.virulence_subscores.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
    lines.append("")

    # Classification detail
    if profile.sentiment_distribution:
        lines.append("### Sentiment Distribution")
        lines.append("| Label | Share |")
        lines.append("| --- | ---: |")
        for k, v in sorted(profile.sentiment_distribution.items()):
            lines.append(f"| {k} | {v:.1%} |")
        lines.append("")

    if profile.emotion_distribution:
        lines.append("### Emotion Distribution")
        lines.append("| Emotion | Share |")
        lines.append("| --- | ---: |")
        for k, v in sorted(profile.emotion_distribution.items()):
            lines.append(f"| {k} | {v:.1%} |")
        lines.append("")

    if profile.toxicity_stats:
        lines.append("### Toxicity")
        tox = profile.toxicity_stats
        lines.append(f"- Mean toxicity score: {tox.get('mean', 0):.4f}")
        lines.append(f"- % classified toxic: {tox.get('pct_toxic', 0):.2f}%")
        lines.append(f"- P90 toxicity: {tox.get('p90', 0):.4f}")
        lines.append("")

    # Top rage events
    if profile.top_rage_tweets:
        lines.append("## Top Rage Events")
        lines.append("| Date | Virulence | Outrage | Ad Hominem | Text |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for evt in profile.top_rage_tweets[:10]:
            date = str(evt.get("created_at", ""))[:10]
            text = evt.get("text", "")[:80].replace("|", "\\|")
            lines.append(
                f"| {date} | {evt.get('composite_virulence', 0):.3f} "
                f"| {evt.get('outrage_intensity', 0):.3f} "
                f"| {evt.get('ad_hominem', 0):.3f} "
                f"| {text}... |"
            )
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("- **Embeddings**: Qwen/Qwen3-Embedding-0.6B (MTEB #1 under 1B params)")
    lines.append("- **Sentiment**: cardiffnlp/twitter-roberta-base-sentiment-latest")
    lines.append("- **Emotion**: cardiffnlp/twitter-roberta-base-emotion (anger/joy/optimism/sadness)")
    lines.append("- **Offensive**: cardiffnlp/twitter-roberta-base-offensive")
    lines.append("- **Irony**: cardiffnlp/twitter-roberta-base-irony")
    lines.append("- **Hate speech**: cardiffnlp/twitter-roberta-base-hate-multiclass-latest")
    lines.append("- **Toxicity**: s-nlp/roberta_toxicity_classifier")
    lines.append("- **Behavioral**: Temporal/metadata features with sigmoid-scaled scoring")
    lines.append("")

    # Disclaimers
    lines.append("## Disclaimers")
    lines.append("- This analysis does not constitute a clinical diagnosis.")
    lines.append("- Classifier scores are probabilistic and subject to error.")
    lines.append("- UTC timestamps may not reflect the poster's local timezone.")
    lines.append("- Models trained on general Twitter data, not political speech specifically.")
    lines.append("")

    text = "\n".join(lines)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        log.info("Markdown report saved to %s", p)

    return text
