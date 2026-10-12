"""
Score fusion — combine text classification signals with behavioral profile
into composite virulence and compulsion scores per senator.

Architecture:
1. Per-tweet virulence vector from multi-head classifier
2. Per-senator behavioral profile from temporal analysis
3. Aggregate tweet-level scores to senator-level
4. Fuse with behavioral signals
5. Output: composite profile with uncertainty estimates
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .behavioral import BehavioralProfile
from .config import VIRULENCE_WEIGHTS

log = logging.getLogger(__name__)


@dataclass
class VirulenceVector:
    """Per-tweet virulence decomposition."""
    outrage_intensity: float = 0.0      # sentiment_negative * emotion_anger
    fear_mongering: float = 0.0         # negative sentiment + threat-like toxicity
    ad_hominem: float = 0.0             # offensive + hate targeting individuals
    divisive_framing: float = 0.0       # irony + controversial engagement pattern
    engagement_bait: float = 0.0        # high reply ratio + question marks + caps
    composite_virulence: float = 0.0    # weighted combination


@dataclass
class SenatorProfile:
    """Complete fused profile for a senator."""
    senator_name: str = ""
    twitter_handle: str = ""
    party: str = ""
    state: str = ""

    # Behavioral scores
    compulsion_score: float = 0.0
    compulsion_subscores: dict = field(default_factory=dict)

    # Virulence scores (aggregated from tweets)
    virulence_score: float = 0.0
    virulence_subscores: dict = field(default_factory=dict)
    virulence_distribution: dict = field(default_factory=dict)

    # Classification distributions
    sentiment_distribution: dict = field(default_factory=dict)
    emotion_distribution: dict = field(default_factory=dict)
    toxicity_stats: dict = field(default_factory=dict)

    # Combined
    overall_risk_score: float = 0.0

    # Top rage events
    top_rage_tweets: list = field(default_factory=list)

    # Metadata
    n_tweets_analyzed: int = 0
    date_range: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ScoreFusion:
    """Fuse per-tweet classifications with behavioral profile."""

    def compute_tweet_virulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-tweet virulence vectors from classification columns.
        Expects df to have columns from MultiHeadClassifier.classify_tweets().
        """
        out = df.copy()

        # Outrage intensity: negative sentiment * anger emotion
        neg = df.get("sentiment_negative", pd.Series(0, index=df.index))
        anger = df.get("emotion_anger", pd.Series(0, index=df.index))
        out["outrage_intensity"] = (neg * 0.6 + anger * 0.4).clip(0, 1)

        # Fear-mongering: negative sentiment * toxicity (threat-like)
        tox = df.get("toxicity_score", pd.Series(0, index=df.index))
        out["fear_mongering"] = (neg * 0.5 + tox * 0.5).clip(0, 1)

        # Ad hominem: offensive + hate score
        off = df.get("offensive_score", pd.Series(0, index=df.index))
        hate = df.get("hate_score", pd.Series(0, index=df.index))
        out["ad_hominem"] = (off * 0.6 + hate * 0.4).clip(0, 1)

        # Divisive framing: irony + engagement controversy proxy
        irony = df.get("irony_score", pd.Series(0, index=df.index))
        # Controversy proxy: high reply-to-like ratio
        controversy = pd.Series(0, index=df.index)
        if "reply_count" in df.columns and "like_count" in df.columns:
            likes = df["like_count"].clip(lower=1)
            controversy = (df["reply_count"] / likes).clip(0, 1)
        out["divisive_framing"] = (irony * 0.5 + controversy * 0.5).clip(0, 1)

        # Engagement bait: text features
        if "text" in df.columns:
            text = df["text"]
            has_question = text.str.contains(r"\?", regex=True).astype(float)
            has_caps = (
                text.str.count(r"[A-Z]") / text.str.len().clip(lower=1)
            ).clip(0, 1)
            has_exclaim = text.str.contains(r"!{2,}", regex=True).astype(float)
            out["engagement_bait"] = (
                has_question * 0.4 + has_caps * 0.3 + has_exclaim * 0.3
            ).clip(0, 1)
        else:
            out["engagement_bait"] = 0.0

        # Composite virulence
        components = ["outrage_intensity", "fear_mongering", "ad_hominem",
                       "divisive_framing", "engagement_bait"]
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        out["composite_virulence"] = sum(
            out[c] * w for c, w in zip(components, weights)
        ).clip(0, 1)

        return out

    def aggregate_senator_profile(
        self,
        classified_df: pd.DataFrame,
        behavioral: BehavioralProfile,
        senator_name: str = "",
        twitter_handle: str = "",
        party: str = "",
        state: str = "",
        top_n_rage: int = 10,
    ) -> SenatorProfile:
        """
        Aggregate tweet-level classifications into a senator-level profile
        and fuse with behavioral analysis.
        """
        profile = SenatorProfile(
            senator_name=senator_name or behavioral.senator_name,
            twitter_handle=twitter_handle or behavioral.twitter_handle,
            party=party,
            state=state,
            n_tweets_analyzed=len(classified_df),
        )

        if classified_df.empty:
            return profile

        # Date range
        if "created_at" in classified_df.columns:
            dates = pd.to_datetime(classified_df["created_at"])
            profile.date_range = f"{dates.min().date()} to {dates.max().date()}"

        # ── Behavioral scores ─────────────────────────
        profile.compulsion_score = behavioral.compulsion_score
        profile.compulsion_subscores = behavioral.compulsion_subscores

        # ── Virulence aggregation ─────────────────────
        virulence_cols = [
            "outrage_intensity", "fear_mongering", "ad_hominem",
            "divisive_framing", "engagement_bait", "composite_virulence",
        ]
        available_v = [c for c in virulence_cols if c in classified_df.columns]

        if available_v:
            profile.virulence_subscores = {
                col: round(float(classified_df[col].mean()) * 100, 1)
                for col in available_v
            }
            profile.virulence_distribution = {
                col: {
                    "mean": round(float(classified_df[col].mean()), 4),
                    "std": round(float(classified_df[col].std()), 4),
                    "p50": round(float(classified_df[col].median()), 4),
                    "p90": round(float(classified_df[col].quantile(0.9)), 4),
                    "p99": round(float(classified_df[col].quantile(0.99)), 4),
                }
                for col in available_v
            }
            if "composite_virulence" in classified_df.columns:
                # Scale to 0-100
                profile.virulence_score = round(
                    float(classified_df["composite_virulence"].mean()) * 100, 1
                )

        # ── Sentiment distribution ────────────────────
        if "sentiment_label" in classified_df.columns:
            profile.sentiment_distribution = (
                classified_df["sentiment_label"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            )

        # ── Emotion distribution ──────────────────────
        if "emotion_label" in classified_df.columns:
            profile.emotion_distribution = (
                classified_df["emotion_label"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            )

        # ── Toxicity stats ────────────────────────────
        if "toxicity_score" in classified_df.columns:
            tox = classified_df["toxicity_score"]
            profile.toxicity_stats = {
                "mean": round(float(tox.mean()), 4),
                "pct_toxic": round(float((tox > 0.5).mean()) * 100, 2),
                "p90": round(float(tox.quantile(0.9)), 4),
                "max": round(float(tox.max()), 4),
            }

        # ── Top rage tweets ───────────────────────────
        if "composite_virulence" in classified_df.columns:
            rage_df = classified_df.nlargest(top_n_rage, "composite_virulence")
            for _, row in rage_df.iterrows():
                event = {
                    "text": str(row.get("text", ""))[:200],
                    "composite_virulence": round(float(row.get("composite_virulence", 0)), 4),
                    "outrage_intensity": round(float(row.get("outrage_intensity", 0)), 4),
                    "ad_hominem": round(float(row.get("ad_hominem", 0)), 4),
                    "divisive_framing": round(float(row.get("divisive_framing", 0)), 4),
                }
                if "created_at" in row:
                    event["created_at"] = str(row["created_at"])
                if "tweet_id" in row:
                    event["tweet_id"] = str(row["tweet_id"])
                engagement_cols = ["like_count", "retweet_count", "reply_count", "quote_count"]
                eng = sum(row.get(c, 0) for c in engagement_cols if c in row.index)
                if eng > 0:
                    event["engagement"] = int(eng)
                profile.top_rage_tweets.append(event)

        # ── Overall risk score ────────────────────────
        # Weighted fusion of compulsion and virulence
        profile.overall_risk_score = round(
            profile.compulsion_score * 0.4 + profile.virulence_score * 0.6, 1
        )

        return profile
