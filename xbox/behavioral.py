"""
Compulsive usage behavioral analyzer — temporal and metadata features.

Analyzes posting patterns to score compulsion-like behavior:
- Posting frequency and volume
- Burst detection (clusters of rapid posts)
- Night activity patterns
- Session analysis (reactive loops)
- Reply/quote/RT ratios
- Repetition patterns
- Engagement distribution skew
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    BURST_WINDOW_MINUTES,
    COMPULSION_WEIGHTS,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    SESSION_GAP_MINUTES,
)

log = logging.getLogger(__name__)


@dataclass
class BehavioralProfile:
    """Complete behavioral profile for a single account."""
    senator_name: str = ""
    twitter_handle: str = ""

    # Volume metrics
    n_tweets: int = 0
    span_days: int = 0
    tweets_per_day: float = 0.0
    active_days_pct: float = 0.0

    # Temporal metrics
    median_gap_minutes: float = 0.0
    p_gap_lt_10m: float = 0.0
    p_gap_lt_60m: float = 0.0

    # Session metrics
    sessions_per_day: float = 0.0
    median_session_length: float = 0.0
    max_session_length: int = 0
    max_posts_in_hour: int = 0

    # Night activity
    night_share: float = 0.0

    # Post type distribution
    reply_ratio: float = 0.0
    quote_ratio: float = 0.0
    retweet_ratio: float = 0.0
    original_ratio: float = 0.0

    # Repetition
    exact_repeat_share: float = 0.0
    prefix_repeat_share: float = 0.0

    # Engagement distribution
    engagement_mean: float = 0.0
    engagement_median: float = 0.0
    engagement_p90: float = 0.0
    engagement_gini: float = 0.0

    # Scores
    compulsion_score: float = 0.0
    compulsion_subscores: dict = field(default_factory=dict)

    # Burst events
    burst_events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BehavioralAnalyzer:
    """Analyze temporal posting patterns for compulsion-like behavior."""

    def analyze(
        self,
        df: pd.DataFrame,
        senator_name: str = "",
        twitter_handle: str = "",
    ) -> BehavioralProfile:
        """
        Analyze a DataFrame of tweets from a single account.
        Expects columns: created_at (datetime), text, and optionally
        tweet_type, like_count, retweet_count, reply_count, quote_count.
        """
        profile = BehavioralProfile(
            senator_name=senator_name,
            twitter_handle=twitter_handle,
        )

        if df.empty:
            return profile

        # Ensure sorted by time
        df = df.sort_values("created_at").reset_index(drop=True)

        # ── Volume ────────────────────────────────────
        profile.n_tweets = len(df)
        ts = df["created_at"]
        span = (ts.max() - ts.min()).days
        profile.span_days = max(span, 1)
        profile.tweets_per_day = profile.n_tweets / profile.span_days
        unique_days = ts.dt.date.nunique()
        profile.active_days_pct = unique_days / profile.span_days

        # ── Inter-post gaps ───────────────────────────
        gaps = ts.diff().dt.total_seconds().dropna() / 60.0  # minutes
        if len(gaps) > 0:
            profile.median_gap_minutes = float(gaps.median())
            profile.p_gap_lt_10m = float((gaps < 10).mean())
            profile.p_gap_lt_60m = float((gaps < 60).mean())

        # ── Sessions ──────────────────────────────────
        sessions = self._detect_sessions(ts, gap_minutes=SESSION_GAP_MINUTES)
        if sessions:
            session_lengths = [s["length"] for s in sessions]
            profile.sessions_per_day = len(sessions) / profile.span_days
            profile.median_session_length = float(np.median(session_lengths))
            profile.max_session_length = max(session_lengths)

        # ── Hourly burst ──────────────────────────────
        hourly_counts = ts.dt.floor("h").value_counts()
        profile.max_posts_in_hour = int(hourly_counts.max()) if len(hourly_counts) > 0 else 0

        # ── Night activity ────────────────────────────
        hours = ts.dt.hour
        night_mask = (hours >= NIGHT_START_HOUR) & (hours < NIGHT_END_HOUR)
        profile.night_share = float(night_mask.mean())

        # ── Post type distribution ────────────────────
        if "tweet_type" in df.columns:
            type_counts = df["tweet_type"].str.lower().value_counts(normalize=True)
            profile.reply_ratio = float(type_counts.get("reply", type_counts.get("replied_to", 0)))
            profile.quote_ratio = float(type_counts.get("quote", type_counts.get("quoted", 0)))
            profile.retweet_ratio = float(type_counts.get("retweet", type_counts.get("retweeted", 0)))
            profile.original_ratio = float(type_counts.get("tweet", type_counts.get("original", 0)))
        elif "in_reply_to_user_id" in df.columns:
            has_reply = df["in_reply_to_user_id"].notna()
            profile.reply_ratio = float(has_reply.mean())
            profile.original_ratio = 1.0 - profile.reply_ratio

        # ── Repetition ────────────────────────────────
        if "text" in df.columns:
            import re
            texts = df["text"].str.strip().str.lower()
            # Strip URLs — t.co links make otherwise-identical tweets unique
            stripped = texts.apply(lambda t: re.sub(r'https?://\S+', '', t).strip())
            profile.exact_repeat_share = float(1 - stripped.nunique() / max(len(stripped), 1))

            # Structural repetition: first 3 words (captures rhetorical patterns
            # like "Do you think...", "Should we...", "Raise your hand if...")
            first_words = stripped.str.split().str[:3].str.join(" ")
            first_words = first_words[first_words.str.len() > 0]
            n_fw = len(first_words)
            if n_fw > 0:
                # Share of tweets whose opening 3-word phrase appears 3+ times
                fw_counts = first_words.value_counts()
                repeated_openings = fw_counts[fw_counts >= 3].sum()
                profile.prefix_repeat_share = float(repeated_openings / n_fw)
            else:
                profile.prefix_repeat_share = 0.0

        # ── Engagement distribution ───────────────────
        engagement = self._compute_engagement(df)
        if engagement is not None and len(engagement) > 0:
            profile.engagement_mean = float(engagement.mean())
            profile.engagement_median = float(engagement.median())
            profile.engagement_p90 = float(np.percentile(engagement, 90))
            profile.engagement_gini = float(self._gini(engagement.values))

        # ── Burst detection ───────────────────────────
        profile.burst_events = self._detect_bursts(
            ts, window_minutes=BURST_WINDOW_MINUTES
        )

        # ── Compulsion scoring ────────────────────────
        profile.compulsion_subscores = self._compute_subscores(profile)
        profile.compulsion_score = self._weighted_score(
            profile.compulsion_subscores, COMPULSION_WEIGHTS
        )

        return profile

    def _detect_sessions(
        self, timestamps: pd.Series, gap_minutes: float = 30
    ) -> list[dict]:
        """Detect posting sessions (new session when gap >= threshold)."""
        if len(timestamps) < 2:
            return []

        gaps = timestamps.diff().dt.total_seconds() / 60.0
        session_breaks = gaps > gap_minutes
        session_ids = session_breaks.cumsum()

        sessions = []
        for sid, group in timestamps.groupby(session_ids):
            sessions.append({
                "start": group.iloc[0],
                "end": group.iloc[-1],
                "length": len(group),
                "duration_minutes": (group.iloc[-1] - group.iloc[0]).total_seconds() / 60.0,
            })
        return sessions

    def _detect_bursts(
        self, timestamps: pd.Series, window_minutes: float = 60
    ) -> list[dict]:
        """Detect unusual posting bursts using z-score on rolling windows."""
        if len(timestamps) < 10:
            return []

        # Count posts per window
        counts = timestamps.dt.floor(f"{int(window_minutes)}min").value_counts().sort_index()
        if len(counts) < 3:
            return []

        mean_count = counts.mean()
        std_count = counts.std()
        if std_count == 0:
            return []

        z_scores = (counts - mean_count) / std_count
        burst_mask = z_scores > 2.0  # >2 standard deviations

        bursts = []
        for ts_window, z in z_scores[burst_mask].items():
            bursts.append({
                "window_start": str(ts_window),
                "count": int(counts[ts_window]),
                "z_score": round(float(z), 2),
            })

        return sorted(bursts, key=lambda x: x["z_score"], reverse=True)[:20]

    def _compute_engagement(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Compute total engagement per tweet."""
        eng_cols = ["like_count", "retweet_count", "reply_count", "quote_count"]
        available = [c for c in eng_cols if c in df.columns]
        if not available:
            return None
        return df[available].sum(axis=1)

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Compute Gini coefficient for engagement inequality."""
        values = np.sort(values)
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * values).sum() / (n * values.sum()))

    def _compute_subscores(self, profile: BehavioralProfile) -> dict:
        """
        Compute normalized 0-100 subscores for each compulsion dimension.
        Calibrated so that Mike Lee's known profile (~37 tweets/day, 2.5min
        median gap, 40% night share) scores ~99+ on most dimensions.
        """
        scores = {}

        # Activity: tweets per day — sigmoid with 50% at 5/day (most senators do <5)
        scores["activity"] = self._sigmoid_score(
            profile.tweets_per_day, midpoint=5, steepness=0.3
        )

        # Burstiness: fraction of gaps < 10 minutes — 50% threshold = extreme
        scores["burstiness"] = min(100, profile.p_gap_lt_10m * 100 / 0.5) if profile.p_gap_lt_10m else 0

        # Night activity: share of posts 00:00-06:00
        # Expected uniform would be 25%; >30% is elevated; >35% is extreme
        scores["night_activity"] = min(100, profile.night_share * 100 / 0.25)

        # Session intensity: combine sessions/day and max session length
        # High sessions/day AND long max sessions = compulsive pattern
        session_score = self._sigmoid_score(
            profile.sessions_per_day, midpoint=3, steepness=0.5
        )
        max_session_score = self._sigmoid_score(
            profile.max_session_length, midpoint=20, steepness=0.1
        )
        scores["session_intensity"] = (session_score * 0.5 + max_session_score * 0.5)

        # Reply reactivity: reply ratio (high = reactive posting)
        # Most broadcast accounts are <10% reply; >30% is reactive
        scores["reply_reactivity"] = min(100, profile.reply_ratio * 100 / 0.3)

        # Repetition: structural pattern reuse
        # prefix_repeat_share = share of tweets with a 3+ times reused opening
        # For political accounts, >15% structural repetition is very high
        # Combine with exact repeats (URL-stripped)
        rep_combined = (
            profile.exact_repeat_share * 0.3
            + profile.prefix_repeat_share * 0.7
        )
        scores["repetition"] = min(100, rep_combined * 100 / 0.15)

        # Emoji/media sparsity: low emoji/media usage = text-heavy engagement-seeking
        # (Proxy: if emoji_share < 10% and we don't have media data, score high)
        scores["emoji_media_sparsity"] = 100  # default; will refine when media data available

        return {k: round(min(100, max(0, v)), 1) for k, v in scores.items()}

    @staticmethod
    def _sigmoid_score(value: float, midpoint: float = 10, steepness: float = 0.2) -> float:
        """Map a value to 0-100 using a sigmoid curve."""
        return 100 / (1 + np.exp(-steepness * (value - midpoint)))

    @staticmethod
    def _weighted_score(subscores: dict, weights: dict) -> float:
        """Compute weighted average score."""
        total = 0.0
        weight_sum = 0.0
        for key, weight in weights.items():
            if key in subscores:
                total += subscores[key] * weight
                weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return round(total / weight_sum, 1)
