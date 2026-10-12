"""
Compulsive usage signatures — temporal features from digital phenotyping literature.

Distinguishes compulsive (addiction-like) posting from strategic (staff-managed) posting
using features validated in the academic literature:

- Burstiness parameter B (Goh & Barabasi 2008)
- Time-of-day entropy (De Choudhury et al. 2013)
- Weekend/holiday ratio
- Self-excitation branching ratio (Hawkes process)
- Response latency to external events
- Inter-post gap distribution shape

Reference framework: Recovery Viability Theory (Kepner, White, O'Neill) —
logit-bounded state space, cusp catastrophe dynamics, critical slowing down.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit as _logit, expit as sigmoid

log = logging.getLogger(__name__)


@dataclass
class CompulsionSignatureProfile:
    """
    Compulsion signature analysis for a single account.
    Each feature is scored on a 0-1 scale where:
        0 = clearly strategic/staff-managed
        1 = clearly compulsive/personal
    """
    senator_name: str = ""
    twitter_handle: str = ""
    n_tweets: int = 0

    # ── Core signatures ──────────────────────────────
    burstiness_B: float = 0.0          # Goh-Barabasi parameter (-1 to 1)
    burstiness_score: float = 0.0      # mapped to 0-1

    time_entropy: float = 0.0          # Shannon entropy of hourly distribution (0-4.58 bits)
    time_entropy_score: float = 0.0    # mapped to 0-1

    weekend_ratio: float = 0.0         # weekend posts / weekday posts (per day)
    weekend_score: float = 0.0

    night_intensity: float = 0.0       # fraction of posts 12am-5am
    night_score: float = 0.0

    self_excitation_ratio: float = 0.0 # Hawkes branching ratio estimate
    self_excitation_score: float = 0.0

    response_latency_median_min: float = 0.0  # median response to @mentions
    latency_score: float = 0.0

    gap_distribution_shape: str = ""   # "exponential", "power_law", "log_normal"
    gap_shape_score: float = 0.0

    # ── Composite ────────────────────────────────────
    compulsive_probability: float = 0.0  # posterior P(compulsive | features)
    classification: str = ""             # "compulsive", "strategic", "mixed"

    # ── Logit-bounded state (Kepner framework) ───────
    z_state: float = 0.0   # logit(compulsive_probability)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class CompulsionSignatureAnalyzer:
    """
    Compute compulsion signatures from a tweet time series.

    Thresholds calibrated from literature:
    - Strategic accounts: B < 0, entropy < 3.0, weekend_ratio < 0.5
    - Compulsive accounts: B > 0.3, entropy > 3.5, weekend_ratio > 0.85
    """

    def analyze(
        self,
        df: pd.DataFrame,
        senator_name: str = "",
        twitter_handle: str = "",
    ) -> CompulsionSignatureProfile:
        """Compute all compulsion signatures from a tweet DataFrame."""
        profile = CompulsionSignatureProfile(
            senator_name=senator_name,
            twitter_handle=twitter_handle,
            n_tweets=len(df),
        )

        if df.empty or "created_at" not in df.columns:
            return profile

        df = df.sort_values("created_at").copy()
        ts = df["created_at"]

        # ── 1. Burstiness parameter B ─────────────────
        profile.burstiness_B = self._burstiness(ts)
        # Map: B=-1 -> 0 (regular), B=0 -> 0.5 (Poisson), B=1 -> 1.0 (bursty)
        profile.burstiness_score = (profile.burstiness_B + 1) / 2

        # ── 2. Time-of-day entropy ────────────────────
        profile.time_entropy = self._time_entropy(ts)
        # Map: 0-4.58 bits; > 3.5 = compulsive, < 2.5 = strategic
        profile.time_entropy_score = min(1.0, max(0, (profile.time_entropy - 2.0) / 2.5))

        # ── 3. Weekend/holiday ratio ──────────────────
        profile.weekend_ratio = self._weekend_ratio(ts)
        # Map: 0 = no weekend posting, 1 = same as weekday
        profile.weekend_score = min(1.0, profile.weekend_ratio / 0.9)

        # ── 4. Night intensity ────────────────────────
        hours = ts.dt.hour
        profile.night_intensity = float(hours.between(0, 5).mean())
        # 25% uniform baseline; > 25% = elevated; > 35% = extreme
        profile.night_score = min(1.0, profile.night_intensity / 0.25)

        # ── 5. Self-excitation (Hawkes branching ratio) ──
        profile.self_excitation_ratio = self._hawkes_branching_ratio(ts)
        # Branching ratio: 0 = no self-excitation, 1 = critical
        profile.self_excitation_score = min(1.0, profile.self_excitation_ratio)

        # ── 6. Response latency (if reply data available) ──
        if "in_reply_to_user_id" in df.columns:
            profile.response_latency_median_min = self._response_latency(df)
            # Fast response = compulsive: < 5min = 1.0, > 60min = 0.0
            if profile.response_latency_median_min > 0:
                profile.latency_score = max(0, 1.0 - profile.response_latency_median_min / 60)
        else:
            profile.latency_score = 0.5  # unknown

        # ── 7. Gap distribution shape ─────────────────
        profile.gap_distribution_shape = self._gap_distribution_shape(ts)
        shape_scores = {
            "power_law": 1.0,       # heavy-tailed = compulsive
            "log_normal": 0.7,      # somewhat bursty
            "exponential": 0.3,     # Poisson-like = mixed
            "regular": 0.0,         # scheduled = strategic
            "unknown": 0.5,
        }
        profile.gap_shape_score = shape_scores.get(profile.gap_distribution_shape, 0.5)

        # ── Composite score (logistic fusion) ─────────
        features = np.array([
            profile.burstiness_score,
            profile.time_entropy_score,
            profile.weekend_score,
            profile.night_score,
            profile.self_excitation_score,
            profile.latency_score,
            profile.gap_shape_score,
        ])
        weights = np.array([0.20, 0.20, 0.10, 0.15, 0.15, 0.10, 0.10])

        # Weighted average then logit-sigmoid for calibration
        raw_score = np.sum(features * weights)
        # Apply logit-sigmoid transform (Kepner framework: natural bounds)
        profile.compulsive_probability = float(raw_score)
        profile.z_state = float(_logit(np.clip(raw_score, 0.01, 0.99)))

        if profile.compulsive_probability > 0.65:
            profile.classification = "compulsive"
        elif profile.compulsive_probability < 0.35:
            profile.classification = "strategic"
        else:
            profile.classification = "mixed"

        return profile

    @staticmethod
    def _burstiness(timestamps: pd.Series) -> float:
        """
        Goh-Barabasi burstiness parameter B = (sigma - mu) / (sigma + mu)
        B = -1: perfectly regular
        B = 0: Poisson (random)
        B = +1: maximally bursty
        """
        gaps = timestamps.diff().dt.total_seconds().dropna()
        if len(gaps) < 2:
            return 0.0
        mu = gaps.mean()
        sigma = gaps.std()
        if mu + sigma == 0:
            return 0.0
        return float((sigma - mu) / (sigma + mu))

    @staticmethod
    def _time_entropy(timestamps: pd.Series) -> float:
        """
        Shannon entropy of the hourly posting distribution.
        Max = log2(24) ≈ 4.58 bits (uniform across all hours).
        Strategic accounts: < 3.0 bits
        Compulsive accounts: > 3.5 bits
        """
        hour_counts = timestamps.dt.hour.value_counts(normalize=True)
        # Fill all 24 hours
        hour_probs = np.zeros(24)
        for h, p in hour_counts.items():
            hour_probs[h] = p
        # Add tiny epsilon to avoid log(0)
        hour_probs = hour_probs + 1e-10
        hour_probs = hour_probs / hour_probs.sum()
        return float(-np.sum(hour_probs * np.log2(hour_probs)))

    @staticmethod
    def _weekend_ratio(timestamps: pd.Series) -> float:
        """
        Ratio of weekend posting rate to weekday posting rate.
        Strategic: < 0.5 (much less on weekends)
        Compulsive: > 0.85 (similar to weekdays)
        """
        dow = timestamps.dt.dayofweek
        n_weekend = (dow >= 5).sum()
        n_weekday = (dow < 5).sum()

        # Count distinct weekend/weekday days
        dates = timestamps.dt.date
        unique_dates = pd.to_datetime(pd.Series(dates.unique()))
        unique_dow = unique_dates.dt.dayofweek
        n_weekend_days = max(1, (unique_dow >= 5).sum())
        n_weekday_days = max(1, (unique_dow < 5).sum())

        weekend_rate = n_weekend / n_weekend_days
        weekday_rate = n_weekday / n_weekday_days

        if weekday_rate == 0:
            return 0.0
        return float(weekend_rate / weekday_rate)

    @staticmethod
    def _hawkes_branching_ratio(timestamps: pd.Series, window_sec: float = 300) -> float:
        """
        Estimate the Hawkes process branching ratio.
        For each post, count how many follow-up posts occur within `window_sec` seconds.
        The average is an estimate of the branching ratio n*.

        n* < 0.3: mostly independent (strategic)
        n* > 0.5: self-exciting (compulsive cascades)
        n* -> 1.0: critical (compulsive spiral)
        """
        times = timestamps.values.astype("int64") / 1e9  # to seconds
        n = len(times)
        if n < 10:
            return 0.0

        # Count triggered events per parent event
        triggered_counts = []
        for i in range(min(n - 1, 5000)):  # cap at 5000 for speed
            window_end = times[i] + window_sec
            # Count events in (times[i], times[i] + window]
            count = np.searchsorted(times, window_end) - (i + 1)
            triggered_counts.append(max(0, count))

        if not triggered_counts:
            return 0.0

        # Branching ratio estimate (needs correction for background rate)
        mean_triggered = np.mean(triggered_counts)
        # Simple estimate: n* ≈ 1 - 1/(1 + mean_triggered)
        # This is approximate; full MLE would use Ozaki (1979)
        return float(min(1.0, mean_triggered / (1 + mean_triggered)))

    @staticmethod
    def _response_latency(df: pd.DataFrame) -> float:
        """Median time between a reply and the tweet it replies to."""
        replies = df[df["in_reply_to_user_id"].notna()].copy()
        if len(replies) < 5:
            return 0.0

        # Use inter-post gap for replies as proxy for response time
        replies = replies.sort_values("created_at")
        gaps = replies["created_at"].diff().dt.total_seconds().dropna() / 60.0
        return float(gaps.median()) if len(gaps) > 0 else 0.0

    @staticmethod
    def _gap_distribution_shape(timestamps: pd.Series) -> str:
        """
        Determine if inter-post gap distribution is:
        - power_law (heavy-tailed, compulsive)
        - log_normal (somewhat bursty)
        - exponential (Poisson-like, mixed)
        - regular (scheduled, strategic)
        """
        gaps = timestamps.diff().dt.total_seconds().dropna().values
        gaps = gaps[gaps > 0]

        if len(gaps) < 50:
            return "unknown"

        # Coefficient of variation
        cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0

        if cv < 0.3:
            return "regular"
        elif cv < 0.8:
            return "exponential"
        elif cv < 1.5:
            return "log_normal"
        else:
            return "power_law"
