"""
Disorder baseline comparison — comparing temporal posting signatures
against known psychological disorder profiles.

Maps compulsion signatures to clinical disorder constructs from the
Recovery Viability Theory (Kepner, White, O'Neill):
- Behavioral addiction criteria (DSM-5-adjacent)
- Compulsive behavior markers
- Impulse control disorder signatures
- Cusp catastrophe dynamics (sudden behavioral transitions)
- Critical slowing down (early warning of behavioral crisis)

Ground truth baselines:
- Strategic/staff-managed accounts (healthy baseline)
- Known compulsive posters (Trump Android, high-compulsion senators)
- Literature thresholds for digital phenotyping (De Choudhury et al.)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# ── Clinical disorder criteria mapped to temporal features ────────
# Based on DSM-5 behavioral addiction criteria + digital phenotyping literature

DISORDER_CRITERIA = {
    "tolerance": {
        "description": "Needing increasing amounts of the behavior to achieve the same effect",
        "temporal_proxy": "posting_rate_trend",
        "measurement": "Positive slope of weekly posting rate over 6+ months",
        "threshold": 0.05,  # >5% weekly increase = escalation
    },
    "withdrawal": {
        "description": "Distress or compensatory behavior when unable to post",
        "temporal_proxy": "gap_rebound_intensity",
        "measurement": "Ratio of post-gap burst intensity to baseline rate",
        "threshold": 3.0,  # 3x baseline after a gap = withdrawal rebound
    },
    "loss_of_control": {
        "description": "Unsuccessful efforts to cut down or stop",
        "temporal_proxy": "failed_reduction_events",
        "measurement": "Count of periods where rate drops then rebounds higher",
        "threshold": 2,  # 2+ failed reduction attempts
    },
    "preoccupation": {
        "description": "Excessive time spent on the behavior",
        "temporal_proxy": "active_hours_per_day",
        "measurement": "Mean distinct hours with posts per day",
        "threshold": 12,  # >12 distinct posting hours = preoccupied
    },
    "escape": {
        "description": "Using the behavior to escape negative mood states",
        "temporal_proxy": "crisis_posting_surge",
        "measurement": "Ratio of posting rate during high-negativity periods to baseline",
        "threshold": 1.5,  # 50% increase during negative events
    },
    "continued_despite_harm": {
        "description": "Continuing despite knowledge of persistent problems caused by the behavior",
        "temporal_proxy": "night_posting_persistence",
        "measurement": "Night posting share sustained >20% for 3+ months",
        "threshold": 0.20,
    },
    "deception": {
        "description": "Lying about extent of involvement",
        "temporal_proxy": "deletion_rate",
        "measurement": "Estimated deletion rate (if detectable)",
        "threshold": 0.05,  # >5% deleted = concealment behavior
    },
    "relationship_jeopardy": {
        "description": "Risking significant relationships or opportunities",
        "temporal_proxy": "controversy_escalation_trend",
        "measurement": "Positive slope of toxicity/controversy scores over time",
        "threshold": 0.03,
    },
}


@dataclass
class DisorderProfile:
    """Clinical-adjacent disorder assessment for a social media account."""
    senator_name: str = ""
    twitter_handle: str = ""

    # Criteria met (out of 8)
    criteria_met: int = 0
    criteria_details: dict = field(default_factory=dict)

    # Severity classification
    # DSM-5 gambling disorder uses: 4-5 = mild, 6-7 = moderate, 8-9 = severe
    # Adapted for 8 criteria: 3 = mild, 4-5 = moderate, 6+ = severe
    severity: str = ""  # "none", "subclinical", "mild", "moderate", "severe"

    # Baseline comparisons
    vs_strategic_baseline: dict = field(default_factory=dict)
    vs_population_percentile: dict = field(default_factory=dict)

    # Temporal trajectory
    trajectory: str = ""  # "stable", "escalating", "de-escalating", "cycling"
    trajectory_details: dict = field(default_factory=dict)

    # Kepner framework
    viability_zone_estimate: str = ""  # "within", "boundary", "outside"
    early_warning_signals: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Strategic baseline (from our senator analysis) ────────────────
# These are empirical values from the "most strategic" senators in the HF dataset
STRATEGIC_BASELINE = {
    "burstiness_B": -0.02,      # Near-Poisson (scheduled posting)
    "time_entropy": 3.3,        # Concentrated in business hours
    "weekend_ratio": 0.4,       # Much less on weekends
    "night_intensity": 0.03,    # ~3% night posting (minimal)
    "self_excitation": 0.75,    # Low self-triggering
    "tweets_per_day": 5.0,      # Moderate volume
    "median_gap_min": 45.0,     # ~45 min between posts
    "sessions_per_day": 3.0,    # 3 posting sessions
}

# Population distribution parameters (from our 98-senator analysis)
POPULATION_STATS = {
    "burstiness_B": {"mean": 0.18, "std": 0.09},
    "time_entropy": {"mean": 3.75, "std": 0.30},
    "weekend_ratio": {"mean": 0.65, "std": 0.15},
    "night_intensity": {"mean": 0.10, "std": 0.08},
    "self_excitation": {"mean": 0.89, "std": 0.06},
}


class DisorderBaselineAnalyzer:
    """
    Compare an account's temporal signatures against clinical disorder
    criteria and population baselines.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        signatures: dict,
        behavioral: dict,
        senator_name: str = "",
        twitter_handle: str = "",
    ) -> DisorderProfile:
        """
        Assess disorder criteria using temporal posting data.

        Args:
            df: Tweet DataFrame with created_at column
            signatures: CompulsionSignatureProfile.to_dict()
            behavioral: BehavioralProfile.to_dict()
        """
        profile = DisorderProfile(
            senator_name=senator_name,
            twitter_handle=twitter_handle,
        )

        if df.empty or "created_at" not in df.columns:
            return profile

        df = df.sort_values("created_at").copy()

        # ── Assess each criterion ─────────────────────
        criteria_results = {}

        # 1. Tolerance: escalating posting rate
        rate_trend = self._assess_tolerance(df)
        criteria_results["tolerance"] = {
            "met": rate_trend["slope"] > DISORDER_CRITERIA["tolerance"]["threshold"],
            "value": round(rate_trend["slope"], 4),
            "detail": rate_trend,
        }

        # 2. Withdrawal: rebound after gaps
        rebound = self._assess_withdrawal(df)
        criteria_results["withdrawal"] = {
            "met": rebound["rebound_ratio"] > DISORDER_CRITERIA["withdrawal"]["threshold"],
            "value": round(rebound["rebound_ratio"], 2),
            "detail": rebound,
        }

        # 3. Loss of control: failed reduction attempts
        failed = self._assess_loss_of_control(df)
        criteria_results["loss_of_control"] = {
            "met": failed["n_failed_reductions"] >= DISORDER_CRITERIA["loss_of_control"]["threshold"],
            "value": failed["n_failed_reductions"],
            "detail": failed,
        }

        # 4. Preoccupation: excessive hours per day
        hours = self._assess_preoccupation(df)
        criteria_results["preoccupation"] = {
            "met": hours["mean_active_hours"] > DISORDER_CRITERIA["preoccupation"]["threshold"],
            "value": round(hours["mean_active_hours"], 1),
            "detail": hours,
        }

        # 5. Escape: posting surges during negative events
        escape = self._assess_escape(df)
        criteria_results["escape"] = {
            "met": escape["crisis_ratio"] > DISORDER_CRITERIA["escape"]["threshold"],
            "value": round(escape["crisis_ratio"], 2),
            "detail": escape,
        }

        # 6. Continued despite harm: persistent night posting
        night_persist = self._assess_continued_despite_harm(df)
        criteria_results["continued_despite_harm"] = {
            "met": night_persist["persistent_night"],
            "value": round(night_persist["night_share_trend"], 4),
            "detail": night_persist,
        }

        # 7. Deception: (proxy — not directly measurable from archive)
        criteria_results["deception"] = {
            "met": False,
            "value": 0,
            "detail": {"note": "Cannot assess from public archive alone"},
        }

        # 8. Relationship jeopardy: escalating controversy
        controversy = self._assess_relationship_jeopardy(df)
        criteria_results["relationship_jeopardy"] = {
            "met": controversy["escalating"],
            "value": round(controversy["controversy_trend"], 4),
            "detail": controversy,
        }

        profile.criteria_details = criteria_results
        profile.criteria_met = sum(1 for c in criteria_results.values() if c["met"])

        # Severity
        if profile.criteria_met >= 6:
            profile.severity = "severe"
        elif profile.criteria_met >= 4:
            profile.severity = "moderate"
        elif profile.criteria_met >= 3:
            profile.severity = "mild"
        elif profile.criteria_met >= 1:
            profile.severity = "subclinical"
        else:
            profile.severity = "none"

        # ── Baseline comparisons ──────────────────────
        profile.vs_strategic_baseline = self._compare_to_baseline(signatures)
        profile.vs_population_percentile = self._compute_percentiles(signatures)

        # ── Trajectory ────────────────────────────────
        profile.trajectory, profile.trajectory_details = self._assess_trajectory(df)

        # ── Viability zone ────────────────────────────
        profile.viability_zone_estimate = self._estimate_viability_zone(
            profile.criteria_met, signatures
        )

        return profile

    def _assess_tolerance(self, df: pd.DataFrame) -> dict:
        """Check if posting rate is escalating over time."""
        weekly = df.set_index("created_at").resample("W").size()
        if len(weekly) < 8:
            return {"slope": 0, "p_value": 1, "weeks": len(weekly)}

        x = np.arange(len(weekly))
        slope, intercept, r, p, se = stats.linregress(x, weekly.values)
        # Normalize slope by mean rate
        mean_rate = weekly.mean()
        norm_slope = slope / mean_rate if mean_rate > 0 else 0

        return {
            "slope": float(norm_slope),
            "raw_slope": float(slope),
            "p_value": float(p),
            "r_squared": float(r**2),
            "weeks": len(weekly),
            "start_rate": float(weekly.iloc[:4].mean()),
            "end_rate": float(weekly.iloc[-4:].mean()),
        }

    def _assess_withdrawal(self, df: pd.DataFrame) -> dict:
        """Check for rebound surges after posting gaps."""
        daily = df.set_index("created_at").resample("D").size()
        gaps = daily[daily == 0]

        if len(gaps) < 3:
            return {"rebound_ratio": 1.0, "n_gaps": 0}

        baseline = daily[daily > 0].mean()
        rebounds = []

        for gap_date in gaps.index:
            # Look at the 3 days after the gap
            post_gap = daily[gap_date:gap_date + pd.Timedelta(days=3)]
            post_gap_rate = post_gap[post_gap > 0].mean() if len(post_gap[post_gap > 0]) > 0 else 0
            if baseline > 0:
                rebounds.append(post_gap_rate / baseline)

        mean_rebound = np.mean(rebounds) if rebounds else 1.0

        return {
            "rebound_ratio": float(mean_rebound),
            "n_gaps": len(gaps),
            "baseline_daily_rate": float(baseline),
            "n_rebounds_above_2x": sum(1 for r in rebounds if r > 2),
        }

    def _assess_loss_of_control(self, df: pd.DataFrame) -> dict:
        """Count periods where posting rate drops then rebounds higher."""
        weekly = df.set_index("created_at").resample("W").size()
        if len(weekly) < 12:
            return {"n_failed_reductions": 0}

        # Smooth with 4-week rolling average
        smoothed = weekly.rolling(4, min_periods=2).mean()
        failed = 0
        in_reduction = False
        reduction_low = None

        for i in range(1, len(smoothed)):
            if smoothed.iloc[i] < smoothed.iloc[i-1] * 0.7:  # 30% drop
                in_reduction = True
                reduction_low = smoothed.iloc[i]
            elif in_reduction and reduction_low is not None:
                if smoothed.iloc[i] > reduction_low * 1.5:  # Rebounds 50% above low
                    failed += 1
                    in_reduction = False
                    reduction_low = None

        return {"n_failed_reductions": failed}

    def _assess_preoccupation(self, df: pd.DataFrame) -> dict:
        """Count mean distinct posting hours per day."""
        daily_hours = df.groupby(df["created_at"].dt.date)["created_at"].apply(
            lambda x: x.dt.hour.nunique()
        )
        return {
            "mean_active_hours": float(daily_hours.mean()),
            "max_active_hours": int(daily_hours.max()),
            "pct_days_over_12h": float((daily_hours > 12).mean()),
        }

    def _assess_escape(self, df: pd.DataFrame) -> dict:
        """Check if posting surges during high-negativity periods."""
        if "sentiment_negative" not in df.columns:
            return {"crisis_ratio": 1.0, "note": "No sentiment data"}

        weekly = df.set_index("created_at").resample("W")
        weekly_rate = weekly.size()
        weekly_neg = weekly["sentiment_negative"].mean()

        if len(weekly_rate) < 4:
            return {"crisis_ratio": 1.0}

        # High-negativity weeks (top quartile)
        neg_threshold = weekly_neg.quantile(0.75)
        high_neg_weeks = weekly_neg >= neg_threshold
        crisis_rate = weekly_rate[high_neg_weeks].mean()
        normal_rate = weekly_rate[~high_neg_weeks].mean()

        ratio = crisis_rate / normal_rate if normal_rate > 0 else 1.0

        return {
            "crisis_ratio": float(ratio),
            "crisis_weeks": int(high_neg_weeks.sum()),
            "crisis_rate": float(crisis_rate),
            "normal_rate": float(normal_rate),
        }

    def _assess_continued_despite_harm(self, df: pd.DataFrame) -> dict:
        """Check for persistent night posting (sustained >20% for 3+ months)."""
        ts = df.set_index("created_at")
        is_night = (ts.index.hour < 6).astype(float)
        night_series = pd.Series(is_night, index=ts.index)
        monthly_night = night_series.resample("ME").mean().dropna()

        if len(monthly_night) < 3:
            return {"persistent_night": False, "night_share_trend": 0}

        consecutive_high = 0
        max_consecutive = 0
        for val in monthly_night:
            if val > 0.20:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0

        x = np.arange(len(monthly_night))
        slope = stats.linregress(x, monthly_night.values).slope if len(monthly_night) > 2 else 0

        return {
            "persistent_night": max_consecutive >= 3,
            "max_consecutive_months_over_20pct": max_consecutive,
            "night_share_trend": float(slope),
        }

    def _assess_relationship_jeopardy(self, df: pd.DataFrame) -> dict:
        """Check for escalating controversy over time."""
        if "offensive_score" not in df.columns:
            return {"escalating": False, "controversy_trend": 0}

        weekly_off = df.set_index("created_at")["offensive_score"].resample("W").mean().dropna()
        if len(weekly_off) < 8:
            return {"escalating": False, "controversy_trend": 0}

        x = np.arange(len(weekly_off))
        slope = stats.linregress(x, weekly_off.values).slope

        return {
            "escalating": slope > DISORDER_CRITERIA["relationship_jeopardy"]["threshold"],
            "controversy_trend": float(slope),
        }

    def _compare_to_baseline(self, signatures: dict) -> dict:
        """Compare account signatures to strategic baseline."""
        comparisons = {}
        sig_map = {
            "burstiness_B": "burstiness_B",
            "time_entropy": "time_entropy",
            "weekend_ratio": "weekend_ratio",
            "night_intensity": "night_intensity",
            "self_excitation": "self_excitation_ratio",
        }
        for baseline_key, sig_key in sig_map.items():
            baseline_val = STRATEGIC_BASELINE.get(baseline_key, 0)
            actual_val = signatures.get(sig_key, 0)
            if baseline_val != 0:
                ratio = actual_val / baseline_val
            else:
                ratio = float('inf') if actual_val > 0 else 1.0
            comparisons[baseline_key] = {
                "baseline": baseline_val,
                "actual": round(actual_val, 4) if isinstance(actual_val, float) else actual_val,
                "ratio": round(ratio, 2) if ratio != float('inf') else "inf",
                "deviation": "elevated" if ratio > 1.5 else "normal" if ratio < 1.5 else "baseline",
            }
        return comparisons

    def _compute_percentiles(self, signatures: dict) -> dict:
        """Compute where this account falls in the senator population."""
        percentiles = {}
        sig_map = {
            "burstiness_B": "burstiness_B",
            "time_entropy": "time_entropy",
            "weekend_ratio": "weekend_ratio",
            "night_intensity": "night_intensity",
            "self_excitation": "self_excitation_ratio",
        }
        for key, sig_key in sig_map.items():
            if key in POPULATION_STATS:
                pop = POPULATION_STATS[key]
                val = signatures.get(sig_key, 0)
                z = (val - pop["mean"]) / pop["std"] if pop["std"] > 0 else 0
                percentile = stats.norm.cdf(z) * 100
                percentiles[key] = {
                    "value": round(val, 4) if isinstance(val, float) else val,
                    "z_score": round(z, 2),
                    "percentile": round(percentile, 1),
                }
        return percentiles

    def _assess_trajectory(self, df: pd.DataFrame) -> tuple:
        """Determine if behavior is escalating, stable, cycling, or de-escalating."""
        weekly = df.set_index("created_at").resample("W").size()
        if len(weekly) < 12:
            return "unknown", {}

        # Split into thirds
        n = len(weekly)
        third = n // 3
        early = weekly.iloc[:third].mean()
        mid = weekly.iloc[third:2*third].mean()
        late = weekly.iloc[2*third:].mean()

        # Trend
        x = np.arange(len(weekly))
        slope, _, r, p, _ = stats.linregress(x, weekly.values)

        # Variance trend (critical slowing down)
        window = max(8, n // 5)
        variances = weekly.rolling(window).var().dropna()
        if len(variances) > 3:
            var_trend = stats.linregress(np.arange(len(variances)), variances.values).slope
        else:
            var_trend = 0

        details = {
            "early_rate": round(float(early), 1),
            "mid_rate": round(float(mid), 1),
            "late_rate": round(float(late), 1),
            "overall_slope": round(float(slope), 3),
            "r_squared": round(float(r**2), 3),
            "p_value": round(float(p), 4),
            "variance_trend": round(float(var_trend), 4),
        }

        if p > 0.05:
            trajectory = "stable"
        elif slope > 0 and late > early * 1.3:
            trajectory = "escalating"
        elif slope < 0 and late < early * 0.7:
            trajectory = "de-escalating"
        else:
            # Check for cycling (high variance, no trend)
            cv = weekly.std() / weekly.mean() if weekly.mean() > 0 else 0
            trajectory = "cycling" if cv > 0.5 else "stable"

        return trajectory, details

    def _estimate_viability_zone(self, criteria_met: int, signatures: dict) -> str:
        """
        Estimate position relative to the Recovery Viability Zone (Kepner framework).

        Within = sustainable behavior patterns
        Boundary = approaching critical transition
        Outside = unsustainable, compulsive regime
        """
        compulsive_prob = signatures.get("compulsive_probability", 0)

        if criteria_met >= 5 and compulsive_prob > 0.85:
            return "outside"
        elif criteria_met >= 3 or compulsive_prob > 0.70:
            return "boundary"
        else:
            return "within"
