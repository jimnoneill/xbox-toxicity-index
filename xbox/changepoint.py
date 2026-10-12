"""
Bayesian changepoint detection for compulsive usage patterns.

Inspired by the Recovery Viability Theory (Kepner, White, O'Neill):
- Cusp catastrophe model: sudden transitions in posting behavior
- Critical slowing down: variance/autocorrelation increase before transitions
- Logit-bounded state space: natural [0,1] bounds for behavioral metrics
- Recursive parameter updates: behavior shifts with each "engagement cycle"

Detects:
1. Changepoints in posting frequency (sudden acceleration/deceleration)
2. Changepoints in virulence (tone shifts)
3. Critical slowing down signatures (variance, autocorrelation approaching 1)
4. Cusp catastrophe dynamics (bistable regime detection)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.special import expit, logit  # sigmoid / logit transform

from .config import SESSION_GAP_MINUTES

log = logging.getLogger(__name__)


@dataclass
class Changepoint:
    """A detected changepoint in the time series."""
    date: str
    metric: str
    direction: str  # "increase" or "decrease"
    magnitude: float  # effect size
    confidence: float  # posterior probability
    before_mean: float
    after_mean: float


@dataclass
class EarlyWarningSignal:
    """Critical slowing down signature before a behavioral transition."""
    window_end: str
    metric: str
    variance_trend: float  # slope of variance over time
    autocorrelation_trend: float  # slope of lag-1 autocorrelation
    signal_strength: float  # combined indicator (0-1)


@dataclass
class ChangepointProfile:
    """Complete changepoint analysis for an account."""
    senator_name: str = ""
    changepoints: list = field(default_factory=list)
    early_warnings: list = field(default_factory=list)
    regime_periods: list = field(default_factory=list)
    cusp_events: list = field(default_factory=list)
    posting_rate_ts: Optional[pd.Series] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "posting_rate_ts"}
        d["n_changepoints"] = len(self.changepoints)
        d["n_early_warnings"] = len(self.early_warnings)
        return d


class BayesianChangepointDetector:
    """
    Bayesian online changepoint detection (Adams & MacKay 2007)
    adapted for social media posting patterns.

    The core idea: at each time step, compute the posterior probability
    of a changepoint having occurred. When this probability exceeds a
    threshold, we've found a regime change.
    """

    def __init__(self, hazard_rate: float = 1 / 250):
        """
        Args:
            hazard_rate: Prior probability of changepoint at any step.
                1/250 means we expect a change roughly every 250 time steps.
        """
        self.hazard_rate = hazard_rate

    def detect_changepoints(
        self,
        df: pd.DataFrame,
        resolution: str = "D",
        min_confidence: float = 0.5,
    ) -> ChangepointProfile:
        """
        Run full changepoint analysis on a tweet dataset.

        Args:
            df: DataFrame with created_at and optionally virulence scores
            resolution: Time aggregation ('D'=daily, 'W'=weekly)
            min_confidence: Minimum posterior probability to report a changepoint
        """
        profile = ChangepointProfile()

        if "created_at" not in df.columns or df.empty:
            return profile

        df = df.sort_values("created_at").copy()

        # ── Posting rate changepoints ─────────────────
        rate_ts = self._compute_posting_rate(df, resolution)
        if len(rate_ts) > 10:
            profile.posting_rate_ts = rate_ts
            rate_cps = self._bayesian_online_cpd(
                rate_ts.values, metric_name="posting_rate",
                dates=rate_ts.index, min_confidence=min_confidence,
            )
            profile.changepoints.extend(rate_cps)

        # ── Virulence changepoints ────────────────────
        if "composite_virulence" in df.columns:
            vir_ts = df.set_index("created_at")["composite_virulence"].resample(resolution).mean().dropna()
            if len(vir_ts) > 10:
                vir_cps = self._bayesian_online_cpd(
                    vir_ts.values, metric_name="virulence",
                    dates=vir_ts.index, min_confidence=min_confidence,
                )
                profile.changepoints.extend(vir_cps)

        # ── Night activity changepoints ───────────────
        night_ts = self._compute_night_share(df, resolution)
        if len(night_ts) > 10:
            night_cps = self._bayesian_online_cpd(
                night_ts.values, metric_name="night_share",
                dates=night_ts.index, min_confidence=min_confidence,
            )
            profile.changepoints.extend(night_cps)

        # ── Early warning signals (critical slowing down) ──
        if len(rate_ts) > 30:
            ews = self._detect_critical_slowing_down(rate_ts, metric="posting_rate")
            profile.early_warnings.extend(ews)

        # ── Regime identification ─────────────────────
        if profile.changepoints:
            profile.regime_periods = self._identify_regimes(
                rate_ts, profile.changepoints
            )

        # ── Cusp catastrophe events ───────────────────
        if len(rate_ts) > 30:
            profile.cusp_events = self._detect_cusp_events(rate_ts)

        return profile

    def _compute_posting_rate(self, df: pd.DataFrame, resolution: str) -> pd.Series:
        """Compute posting rate time series."""
        ts = df.set_index("created_at")
        counts = ts.resample(resolution).size()
        # Fill missing periods with 0
        full_range = pd.date_range(counts.index.min(), counts.index.max(), freq=resolution)
        counts = counts.reindex(full_range, fill_value=0)
        return counts

    def _compute_night_share(self, df: pd.DataFrame, resolution: str) -> pd.Series:
        """Compute night posting share (00:00-06:00 UTC) per period."""
        df = df.copy()
        df["is_night"] = df["created_at"].dt.hour.between(0, 5)
        ts = df.set_index("created_at")
        night_share = ts["is_night"].resample(resolution).mean().dropna()
        return night_share

    def _bayesian_online_cpd(
        self,
        data: np.ndarray,
        metric_name: str,
        dates: pd.DatetimeIndex,
        min_confidence: float = 0.5,
    ) -> list[Changepoint]:
        """
        Bayesian Online Changepoint Detection (Adams & MacKay 2007).

        Uses a conjugate Normal-Gamma prior for Gaussian observations.
        At each step, maintains a run-length distribution P(r_t | x_{1:t}).
        """
        n = len(data)
        if n < 5:
            return []

        # Hyperparameters for Normal-Gamma prior
        mu0 = np.mean(data)
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = np.var(data) if np.var(data) > 0 else 1.0

        # Run-length probabilities: R[t, r] = P(r_t = r | x_{1:t})
        # For memory efficiency, only track the current distribution
        max_run = n + 1
        R = np.zeros(max_run)
        R[0] = 1.0  # Initial run length is 0

        # Sufficient statistics for each run length
        mu = np.full(max_run, mu0)
        kappa = np.full(max_run, kappa0)
        alpha = np.full(max_run, alpha0)
        beta = np.full(max_run, beta0)

        changepoint_probs = np.zeros(n)

        for t in range(n):
            x = data[t]

            # Predictive probability under each run length (Student-t)
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_var = np.maximum(pred_var, 1e-10)
            pred_probs = stats.t.pdf(x, 2 * alpha, loc=mu, scale=np.sqrt(pred_var))

            # Growth: P(r_t = r_{t-1} + 1 | ...)
            R_new = np.zeros(max_run)
            R_new[1:t + 2] = R[:t + 1] * pred_probs[:t + 1] * (1 - self.hazard_rate)

            # Changepoint: P(r_t = 0 | ...)
            R_new[0] = np.sum(R[:t + 1] * pred_probs[:t + 1] * self.hazard_rate)

            # Normalize
            total = R_new.sum()
            if total > 0:
                R_new /= total
            else:
                R_new[0] = 1.0

            R = R_new
            changepoint_probs[t] = R[0]

            # Update sufficient statistics
            mu_new = (kappa * mu + x) / (kappa + 1)
            kappa_new = kappa + 1
            alpha_new = alpha + 0.5
            beta_new = beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1))

            # Shift for growth
            mu[1:] = mu_new[:-1]
            kappa[1:] = kappa_new[:-1]
            alpha[1:] = alpha_new[:-1]
            beta[1:] = beta_new[:-1]

            # Reset for changepoint
            mu[0] = mu0
            kappa[0] = kappa0
            alpha[0] = alpha0
            beta[0] = beta0

        # Extract changepoints above threshold
        changepoints = []
        # Find peaks in changepoint probability
        peaks, properties = signal.find_peaks(
            changepoint_probs, height=min_confidence, distance=7
        )

        for peak in peaks:
            if peak < 3 or peak >= n - 3:
                continue

            before = data[max(0, peak - 14):peak]
            after = data[peak:min(n, peak + 14)]

            if len(before) < 3 or len(after) < 3:
                continue

            before_mean = np.mean(before)
            after_mean = np.mean(after)
            magnitude = abs(after_mean - before_mean) / (np.std(data) + 1e-10)
            direction = "increase" if after_mean > before_mean else "decrease"

            changepoints.append(Changepoint(
                date=str(dates[peak].date()) if peak < len(dates) else "",
                metric=metric_name,
                direction=direction,
                magnitude=round(float(magnitude), 3),
                confidence=round(float(changepoint_probs[peak]), 3),
                before_mean=round(float(before_mean), 3),
                after_mean=round(float(after_mean), 3),
            ))

        return changepoints

    def _detect_critical_slowing_down(
        self,
        ts: pd.Series,
        metric: str,
        window_size: int = 30,
        step: int = 7,
    ) -> list[EarlyWarningSignal]:
        """
        Detect critical slowing down signatures:
        - Variance increases (Var ~ sigma^2 / (2|lambda|), diverges as lambda -> 0)
        - Lag-1 autocorrelation approaches 1 (rho = exp(lambda * dt) -> 1)

        From Kepner et al.: these are derived markers of approaching
        a cusp bifurcation boundary.
        """
        data = ts.values
        n = len(data)
        signals = []

        if n < window_size + step:
            return signals

        variances = []
        autocorrs = []
        window_ends = []

        for start in range(0, n - window_size, step):
            window = data[start:start + window_size]
            variances.append(np.var(window))

            # Lag-1 autocorrelation
            if np.std(window) > 0:
                ac = np.corrcoef(window[:-1], window[1:])[0, 1]
                autocorrs.append(ac if np.isfinite(ac) else 0)
            else:
                autocorrs.append(0)

            idx = min(start + window_size, len(ts.index) - 1)
            window_ends.append(ts.index[idx])

        if len(variances) < 3:
            return signals

        variances = np.array(variances)
        autocorrs = np.array(autocorrs)

        # Compute trends using Kendall's tau (nonparametric trend)
        var_trend = stats.kendalltau(range(len(variances)), variances)
        ac_trend = stats.kendalltau(range(len(autocorrs)), autocorrs)

        # Signal strength: combined increasing variance + increasing autocorrelation
        combined = (
            max(0, var_trend.statistic) * 0.5
            + max(0, ac_trend.statistic) * 0.5
        )

        if combined > 0.2:  # Meaningful signal
            signals.append(EarlyWarningSignal(
                window_end=str(window_ends[-1].date()) if window_ends else "",
                metric=metric,
                variance_trend=round(float(var_trend.statistic), 3),
                autocorrelation_trend=round(float(ac_trend.statistic), 3),
                signal_strength=round(float(combined), 3),
            ))

        return signals

    def _detect_cusp_events(
        self, ts: pd.Series, z_threshold: float = 3.0
    ) -> list[dict]:
        """
        Detect cusp catastrophe events: sudden, discontinuous jumps
        in posting rate (like a saddle-node bifurcation).

        From Kepner et al.: dz3 = (-z3^3 + b*z3 + a) dt + sigma dW
        When a crosses a_cusp, the system jumps discontinuously.
        """
        data = ts.values
        diffs = np.diff(data)

        if len(diffs) < 5 or np.std(diffs) == 0:
            return []

        z_scores = (diffs - np.mean(diffs)) / np.std(diffs)
        events = []

        for i, z in enumerate(z_scores):
            if abs(z) > z_threshold:
                events.append({
                    "date": str(ts.index[i + 1].date()) if i + 1 < len(ts.index) else "",
                    "direction": "surge" if z > 0 else "collapse",
                    "z_score": round(float(z), 2),
                    "before": round(float(data[i]), 1),
                    "after": round(float(data[i + 1]), 1),
                })

        return sorted(events, key=lambda x: abs(x["z_score"]), reverse=True)[:10]

    def _identify_regimes(
        self, ts: pd.Series, changepoints: list[Changepoint]
    ) -> list[dict]:
        """Identify behavioral regimes between changepoints."""
        if not changepoints or ts is None or ts.empty:
            return []

        # Sort changepoints by date
        cp_dates = sorted(set(
            cp.date for cp in changepoints if cp.metric == "posting_rate"
        ))

        if not cp_dates:
            return []

        regimes = []
        boundaries = [str(ts.index[0].date())] + cp_dates + [str(ts.index[-1].date())]

        for i in range(len(boundaries) - 1):
            start = pd.Timestamp(boundaries[i])
            end = pd.Timestamp(boundaries[i + 1])
            mask = (ts.index >= start) & (ts.index <= end)
            segment = ts[mask]

            if len(segment) < 3:
                continue

            regimes.append({
                "start": boundaries[i],
                "end": boundaries[i + 1],
                "mean_rate": round(float(segment.mean()), 1),
                "std_rate": round(float(segment.std()), 1),
                "days": len(segment),
            })

        return regimes
