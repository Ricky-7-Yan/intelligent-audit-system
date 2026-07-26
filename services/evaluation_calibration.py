"""Statistical calibration helpers shared by production evaluation paths."""

from __future__ import annotations

import math
from typing import Iterable


def beta_posterior_mean(
    successes: int,
    total: int,
    *,
    alpha: float = 3.0,
    beta: float = 1.0,
) -> float:
    """Return a smoothed rate so small samples never masquerade as certainty."""
    numerator = max(successes, 0) + alpha
    denominator = max(total, 0) + alpha + beta
    return round(min(0.999, max(0.001, numerator / denominator)), 4)


def calibrate_continuous(
    raw_score: float,
    *,
    evidence_units: int = 1,
    prior_mean: float = 0.72,
    prior_strength: float = 2.0,
) -> float:
    """Shrink a heuristic/model score toward a conservative prior.

    A single perfect observation is not enough to report 1.0. As evidence
    units grow, the calibrated score approaches the observed score.
    """
    bounded = min(1.0, max(0.0, float(raw_score)))
    units = max(1, int(evidence_units))
    posterior = (bounded * units + prior_mean * prior_strength) / (units + prior_strength)
    return round(min(0.999, max(0.001, posterior)), 4)


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + (z * z / total)
    centre = proportion + (z * z / (2 * total))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / total) + (z * z / (4 * total * total))
    )
    return round(max(0.0, (centre - margin) / denominator), 4)


def mean_score(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)
