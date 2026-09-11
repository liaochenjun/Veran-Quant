"""PIT-safe market regime labels (P2-1.5).

Regimes are computed ONLY from T-visible data: the assembled feature dict
(point-in-time by construction) plus closed bars stored in the sample's
market_state (all close_time < T). No future prices, no outcomes, no
test labels. Two minimal families:

- trend: trend_up / trend_down / sideways (from geometry trend one-hots)
- volatility: high_volatility / low_volatility (realized vol of 1h closes,
  split at the EARLY-period median — a rule fixed in advance)

Families that cannot be computed for a sample yield None (UNAVAILABLE).
"""

from __future__ import annotations

import math
from typing import Optional

from src.dataset.behavior_dataset import BehaviorSample

TREND_VALUES = ("trend_up", "trend_down", "sideways")
VOL_VALUES = ("high_volatility", "low_volatility")


def trend_regime(features: dict) -> str:
    """Dominant trend from multi-timeframe geometry one-hots (4h first)."""
    for timeframe in ("4h", "1h", "15m"):
        up = features.get(f"market__{timeframe}__geometry__local_trend_structure__uptrend", 0.0)
        down = features.get(f"market__{timeframe}__geometry__local_trend_structure__downtrend", 0.0)
        if up == 1.0:
            return "trend_up"
        if down == 1.0:
            return "trend_down"
    return "sideways"


def realized_volatility(sample: BehaviorSample) -> Optional[float]:
    """Std of log returns of 1h closes visible at T (all bars close < T)."""
    records = sample.market_state.get("1h", [])
    closes = [float(r["close"]) for r in records if r.get("close") is not None]
    if len(closes) < 5:
        return None
    returns = [
        math.log(c2 / c1) for c1, c2 in zip(closes[:-1], closes[1:]) if c1 > 0 and c2 > 0
    ]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


def early_median_volatility(samples: list[BehaviorSample]) -> Optional[float]:
    """Median realized volatility over the EARLY half of the samples.

    The split rule is fixed in advance and uses no future/test information
    beyond what the sample itself already saw at its own T.
    """
    ordered = sorted(samples, key=lambda s: s.timestamp)
    early = ordered[: max(1, len(ordered) // 2)]
    vols = [v for v in (realized_volatility(s) for s in early) if v is not None]
    if not vols:
        return None
    vols.sort()
    return vols[len(vols) // 2]


def volatility_regime(sample: BehaviorSample, median_vol: Optional[float]) -> Optional[str]:
    vol = realized_volatility(sample)
    if vol is None or median_vol is None:
        return None
    return "high_volatility" if vol >= median_vol else "low_volatility"


def label_regimes(
    samples: list[BehaviorSample],
    features_cache: dict[int, dict],
) -> dict[int, dict]:
    """Per-sample regime labels (trend + volatility), both PIT-safe."""
    median_vol = early_median_volatility(samples)
    labels = {}
    for sample in samples:
        features = features_cache[id(sample)]
        labels[id(sample)] = {
            "trend": trend_regime(features),
            "volatility": volatility_regime(sample, median_vol),
        }
    return labels
