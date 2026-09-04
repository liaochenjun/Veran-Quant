"""Encode ChanState snapshots into a deterministic flat numeric feature vector.

Conventions:
- mask/presence flags and counts are always numeric (0/1 or ints);
- value fields for structures that do not exist are NaN — missing structure
  is never silently treated as 0;
- key order is fixed so downstream datasets get identical columns.
"""

from __future__ import annotations

import math
from typing import ClassVar

from src.chan.chan_state import ChanState

_BSP_TYPES = ("1", "1p", "2", "2s", "3a", "3b")


def _bool_to_float(value: bool | None) -> float:
    if value is None:
        return math.nan
    return 1.0 if value else 0.0


def _direction_to_float(value: str | None) -> float:
    if value is None:
        return math.nan
    return 1.0 if value == "UP" else 0.0


class ChanFeatureEncoder:
    """Flatten one ChanState (or a multi-timeframe dict of them) into floats."""

    # Fixed, ordered feature keys produced for a single state.
    FEATURE_KEYS: ClassVar[tuple[str, ...]] = (
        "fractal_present",
        "fractal_top",
        "fractal_bottom",
        "fractal_price",
        "bi_count",
        "bi_direction_up",
        "bi_is_sure",
        "bi_amplitude",
        "bi_length",
        "segment_count",
        "segment_direction_up",
        "segment_is_sure",
        "zs_count",
        "zhongshu_present",
        "zhongshu_high",
        "zhongshu_low",
        "zhongshu_is_sure",
        "distance_to_zhongshu",
        "divergence_present",
        "divergence_strength",
        "buy_sell_point_present",
        "buy_sell_point_is_buy",
        "buy_sell_point_bi_is_sure",
        *(f"buy_sell_point_type_{t}" for t in _BSP_TYPES),
    )

    def encode(self, state: ChanState) -> dict[str, float]:
        bsp_types = set(state.buy_sell_point_types or ())
        return {
            "fractal_present": float(state.fractal_present),
            "fractal_top": float(state.fractal_type == "TOP"),
            "fractal_bottom": float(state.fractal_type == "BOTTOM"),
            "fractal_price": state.fractal_price if state.fractal_price is not None else math.nan,
            "bi_count": float(state.bi_count),
            "bi_direction_up": _direction_to_float(state.bi_direction),
            "bi_is_sure": _bool_to_float(state.bi_is_sure),
            "bi_amplitude": state.bi_amplitude if state.bi_amplitude is not None else math.nan,
            "bi_length": state.bi_length if state.bi_length is not None else math.nan,
            "segment_count": float(state.segment_count),
            "segment_direction_up": _direction_to_float(state.segment_direction),
            "segment_is_sure": _bool_to_float(state.segment_is_sure),
            "zs_count": float(state.zs_count),
            "zhongshu_present": float(state.zhongshu_present),
            "zhongshu_high": state.zhongshu_high if state.zhongshu_high is not None else math.nan,
            "zhongshu_low": state.zhongshu_low if state.zhongshu_low is not None else math.nan,
            "zhongshu_is_sure": _bool_to_float(state.zhongshu_is_sure),
            "distance_to_zhongshu": state.distance_to_zhongshu
            if state.distance_to_zhongshu is not None
            else math.nan,
            "divergence_present": float(state.divergence_present),
            "divergence_strength": state.divergence_strength
            if state.divergence_strength is not None
            else math.nan,
            "buy_sell_point_present": float(state.buy_sell_point_present),
            "buy_sell_point_is_buy": _bool_to_float(state.buy_sell_point_is_buy),
            "buy_sell_point_bi_is_sure": _bool_to_float(state.buy_sell_point_bi_is_sure),
            **{f"buy_sell_point_type_{t}": float(t in bsp_types) for t in _BSP_TYPES},
        }

    def encode_multi(self, states: dict[str, ChanState]) -> dict[str, float]:
        """Multi-timeframe encoding with ``<timeframe>__<feature>`` keys."""
        features: dict[str, float] = {}
        for timeframe in sorted(states):
            for key, value in self.encode(states[timeframe]).items():
                features[f"{timeframe}__{key}"] = value
        return features
