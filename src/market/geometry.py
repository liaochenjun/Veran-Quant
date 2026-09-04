from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


def _as_aware_utc(dt: datetime) -> datetime:
    """Normalize to tz-aware UTC, interpreting naive datetimes as UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _close_times_aware_utc(close_times: pd.Series) -> pd.Series:
    """Normalize a close_time series to tz-aware UTC (naive values assumed UTC)."""
    if close_times.dt.tz is None:
        return close_times.dt.tz_localize("UTC")
    return close_times.dt.tz_convert("UTC")


@dataclass(slots=True)
class GeometryFeatureExtractor:
    lookback: int = 50

    def extract(self, candles: pd.DataFrame, as_of_timestamp: datetime | None = None) -> dict[str, float | str]:
        if candles.empty:
            return {
                "previous_high": 0.0,
                "previous_low": 0.0,
                "resistance_line": 0.0,
                "support_line": 0.0,
                "slope": 0.0,
                "angle": 0.0,
                "distance_from_current_price": 0.0,
                "number_of_touches": 0.0,
                "number_of_breaks": 0.0,
                "line_strength": 0.0,
                "local_trend_structure": "flat",
            }

        causal = candles.copy()
        if as_of_timestamp is not None and "close_time" in causal.columns:
            # Normalize both sides so naive/aware mixes compare by instant
            # instead of raising a pandas TypeError.
            mask = _close_times_aware_utc(causal["close_time"]) < _as_aware_utc(as_of_timestamp)
            causal = causal[mask]
        causal = causal.tail(self.lookback)
        if causal.empty:
            return self.extract(pd.DataFrame(), as_of_timestamp=None)

        previous_high = float(causal["high"].max())
        previous_low = float(causal["low"].min())
        resistance_line = previous_high
        support_line = previous_low

        first_close = float(causal.iloc[0]["close"])
        last_close = float(causal.iloc[-1]["close"])
        slope = (last_close - first_close) / max(len(causal) - 1, 1)
        angle = math.degrees(math.atan(slope))

        # Distance from the current price up to the resistance line (>= 0).
        distance_from_current_price = resistance_line - last_close
        touches = float((causal["high"] >= resistance_line * 0.999).sum())
        breaks = float((causal["close"] > resistance_line).sum())
        line_strength = touches - breaks

        if slope > 0:
            trend = "uptrend"
        elif slope < 0:
            trend = "downtrend"
        else:
            trend = "flat"

        return {
            "previous_high": previous_high,
            "previous_low": previous_low,
            "resistance_line": resistance_line,
            "support_line": support_line,
            "slope": slope,
            "angle": angle,
            "distance_from_current_price": distance_from_current_price,
            "number_of_touches": touches,
            "number_of_breaks": breaks,
            "line_strength": line_strength,
            "local_trend_structure": trend,
        }
