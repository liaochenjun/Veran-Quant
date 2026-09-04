from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.market.geometry import GeometryFeatureExtractor


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "close_time": datetime.fromisoformat("2026-08-20T14:30:59"),
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            },
            {
                "close_time": datetime.fromisoformat("2026-08-20T14:31:59"),
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
            },
            {
                "close_time": datetime.fromisoformat("2026-08-20T14:33:59"),
                "high": 999.0,
                "low": 1.0,
                "close": 500.0,
            },
        ]
    )


def test_geometry_features_ignore_future_rows_when_as_of_provided():
    candles = _candles()

    as_of = datetime.fromisoformat("2026-08-20T14:32:18")
    extractor = GeometryFeatureExtractor(lookback=10)
    features = extractor.extract(candles, as_of_timestamp=as_of)

    assert features["previous_high"] == 102.0
    assert features["previous_low"] == 98.0


def test_distance_from_current_price_is_positive_distance_to_resistance():
    # Regression: used to be last_close - resistance_line, which is always <= 0.
    candles = _candles()
    extractor = GeometryFeatureExtractor(lookback=10)

    features = extractor.extract(candles, as_of_timestamp=datetime.fromisoformat("2026-08-20T14:32:18"))
    assert features["resistance_line"] == 102.0
    assert features["distance_from_current_price"] == pytest.approx(102.0 - 101.0)
    assert features["distance_from_current_price"] >= 0


def test_empty_candles_return_defaults():
    features = GeometryFeatureExtractor().extract(pd.DataFrame())
    assert features["previous_high"] == 0.0
    assert features["resistance_line"] == 0.0
    assert features["slope"] == 0.0
    assert features["local_trend_structure"] == "flat"


def test_trend_detection():
    extractor = GeometryFeatureExtractor(lookback=10)

    def frame_with_closes(closes):
        rows = []
        for i, close in enumerate(closes):
            rows.append(
                {
                    "close_time": datetime.fromisoformat("2026-08-20T14:30:59") + pd.Timedelta(minutes=i),
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                }
            )
        return pd.DataFrame(rows)

    assert extractor.extract(frame_with_closes([100.0, 101.0, 102.0]))["local_trend_structure"] == "uptrend"
    assert extractor.extract(frame_with_closes([102.0, 101.0, 100.0]))["local_trend_structure"] == "downtrend"
    assert extractor.extract(frame_with_closes([100.0, 100.0, 100.0]))["local_trend_structure"] == "flat"


def test_mixed_tz_aware_as_of_against_naive_candles():
    # Regression: naive close_time vs aware as_of used to raise a pandas TypeError.
    candles = _candles()
    extractor = GeometryFeatureExtractor(lookback=10)

    as_of = datetime.fromisoformat("2026-08-20T14:32:18").replace(tzinfo=timezone.utc)
    features = extractor.extract(candles, as_of_timestamp=as_of)

    assert features["previous_high"] == 102.0
    assert features["previous_low"] == 98.0
