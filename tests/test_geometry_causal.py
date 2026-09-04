from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.market.geometry import GeometryFeatureExtractor


def test_geometry_features_ignore_future_rows_when_as_of_provided():
    candles = pd.DataFrame(
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

    as_of = datetime.fromisoformat("2026-08-20T14:32:18")
    extractor = GeometryFeatureExtractor(lookback=10)
    features = extractor.extract(candles, as_of_timestamp=as_of)

    assert features["previous_high"] == 102.0
    assert features["previous_low"] == 98.0
