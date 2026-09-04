from __future__ import annotations

from dataclasses import dataclass

from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import MarketState


@dataclass(slots=True)
class MarketFeatureBuilder:
    geometry_extractor: GeometryFeatureExtractor

    def build(self, market_state: MarketState) -> dict[str, dict]:
        features: dict[str, dict] = {}
        for timeframe, candles in market_state.frames.items():
            if candles.empty:
                features[timeframe] = {}
                continue
            last_row = candles.iloc[-1]
            features[timeframe] = {
                "last_close": float(last_row["close"]),
                "last_volume": float(last_row["volume"]),
                "mean_close": float(candles["close"].mean()),
                "geometry": self.geometry_extractor.extract(
                    candles, as_of_timestamp=market_state.as_of_timestamp
                ),
            }
        return features
