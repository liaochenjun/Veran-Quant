from __future__ import annotations

from dataclasses import dataclass

from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.chan_engine import ChanEngine
from src.market.geometry import GeometryFeatureExtractor


@dataclass(slots=True)
class BehaviorSample:
    kol: str
    symbol: str
    timestamp: str
    side: str
    market_state: dict
    chan_state: dict
    geometry_features: dict


@dataclass(slots=True)
class BehaviorDataset:
    samples: list[BehaviorSample]

    @classmethod
    def from_trades(
        cls,
        trades: list[KOLTrade],
        aligner: TradeAligner,
        chan_engine: ChanEngine,
        geometry_extractor: GeometryFeatureExtractor,
        geometry_timeframe: str = "15m",
    ) -> "BehaviorDataset":
        # Align first so timestamps are normalized to aware UTC before sorting
        # (sorting mixed naive/aware timestamps would raise TypeError), and so
        # the sort key matches the timestamp the market state was built from.
        aligned_trades = [aligner.align_trade(trade) for trade in trades]
        aligned_trades.sort(key=lambda a: a.market_state.as_of_timestamp)

        samples: list[BehaviorSample] = []
        for aligned in aligned_trades:
            trade = aligned.trade
            as_of = aligned.market_state.as_of_timestamp
            market_state = {
                tf: frame.to_dict(orient="records") for tf, frame in aligned.market_state.frames.items()
            }
            geom_frame = aligned.market_state.frames.get(geometry_timeframe)
            geometry_features = (
                geometry_extractor.extract(geom_frame, as_of_timestamp=as_of)
                if geom_frame is not None
                else {}
            )
            sample = BehaviorSample(
                kol=trade.kol,
                symbol=trade.symbol,
                timestamp=as_of.isoformat(),
                side=trade.normalized_side(),
                market_state=market_state,
                chan_state=chan_engine.get_state(
                    symbol=trade.symbol,
                    timeframe=geometry_timeframe,
                    as_of_timestamp=as_of,
                    market_state=aligned.market_state,
                ),
                geometry_features=geometry_features,
            )
            samples.append(sample)
        return cls(samples=samples)

    def chronological_split(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> tuple[list[BehaviorSample], list[BehaviorSample], list[BehaviorSample]]:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")

        n = len(self.samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return self.samples[:train_end], self.samples[train_end:val_end], self.samples[val_end:]
