from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.chan_engine import SimpleChanEngine
from src.data.storage import DuckDBStorage
from src.dataset.behavior_dataset import BehaviorDataset
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import PointInTimeMarketState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build behavior cloning dataset")
    parser.add_argument("--trades-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_trades(csv_path: Path) -> list[KOLTrade]:
    trades: list[KOLTrade] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(
                KOLTrade(
                    kol=row["kol"],
                    symbol=row["symbol"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    side=row["side"],
                    entry_price=float(row["entry_price"]),
                )
            )
    return trades


def main() -> None:
    args = parse_args()
    storage = DuckDBStorage(root_dir=Path("data/raw"), database_path=Path("data/database/market.duckdb"))
    pit = PointInTimeMarketState(storage=storage)
    aligner = TradeAligner(point_in_time=pit)
    chan_engine = SimpleChanEngine(point_in_time=pit)
    geometry_extractor = GeometryFeatureExtractor()

    trades = load_trades(Path(args.trades_csv))
    dataset = BehaviorDataset.from_trades(
        trades=trades,
        aligner=aligner,
        chan_engine=chan_engine,
        geometry_extractor=geometry_extractor,
    )

    output = [sample.__dict__ for sample in dataset.samples]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
