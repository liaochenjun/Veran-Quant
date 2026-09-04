from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as `python scripts/build_dataset.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.chan_adapter import SUPPORTED_TIMEFRAMES
from src.chan.causal_chan import CausalChanEngine
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
    # Causal point-in-time chan engine (see docs/chan-integration.md);
    # requires the chan.py submodule: git submodule update --init
    chan_engine = CausalChanEngine(storage=storage)
    geometry_extractor = GeometryFeatureExtractor()

    trades = load_trades(Path(args.trades_csv))
    dataset = BehaviorDataset.from_trades(
        trades=trades,
        aligner=aligner,
        chan_engine=chan_engine,
        geometry_extractor=geometry_extractor,
        chan_timeframes=SUPPORTED_TIMEFRAMES,
    )

    output = [asdict(sample) for sample in dataset.samples]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
