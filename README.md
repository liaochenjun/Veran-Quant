# KOL Trading Digital Twin

Point-in-time-safe research skeleton for behavior cloning of crypto trading KOLs.

## Goal

First stage is **behavior cloning**:
- Input: market information observable before each KOL trade timestamp.
- Target: KOL action (`LONG` / `SHORT`).

This is not a pure price prediction system.

## Architecture

```text
Binance Data
  -> DuckDB + Parquet storage
  -> PointInTimeMarketState (strict causal access)
  -> TradeAligner
  -> Geometry + Chan abstractions
  -> BehaviorDataset
  -> BehaviorModel + ReplayValidator
```

## Point-in-time constraints

`PointInTimeMarketState.get_market_state(symbol, as_of_timestamp)` is the central causal access interface.

Rules enforced:
- only fully completed candles are used;
- every returned candle must satisfy `close_time < as_of_timestamp`;
- no currently forming candle is allowed.

This applies to all supported timeframes:
- `1m`, `5m`, `15m`, `1h`, `4h`.

## Project layout

- `src/data`: Binance client, downloader, DuckDB/Parquet storage.
- `src/alignment`: normalized trade schema and trade alignment.
- `src/market`: point-in-time access and geometry/market feature modules.
- `src/chan`: causal point-in-time chan engine (chan.py adapter, ChanState,
  feature encoder); `SimpleChanEngine` remains as a lightweight fallback.
- `src/dataset`: behavior dataset (incl. optional multi-timeframe chan
  snapshots per sample) + chronological split.
- `src/models`: model interface/baseline and replay validator.
- `scripts`: download data, build dataset, train baseline.
- `tests`: leakage-focused unit tests.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
git submodule update --init -- third_party/chan.py
```

(chan.py is integrated as a pinned git submodule; see `docs/chan-integration.md`.)

Run tests:

```bash
pytest -q
```

Download klines:

```bash
python scripts/download_klines.py \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2026-08-01T00:00:00 \
  --end 2026-08-02T00:00:00
```

Build dataset from normalized trade CSV (each sample freezes the chan state
of every supported timeframe — 1m/5m/15m/1h — at the trade moment):

```bash
python scripts/build_dataset.py --trades-csv data/processed/trades.csv --output data/processed/behavior_dataset.json
```

Train baseline:

```bash
python scripts/train.py --dataset data/processed/behavior_dataset.json
```

## Notes

- Do not hard-code credentials. Optional Binance key is read from `BINANCE_API_KEY`.
- Chronological split is used (`70/15/15`), never random split.
- Outcome fields (`pnl`, `mfe`, `mae`, etc.) are optional metadata and not model inputs in the first stage.
