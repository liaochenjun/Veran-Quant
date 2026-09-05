"""Audit tests: trader one-hot, forbidden feature terms, split discipline."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.features.state_features import KNOWN_TRADERS, StateFeatureAssembler
from src.market.point_in_time import MarketState


def _sample(timestamp: str, side: str, kol: str = "aoying_capital") -> BehaviorSample:
    return BehaviorSample(
        kol=kol, symbol="S", timestamp=timestamp, side=side,
        market_state={}, chan_state={}, geometry_features={}, chan_states={},
    )


def test_trader_one_hot_known_unknown_and_none():
    assembler = StateFeatureAssembler()

    known = assembler.from_sample(_sample("2026-08-01T00:00:00+00:00", "LONG", kol="aoying_capital"))
    assert known["trader__aoying_capital"] == 1.0
    assert known["trader__liuyuan"] == 0.0
    assert known["trader__other"] == 0.0

    unknown = assembler.from_sample(_sample("2026-08-01T00:00:00+00:00", "LONG", kol="brand_new_trader"))
    assert all(unknown[f"trader__{t}"] == 0.0 for t in KNOWN_TRADERS)
    assert unknown["trader__other"] == 1.0  # explicit unknown bucket, never silent

    # trader_id=None (live call without identity): all trader bits zero
    none_features = assembler.from_components(
        MarketState(symbol="S", as_of_timestamp=datetime(2026, 8, 1), frames={}),
        {},
        trader_id=None,
    )
    assert all(none_features[f"trader__{t}"] == 0.0 for t in KNOWN_TRADERS)
    assert none_features["trader__other"] == 0.0


def test_feature_keys_contain_no_outcome_or_influence_terms():
    assembler = StateFeatureAssembler()
    features = assembler.from_sample(_sample("2026-08-01T00:00:00+00:00", "LONG"))
    forbidden = (
        "pnl", "mfe", "mae", "closed", "exit_price", "future", "quality",
        "impact", "influence", "weight", "factor", "realized",
    )
    bad = [k for k in features if any(term in k for term in forbidden)]
    assert bad == []


def test_no_random_split_primitives_in_source():
    src = Path(__file__).resolve().parents[1] / "src"
    forbidden = re.compile(
        r"train_test_split|random\.shuffle|np\.random\.shuffle|shuffle\(|sklearn\.model_selection"
    )
    hits = []
    for path in sorted(src.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{path}:{i}: {line.strip()}")
    assert hits == []


def test_chronological_split_70_15_15_preserves_order():
    samples = [_sample(f"2026-08-{d:02d}T00:00:00+00:00", "LONG") for d in range(1, 21)]
    train, val, test = BehaviorDataset(samples=samples).chronological_split()

    assert (len(train), len(val), len(test)) == (14, 3, 3)  # 70/15/15 of 20
    combined = train + val + test
    assert [s.timestamp for s in combined] == sorted(s.timestamp for s in combined)
    # test is strictly the newest block
    assert test[0].timestamp > val[-1].timestamp
    assert val[0].timestamp > train[-1].timestamp
