"""Multiple point-in-time test windows (P2-1.5).

Windows are auto-generated from the dataset time range — never hardcoded
dates. For each window: train = ALL samples strictly before window start,
test = samples in [start, end). Windows lacking trader diversity,
LONG/SHORT balance, or minimum sizes are SKIPPED with an explicit reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.dataset.behavior_dataset import BehaviorSample


@dataclass(slots=True)
class TestWindow:
    index: int
    start: str
    end: str
    train_samples: list[BehaviorSample] = field(default_factory=list)
    test_samples: list[BehaviorSample] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def _as_utc(timestamp: str) -> datetime:
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_test_windows(
    samples: list[BehaviorSample],
    n_windows: int = 4,
    min_train_samples: int = 20,
    min_test_samples: int = 15,
    min_traders: int = 2,
    min_long: int = 3,
    min_short: int = 3,
) -> list[TestWindow]:
    ordered = sorted(samples, key=lambda s: s.timestamp)
    if not ordered:
        return []
    start_ts = _as_utc(ordered[0].timestamp)
    end_ts = _as_utc(ordered[-1].timestamp)
    span_seconds = (end_ts - start_ts).total_seconds()
    window_span = span_seconds / n_windows

    windows: list[TestWindow] = []
    for i in range(n_windows):
        w_start = start_ts + timedelta(seconds=i * window_span)
        w_end = (
            start_ts + timedelta(seconds=(i + 1) * window_span)
            if i < n_windows - 1
            else end_ts + timedelta(seconds=1)
        )
        train = [s for s in ordered if _as_utc(s.timestamp) < w_start]
        test = [s for s in ordered if w_start <= _as_utc(s.timestamp) < w_end]
        window = TestWindow(
            index=i,
            start=w_start.isoformat(),
            end=w_end.isoformat(),
            train_samples=train,
            test_samples=test,
        )

        reasons = []
        if len(train) < min_train_samples:
            reasons.append(f"insufficient train ({len(train)})")
        if len(test) < min_test_samples:
            reasons.append(f"insufficient test ({len(test)})")
        if len({s.kol for s in test}) < min_traders:
            reasons.append("too few traders in test")
        longs = sum(1 for s in test if s.action == "LONG")
        shorts = len(test) - longs
        if longs < min_long or shorts < min_short:
            reasons.append(f"insufficient LONG/SHORT balance ({longs}/{shorts})")
        if reasons:
            window.skipped = True
            window.skip_reason = "; ".join(reasons)
        windows.append(window)
    return windows
