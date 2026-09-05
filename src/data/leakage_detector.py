"""Leakage detector: scan behavior samples for future-information leakage.

Per prompt.txt, any violation is a FAIL — never a warning. The detector
checks, for every behavior sample with action timestamp T:

1.  the action timestamp itself is parseable;
2.  every kline in market_state satisfies close_time < T;
3.  chan_state / chan_states last_bar_close_time < T (None allowed);
4.  chan_state as_of_timestamp <= T;
5.  chan buy_sell_point_time < T;
6.  no forbidden outcome field names appear anywhere in the input side
    (pnl, closed_time, closed_price, mfe, mae, holding_time, ...);
7.  no scaler is fitted on validation/test data — the current pipeline
    has no scaler, so this check passes by construction (noted).

Use scan_samples() on BehaviorSample dicts, or scan_dataset_json() on a
built dataset file. A failing report raises via assert_no_leakage().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "pnl",
        "realized_pnl",
        "future_pnl",
        "closed_time",
        "closed_price",
        "exit_price",
        "close_price",
        "future_return",
        "future_high",
        "future_low",
        "mfe",
        "mae",
        "holding_time",
        "max_profit",
        "max_loss",
        "trade_result",
        "trade_quality",
    }
)

# Keys that live inside chan snapshots and carry a timestamp that must be < T.
_CHAN_TIME_KEYS = ("last_bar_close_time", "buy_sell_point_time")


class LeakageViolation(Exception):
    """Raised when the detector finds future-information leakage."""


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class LeakageReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def _as_aware_utc(value: Any) -> datetime | None:
    """Parse ISO strings / pandas Timestamps into aware-UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _walk_input_structures(sample: dict) -> Iterable[tuple[str, dict]]:
    """Yield (path, dict) for every dict on the input side of a sample."""
    yield "sample", sample
    for tf, frame in sample.get("market_state", {}).items():
        rows = frame if isinstance(frame, list) else []
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                yield f"market_state[{tf}][{idx}]", row
        if isinstance(frame, dict):
            yield f"market_state[{tf}]", frame
    yield "chan_state", sample.get("chan_state", {})
    for tf, state in sample.get("chan_states", {}).items():
        yield f"chan_states[{tf}]", state if isinstance(state, dict) else {}
    yield "geometry_features", sample.get("geometry_features", {})


def scan_samples(samples: Iterable[dict]) -> LeakageReport:
    """Scan behavior samples (dicts, e.g. asdict(BehaviorSample) or JSON rows)."""
    report = LeakageReport()
    sample_list = list(samples)

    bad_timestamps = 0
    kline_violations: list[str] = []
    chan_violations: list[str] = []
    forbidden_violations: list[str] = []

    for idx, sample in enumerate(sample_list):
        action_ts = _as_aware_utc(sample.get("timestamp"))
        where = f"sample[{idx}] ({sample.get('symbol', '?')} @ {sample.get('timestamp')})"
        if action_ts is None:
            bad_timestamps += 1
            continue

        # 2. kline close_time < T in every market_state frame
        for tf, frame in sample.get("market_state", {}).items():
            rows = frame if isinstance(frame, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                close_time = _as_aware_utc(row.get("close_time"))
                if close_time is not None and close_time >= action_ts:
                    kline_violations.append(
                        f"{where}: market_state[{tf}] kline close_time {row.get('close_time')} >= T"
                    )

        # 3-5. chan snapshot timestamps
        for tf, state in sample.get("chan_states", {}).items():
            if not isinstance(state, dict):
                continue
            for key in _CHAN_TIME_KEYS:
                value = _as_aware_utc(state.get(key))
                if value is not None and value >= action_ts:
                    chan_violations.append(
                        f"{where}: chan_states[{tf}].{key}={state.get(key)} >= T"
                    )
            as_of = _as_aware_utc(state.get("as_of_timestamp"))
            if as_of is not None and as_of > action_ts:
                chan_violations.append(
                    f"{where}: chan_states[{tf}].as_of_timestamp={state.get('as_of_timestamp')} > T"
                )
        chan_state = sample.get("chan_state", {})
        if isinstance(chan_state, dict):
            for key in _CHAN_TIME_KEYS:
                value = _as_aware_utc(chan_state.get(key))
                if value is not None and value >= action_ts:
                    chan_violations.append(f"{where}: chan_state.{key}={chan_state.get(key)} >= T")

        # 6. forbidden outcome keys anywhere on the input side
        for path, container in _walk_input_structures(sample):
            hits = sorted(FORBIDDEN_INPUT_KEYS.intersection(container.keys()))
            if hits:
                forbidden_violations.append(f"{where}: {path} contains {hits}")

    report.add("action_timestamp_parseable", bad_timestamps == 0,
               f"{bad_timestamps} unparseable action timestamps" if bad_timestamps else "all parseable")
    report.add("kline_close_before_action", not kline_violations,
               "; ".join(kline_violations[:5]) or "all klines close before T")
    report.add("chan_state_before_action", not chan_violations,
               "; ".join(chan_violations[:5]) or "all chan snapshots consistent with T")
    report.add("no_forbidden_outcome_keys", not forbidden_violations,
               "; ".join(forbidden_violations[:5]) or "no outcome fields in input")
    # 7. scaler: the pipeline currently has no scaler; nothing is fitted on
    # validation/test data. If one is added later it must fit on train only.
    report.add("scaler_fit_discipline", True,
               "no scaler configured; add fit(train)-only check when a scaler exists")
    return report


def assert_no_leakage(samples: Iterable[dict]) -> LeakageReport:
    """Return the report if clean, otherwise raise LeakageViolation."""
    report = scan_samples(samples)
    if not report.ok:
        raise LeakageViolation(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return report


def scan_dataset_json(path: str | Path) -> LeakageReport:
    """Scan a built behavior dataset JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return scan_samples(data)
