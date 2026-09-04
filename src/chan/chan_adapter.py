"""Adapter isolating the external chan.py library from the rest of the project.

Integration facts (see docs/chan-integration.md):
- Source: https://github.com/Vespa314/chan.py (MIT License)
- Pinned commit: see ``git submodule status`` / .gitmodules (third_party/chan.py)
- chan.py is not on PyPI and ships no packaging metadata, so it is vendored
  as a git submodule and imported via a sys.path bootstrap.
- chan.py master requires Python 3.11 (``typing.Self``).  On 3.10 the shim
  below backports ``typing.Self`` from ``typing_extensions``.
"""

from __future__ import annotations

import sys
import typing
from datetime import datetime, timezone
from pathlib import Path

CHANPY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"

# chan.py KL_TYPE has no 4h level (K_1M .. K_60M, K_DAY).  The 4h timeframe is
# therefore reported as unsupported (ChanState.supported=False), never faked.
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h")

_CONFIG_DEFAULTS = {
    "trigger_step": True,  # external-feeding mode; __init__ then skips data loading
    "print_warning": False,
}


def _patch_typing_self() -> None:
    if not hasattr(typing, "Self"):
        from typing_extensions import Self  # noqa: F401

        typing.Self = Self


def _ensure_chanpy_importable() -> None:
    if not CHANPY_ROOT.exists():
        raise ImportError(
            "chan.py submodule is missing. Run: git submodule update --init -- third_party/chan.py"
        )
    _patch_typing_self()
    if str(CHANPY_ROOT) not in sys.path:
        sys.path.insert(0, str(CHANPY_ROOT))


_ensure_chanpy_importable()

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, FX_TYPE  # noqa: E402
from Common.CTime import CTime  # noqa: E402
from KLine.KLine_Unit import CKLine_Unit  # noqa: E402

from src.chan.chan_state import ChanState  # noqa: E402

_TIMEFRAME_TO_KL_TYPE: dict[str, object] = {}


def _timeframe_to_kl_type(timeframe: str):
    if not _TIMEFRAME_TO_KL_TYPE:
        # KL_TYPE is imported lazily to keep the module import cheap
        from Common.CEnum import KL_TYPE

        _TIMEFRAME_TO_KL_TYPE.update(
            {
                "1m": KL_TYPE.K_1M,
                "5m": KL_TYPE.K_5M,
                "15m": KL_TYPE.K_15M,
                "1h": KL_TYPE.K_60M,
            }
        )
    return _TIMEFRAME_TO_KL_TYPE.get(timeframe)


def _to_ctime(dt: datetime) -> CTime:
    """Convert to chan.py CTime.

    chan.py CTime is tz-naive wall clock; the whole pipeline keeps UTC wall
    clock, so naive datetimes here must already be UTC.  ``auto=False``
    disables the A-share "hour==minute==0 means end of day" convention,
    which is wrong for 24/7 crypto markets.
    """
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)


def _ctime_to_datetime(ctime: CTime) -> datetime:
    return datetime(ctime.year, ctime.month, ctime.day, ctime.hour, ctime.minute, ctime.second)


class ChanPyAdapter:
    """Incremental chan.py wrapper for one symbol/timeframe.

    Bars must be fed strictly in chronological order; every snapshot only
    reflects the bars fed so far (chan.py's documented causal guarantee).
    """

    def __init__(self, timeframe: str, config: dict | None = None) -> None:
        kl_type = _timeframe_to_kl_type(timeframe)
        if kl_type is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}; chan.py levels are: {SUPPORTED_TIMEFRAMES}"
            )
        self._kl_type = kl_type

        merged = dict(_CONFIG_DEFAULTS)
        merged.update(config or {})
        self._chan = CChan(
            code="LOCAL",
            begin_time=None,
            end_time=None,
            data_src=DATA_SRC.CSV,  # unused: bars are fed externally
            lv_list=[self._kl_type],
            config=CChanConfig(merged),
            autype=AUTYPE.QFQ,  # unused for externally fed bars
        )
        self._last_bar_close_time: datetime | None = None

    def feed_bar(self, bar_close_time: datetime, open_: float, high: float, low: float, close: float) -> None:
        """Feed one fully closed bar; chan.py intraday time = bar END time."""
        if self._last_bar_close_time is not None and bar_close_time <= self._last_bar_close_time:
            raise ValueError(
                f"Bars must be fed in strict chronological order: "
                f"got {bar_close_time}, last was {self._last_bar_close_time}"
            )
        klu = CKLine_Unit(
            {
                DATA_FIELD.FIELD_TIME: _to_ctime(bar_close_time),
                DATA_FIELD.FIELD_OPEN: float(open_),
                DATA_FIELD.FIELD_CLOSE: float(close),
                DATA_FIELD.FIELD_HIGH: float(high),
                DATA_FIELD.FIELD_LOW: float(low),
            }
        )
        self._chan.trigger_load({self._kl_type: [klu]})
        self._last_bar_close_time = bar_close_time

    def snapshot(self, symbol: str, timeframe: str, as_of_timestamp: datetime) -> ChanState:
        """Freeze the current state into an immutable ChanState."""
        state = ChanState.empty(symbol, timeframe, as_of_timestamp)
        ckl = self._chan.kl_datas[self._kl_type]
        klines = ckl.lst
        if not klines:
            return state

        last_klc = klines[-1]
        last_klu = last_klc[-1]
        state.last_bar_close_time = _ctime_to_datetime(last_klu.time).replace(tzinfo=timezone.utc).isoformat()

        # fractal on the last merged kline
        if last_klc.fx == FX_TYPE.TOP:
            state.fractal_present = True
            state.fractal_type = "TOP"
            state.fractal_price = float(last_klc.high)
        elif last_klc.fx == FX_TYPE.BOTTOM:
            state.fractal_present = True
            state.fractal_type = "BOTTOM"
            state.fractal_price = float(last_klc.low)

        # latest bi
        bis = list(ckl.bi_list)
        state.bi_count = len(bis)
        if bis:
            bi = bis[-1]
            state.bi_direction = bi.dir.name
            state.bi_is_sure = bool(bi.is_sure)
            state.bi_amplitude = float(bi.amp())
            state.bi_length = len(list(bi.klc_lst))

        # latest segment
        segs = list(ckl.seg_list)
        state.segment_count = len(segs)
        if segs:
            seg = segs[-1]
            state.segment_direction = seg.dir.name
            state.segment_is_sure = bool(seg.is_sure)

        # latest zhongshu
        zss = list(ckl.zs_list)
        state.zs_count = len(zss)
        if zss:
            zs = zss[-1]
            state.zhongshu_present = True
            state.zhongshu_high = float(zs.high)
            state.zhongshu_low = float(zs.low)
            state.zhongshu_is_sure = bool(zs.is_sure)
            last_close = float(last_klu.close)
            if last_close > state.zhongshu_high:
                state.distance_to_zhongshu = last_close - state.zhongshu_high
            elif last_close < state.zhongshu_low:
                state.distance_to_zhongshu = last_close - state.zhongshu_low  # negative
            else:
                state.distance_to_zhongshu = 0.0

        # latest buy/sell point
        bsps = self._chan.get_latest_bsp(number=1)
        if bsps:
            bsp = bsps[0]
            state.buy_sell_point_present = True
            state.buy_sell_point_types = [t.value for t in bsp.type]
            state.buy_sell_point_is_buy = bool(bsp.is_buy)
            state.buy_sell_point_bi_is_sure = bool(bsp.bi.is_sure)
            state.buy_sell_point_time = (
                _ctime_to_datetime(bsp.klu.time).replace(tzinfo=timezone.utc).isoformat()
            )

        return state
