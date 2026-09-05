"""Feature ablation: named feature-group definitions and key filtering.

Groups are defined by feature-key prefixes only. Filtering never reads
the label — two samples differing only in their action produce identical
filtered feature dicts (tested).

A. market_only            market features WITHOUT geometry
B. market_geometry        all market features (incl. per-tf geometry)
C. market_chan            market (no geometry) + chan
D. market_chan_geometry   market (incl. geometry) + chan
E. full                   D + trader identity one-hots
"""

from __future__ import annotations

from typing import Callable


def _is_market_base(key: str) -> bool:
    return key.startswith("market__") and "__geometry__" not in key


def _keep_fn(feature_set: str) -> Callable[[str], bool]:
    if feature_set == "market_only":
        return _is_market_base
    if feature_set == "market_geometry":
        return lambda k: k.startswith("market__")
    if feature_set == "market_chan":
        return lambda k: _is_market_base(k) or k.startswith("chan__")
    if feature_set == "market_chan_geometry":
        return lambda k: k.startswith("market__") or k.startswith("chan__")
    if feature_set == "full":
        return lambda k: True  # market + chan + trader identity
    raise ValueError(f"unknown feature set {feature_set!r}")


FEATURE_SETS = (
    "market_only",
    "market_geometry",
    "market_chan",
    "market_chan_geometry",
    "full",
)


def filter_features(features: dict[str, float], feature_set: str) -> dict[str, float]:
    """Return the feature subset for a named group (label-independent)."""
    keep = _keep_fn(feature_set)
    return dict(sorted((k, v) for k, v in features.items() if keep(k)))
