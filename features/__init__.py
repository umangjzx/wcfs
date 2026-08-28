"""VayuCast coupled feature engineering.

- inversion.compute_inversion_features  -> Inversion Strength Index (met -> chem trapping)
- stubble.compute_stubble_features      -> stubble-plume transport vector
- feedback.compute_feedback_features    -> aerosol -> radiation -> PBL interaction terms
- calendar_feats.add_calendar_features  -> temporal / solar-geometry features
- build.build_matrix                    -> the assembled per-station hourly feature matrix

Submodules are imported lazily so ``python -m features.build`` doesn't double-import.
"""

from importlib import import_module

__all__ = [
    "FEATURE_GROUPS", "build_matrix", "add_calendar_features",
    "compute_feedback_features", "compute_inversion_features", "compute_stubble_features",
]

_LAZY = {
    "FEATURE_GROUPS": "features.build",
    "build_matrix": "features.build",
    "add_calendar_features": "features.calendar_feats",
    "compute_feedback_features": "features.feedback",
    "compute_inversion_features": "features.inversion",
    "compute_stubble_features": "features.stubble",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
