"""VayuCast coupled feature engineering.

- inversion.compute_inversion_features  -> Inversion Strength Index (met -> chem trapping)
- stubble.compute_stubble_features      -> stubble-plume transport vector
- feedback.compute_feedback_features    -> aerosol -> radiation -> PBL interaction terms
- calendar_feats.add_calendar_features  -> temporal / solar-geometry features
- build.build_matrix                    -> the assembled per-station hourly feature matrix
"""

from features.build import FEATURE_GROUPS, build_matrix
from features.calendar_feats import add_calendar_features
from features.feedback import compute_feedback_features
from features.inversion import compute_inversion_features
from features.stubble import compute_stubble_features

__all__ = [
    "FEATURE_GROUPS",
    "build_matrix",
    "add_calendar_features",
    "compute_feedback_features",
    "compute_inversion_features",
    "compute_stubble_features",
]
