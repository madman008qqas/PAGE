"""PAGE patch-wise anomaly-gated editing."""
from .core import exemplar_minimum_score, patch_intervention_normres_v4
from .reference import retrieve_k8, retrieve_k8_bank
from .calibration import CalibrationConfig, fit_background_basis, calibrate_thresholds

__all__ = [
    'exemplar_minimum_score',
    'patch_intervention_normres_v4',
    'retrieve_k8',
    'retrieve_k8_bank',
    'CalibrationConfig',
    'fit_background_basis',
    'calibrate_thresholds',
]
