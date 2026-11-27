"""Run the PAGE calibration, retrieval, gate, and repair chain on synthetic data."""

from __future__ import annotations

import numpy as np
import torch

from page import CalibrationConfig, calibrate_thresholds, fit_background_basis
from page import patch_intervention_normres_v4, retrieve_k8_bank


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(7)
    frames, cameras, patches, channels = 12, 2, 16, 8
    clean_features = rng.normal(size=(frames, cameras, patches, channels)).astype(np.float32)
    clean_states = rng.normal(size=(frames, 6)).astype(np.float32)

    basis = fit_background_basis(clean_features, rank=2, device=device)
    calibration = calibrate_thresholds(
        clean_features,
        clean_states,
        basis,
        config=CalibrationConfig(
            rank=2, clusters=2, k=2, calib_local_window=1, online_local_window=3,
            grid=(4, 4),
            alpha_min=1.0, alpha_max=5.0, alpha_step=0.25,
        ),
        device=device,
    )

    query_state = clean_states[0]
    bank, reference, _ = retrieve_k8_bank(clean_states, clean_features, query_state, k=2)
    camera = 0
    query = torch.as_tensor(clean_features[0, camera], device=device).unsqueeze(0)
    query[:, 0, :] += 8.0
    bank_t = torch.as_tensor(bank[:, camera], device=device)
    reference_t = torch.as_tensor(reference[camera], device=device)
    mean_t = basis["mean"][camera].to(device)
    basis_t = basis["basis"].to(device)
    lam_t = basis["eigenvalues"][:2].to(device)
    threshold = calibration["thresholds"][0, camera].to(device)
    residual_mean = calibration["residual_mean"][camera].to(device)
    residual_scale = calibration["residual_scale"][camera].to(device)

    _, telemetry = patch_intervention_normres_v4(
        query, mean_t, reference_t, basis_t, lam_t, basis["residual_variance"],
        threshold, tau=3.0, residual_mean=residual_mean,
        residual_scale=residual_scale, grid=(4, 4), reference_bank=bank_t,
        local_window=3, repair_mode="par",
    )
    print({"device": device, "alpha": calibration["alpha"], **telemetry})


if __name__ == "__main__":
    main()
