"""PAGE's exemplar-minimum gate and PAR feature repair."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _max_pool(scores: torch.Tensor, grid: tuple[int, int], kernel: int) -> torch.Tensor:
    height, width = grid
    padding = kernel // 2
    x = scores.reshape(scores.shape[0], 1, height, width)
    return F.max_pool2d(x, kernel, stride=1, padding=padding).reshape(scores.shape[0], height * width)


def _channel_mean_squared(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean(-1)


@torch.inference_mode()
def exemplar_minimum_score(query_perp: torch.Tensor, reference_perp: torch.Tensor,
                           residual_scale: torch.Tensor, grid: tuple[int, int] = (16, 16),
                           local_window: int = 9) -> torch.Tensor:
    """Return the minimum normalized residual over K exemplars and a local window."""
    if reference_perp.ndim != 3:
        raise ValueError("reference_perp must have shape (K, patches, channels)")
    if local_window % 2 == 0:
        raise ValueError("local_window must be odd")
    scale = residual_scale.clamp_min(1e-6)
    distance = _channel_mean_squared((query_perp[None] - reference_perp) / scale[None])
    local = -F.max_pool2d(
        -distance.reshape(distance.shape[0], 1, grid[0], grid[1]),
        local_window,
        stride=1,
        padding=local_window // 2,
    ).reshape(distance.shape[0], grid[0] * grid[1])
    return local.min(0).values


@torch.inference_mode()
def patch_intervention_normres_v4(
    y: torch.Tensor,
    mu: torch.Tensor,
    ref: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    sig2: float,
    thresholds: torch.Tensor,
    tau: float,
    rho: float | None = None,
    residual_mean: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
    cluster_id: int = 0,
    score_mean: torch.Tensor | None = None,
    score_scale: torch.Tensor | None = None,
    pool: str = 'max3',
    valid_start: int = 0,
    grid: tuple[int, int] = (16, 16),
    reference_bank: torch.Tensor | None = None,
    local_window: int = 9,
    repair_mode: str = 'par',
) -> tuple[torch.Tensor, dict]:
    """Score and repair one camera's patch features.

    ``reference_bank`` contains the K pose-matched clean feature tensors. When it
    is supplied, the gate uses the minimum normalized residual over those K
    exemplars and each patch's local ``local_window`` neighbourhood. The edit
    target remains ``ref``, the original-position K-exemplar mean.

    ``repair_mode='par'`` applies the reliability-guided soft contraction
    ``m_i + w_i (y_i - m_i)``. Omitting ``rho`` selects the current unbudgeted
    contract with unit edit strength.
    """
    if y.ndim != 3 or y.shape[0] != 1:
        raise ValueError(f"y must have shape (1, patches, channels), got {tuple(y.shape)}")
    if repair_mode not in {'par', 'hard_replace'}:
        raise ValueError("repair_mode must be 'par' or 'hard_replace'")
    patches = grid[0] * grid[1]
    stop = valid_start + patches
    if stop > y.shape[1]:
        raise ValueError("valid patch range exceeds feature tensor")
    yv = y[:, valid_start:stop].float()
    if residual_mean is None:
        residual_mean = torch.zeros_like(mu)
    if residual_scale is None:
        residual_scale = torch.ones_like(mu)
    rs = residual_scale.clamp_min(1e-6)

    d = yv - mu
    z = d @ U
    perp = d - z @ U.t()
    normalized = (perp - residual_mean) / rs
    residual_score = _channel_mean_squared(normalized)[0]

    if reference_bank is not None:
        bank = reference_bank.float()
        bank_delta = bank - mu[None]
        bank_latent = bank_delta @ U
        bank_perp = bank_delta - bank_latent @ U.t()
        gate_score = exemplar_minimum_score(
            perp[0], bank_perp, rs, grid=grid, local_window=local_window
        )
    else:
        gate_score = residual_score

    if pool == 'max3':
        gate_score = _max_pool(gate_score[None], grid, 3)[0]
    elif pool != 'none':
        raise ValueError("pool must be 'none' or 'max3'")

    if score_mean is not None and score_scale is not None:
        gate_score = (gate_score - score_mean[:, 0]) / score_scale[:, 0].clamp_min(1e-6)
    mask = gate_score > thresholds[cluster_id]

    score = (z.square() / lam).sum(-1) + (perp.square() / sig2).sum(-1)
    log_weight = -score
    spread = log_weight.std(1, keepdim=True, correction=0).clamp_min(1e-6)
    weight = torch.exp((log_weight - log_weight.max(1, keepdim=True).values) / (tau * spread))
    if repair_mode == 'hard_replace':
        weight = torch.zeros_like(weight)
    candidate_delta = ref[None] + weight[..., None] * (yv - ref[None]) - yv
    if rho is None:
        eta = torch.ones((1, 1, 1), device=y.device, dtype=y.dtype)
    else:
        raw = candidate_delta.abs().sum((-1, -2), keepdim=True)
        denominator = d.abs().sum((-1, -2), keepdim=True)
        eta = torch.where(raw > 1e-12, torch.minimum(torch.ones_like(raw), rho * denominator / raw),
                          torch.zeros_like(raw))
    edited = yv + eta * candidate_delta
    out = y.clone()
    out[:, valid_start:stop] = torch.where(mask[..., None], edited, yv).to(y.dtype)
    actual = out[:, valid_start:stop].float() - yv

    return out, {
        'gate_active': bool(mask.any()),
        'patch_active_count': int(mask.sum()),
        'patch_total_count': int(mask.numel()),
        'patch_mask': mask[0].tolist(),
        'patch_scores': gate_score.tolist(),
        'patch_residual_scores': residual_score.tolist(),
        'patch_thresholds': thresholds[cluster_id].tolist(),
        'patch_pool': pool,
        'local_window': int(local_window),
        'support_mode': 'exemplar_min' if reference_bank is not None else 'same_position',
        'repair_mode': repair_mode,
        'eta': float(eta.max()),
        'delta_l1': float(candidate_delta.abs().sum()),
        'applied_delta_l1': float(actual.abs().sum()),
        'changed_patch_count': int((actual.abs().sum(-1) > 0).sum()),
    }
