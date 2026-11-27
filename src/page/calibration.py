"""Clean-only basis fitting and leave-one-out exemplar-min calibration."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .core import exemplar_minimum_score


@dataclass
class CalibrationConfig:
    rank: int = 64
    clusters: int = 8
    k: int = 8
    quantile: float = 0.90
    target_fpr: float = 0.015
    alpha_min: float = 1.0
    alpha_max: float = 3.0
    alpha_step: float = 0.05
    pool: str = 'max3'
    calib_local_window: int = 1
    online_local_window: int = 9
    grid: tuple[int, int] = (16, 16)
    eta: float = 1.0


def fit_background_basis(features, rank=64, device='cuda'):
    x = torch.as_tensor(np.asarray(features), device=device, dtype=torch.float32)
    frames, cameras, patches, dim = x.shape
    mean = torch.median(x, dim=0).values
    gram = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    for camera in range(cameras):
        centered = (x[:, camera] - mean[camera]).reshape(-1, dim)
        gram += (centered.t().float() @ centered.float()).double()
    covariance = gram / float(frames * cameras * patches)
    covariance = (covariance + covariance.t()) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0).double()
    vectors = vectors.flip(1).contiguous()
    basis = vectors[:, :rank].float()
    explained = eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12)
    residual_variance = eigenvalues[:rank].sum() * (1.0 / explained - 1.0) / (dim - rank)
    return dict(mean=mean.float(), basis=basis, eigenvalues=eigenvalues,
                residual_variance=float(residual_variance.clamp_min(1e-6)),
                explained=float(explained))


def _state_clusters(states: torch.Tensor, count: int):
    mean = states.mean(0)
    scale = states.std(0, correction=0).clamp_min(1e-6)
    normalized = (states - mean) / scale
    centers = normalized[torch.linspace(0, len(states) - 1, count, device=states.device).long()].clone()
    for _ in range(40):
        labels = ((normalized[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
        updated = torch.stack([
            normalized[labels == i].mean(0) if (labels == i).any() else centers[i]
            for i in range(count)
        ])
        if torch.equal(updated, centers):
            break
        centers = updated
    return normalized, mean, scale, centers, labels


def _max_pool3(scores, grid):
    return F.max_pool2d(scores.reshape(scores.shape[0], 1, grid[0], grid[1]),
                        3, stride=1, padding=1).reshape(scores.shape[0], -1)


def calibrate_thresholds(features, states, basis, config=CalibrationConfig(), device='cuda'):
    """Calibrate thresholds from clean data with leave-one-out exemplar minima.

    The returned thresholds use the first alpha on the configured grid whose
    row-weighted clean firing rate is at most ``target_fpr``.
    """
    x = torch.as_tensor(np.asarray(features), device=device, dtype=torch.float32)
    state = torch.as_tensor(np.asarray(states), device=device, dtype=torch.float32)
    normalized, state_mean, state_scale, centers, labels = _state_clusters(state, config.clusters)
    mean = basis['mean'].to(device)
    U = basis['basis'][:, :config.rank].to(device)
    residual = []
    for camera in range(x.shape[1]):
        delta = x[:, camera] - mean[camera]
        projection = delta @ U
        residual.append(delta - projection @ U.t())
    residual = torch.stack(residual, dim=1)
    residual_mean = residual.mean(0)
    residual_scale = residual.std(0, correction=0).clamp_min(1e-6)

    distances = ((normalized[:, None] - normalized[None]) ** 2).sum(-1)
    distances.fill_diagonal_(float('inf'))
    neighbour_indices = torch.topk(distances, config.k, largest=False).indices
    scores = torch.empty((len(state), x.shape[1], x.shape[2]), device=device)
    selected_rows = torch.empty((len(state), x.shape[1], x.shape[2]), device=device, dtype=torch.long)
    for row in range(len(state)):
        neighbours = neighbour_indices[row]
        for camera in range(x.shape[1]):
            score = exemplar_minimum_score(
                residual[row, camera], residual[neighbours, camera], residual_scale[camera],
                grid=config.grid, local_window=config.calib_local_window,
            )
            scores[row, camera] = score
            distance = ((residual[row, camera][None] - residual[neighbours, camera]) /
                        residual_scale[camera][None]).square().mean(-1)
            selected_rows[row, camera] = neighbours[distance.argmin(0)]
    pooled = _max_pool3(scores.reshape(len(state) * x.shape[1], -1), config.grid)
    pooled = pooled.reshape(len(state), x.shape[1], -1)

    q90 = torch.empty((config.clusters, x.shape[1], x.shape[2]), device=device)
    for cluster in range(config.clusters):
        block = pooled[labels == cluster]
        if len(block) == 0:
            block = pooled
        q90[cluster] = torch.quantile(block, config.quantile, dim=0)

    chosen = None
    fpr_by_alpha = []
    for alpha in torch.arange(config.alpha_min, config.alpha_max + config.alpha_step / 2,
                              config.alpha_step, device=device):
        threshold = alpha * q90[labels]
        fpr = float((pooled > threshold).float().mean().item())
        fpr_by_alpha.append((float(alpha), fpr))
        if chosen is None and fpr <= config.target_fpr:
            chosen = float(alpha)
    if chosen is None:
        raise RuntimeError('alpha grid did not reach target clean FPR')

    return {
        'state_mean': state_mean.cpu(),
        'state_scale': state_scale.cpu(),
        'centers': centers.cpu(),
        'residual_mean': residual_mean.cpu(),
        'residual_scale': residual_scale.cpu(),
        'selected_rows': selected_rows.cpu(),
        'thresholds': (chosen * q90).cpu(),
        'score_mean': torch.zeros((config.clusters, x.shape[1], x.shape[2], 3)),
        'score_scale': torch.ones((config.clusters, x.shape[1], x.shape[2], 3)),
        'alpha': chosen,
        'fpr_by_alpha': fpr_by_alpha,
        'config': config.__dict__,
    }
