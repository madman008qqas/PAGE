"""Clean-data calibration for the PAGE patch gate.

The functions operate on arrays shaped as ``(frames, cameras, patches, dim)``
and a matching proprioceptive state array shaped as ``(frames, state_dim)``.
All matrix-heavy operations run on the selected torch device.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationConfig:
    rank: int = 64
    clusters: int = 8
    k: int = 8
    quantile: float = 0.90
    threshold_scale: float = 1.50
    pool: str = 'max3'
    grid: tuple = (16, 16)


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


def state_normalize(states):
    state = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    mean = state.mean(0)
    scale = state.std(0, correction=0).clamp_min(1e-6)
    return state, mean, scale


def knn_indices(states, query, k=8):
    normalized, mean, scale = state_normalize(states)
    q = (torch.as_tensor(query, dtype=torch.float32) - mean) / scale
    distance = ((normalized - q) ** 2).sum(-1)
    return torch.topk(distance, k, largest=False).indices.cpu().numpy()


def _max_pool(scores, grid):
    frames, patches = scores.shape
    height, width = grid
    if height * width != patches:
        raise ValueError('pooling grid does not match the patch count')
    return F.max_pool2d(scores.reshape(frames, 1, height, width), 3, 1, 1).reshape(frames, patches)


def calibrate_thresholds(features, states, basis, config=CalibrationConfig(), device='cuda'):
    """Fit state-cluster score statistics and clean thresholds."""
    x = torch.as_tensor(np.asarray(features), device=device, dtype=torch.float32)
    state, state_mean, state_scale = state_normalize(states)
    state = state.to(device)
    z_state = (state - state_mean.to(device)) / state_scale.to(device)
    centers = z_state[torch.linspace(0, len(z_state) - 1, config.clusters, device=device).long()]
    for _ in range(8):
        labels = ((z_state[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
        centers = torch.stack([z_state[labels == i].mean(0) if (labels == i).any() else centers[i]
                               for i in range(config.clusters)])
    background_mean = basis['mean'].to(device)
    background_basis = basis['basis'].to(device)
    residual = []
    for camera in range(x.shape[1]):
        delta = x[:, camera] - background_mean[camera]
        projection = delta @ background_basis[:, :config.rank]
        residual.append(delta - projection @ background_basis[:, :config.rank].t())
    residual = torch.stack(residual, dim=1)
    scale = residual.std(0, correction=0).clamp_min(1e-6)
    terms = ((residual / scale) ** 2).mean(-1)
    thresholds = torch.empty((config.clusters, x.shape[1], x.shape[2]), device=device)
    for cluster in range(config.clusters):
        block = terms[labels == cluster]
        if not len(block):
            block = terms
        score = block.mean(-1)
        pooled = _max_pool(score, config.grid) if config.pool == 'max3' else score
        thresholds[cluster] = config.threshold_scale * torch.quantile(pooled, config.quantile, dim=0)
    return dict(state_mean=state_mean.cpu(), state_scale=state_scale.cpu(),
                centers=centers.cpu(), residual_scale=scale.cpu(),
                thresholds=thresholds.cpu(), config=config.__dict__)
