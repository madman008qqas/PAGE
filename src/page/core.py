# Adapted from work-station patch_gate_core_r4.py, SHA-256:
# 7757629eb5e9cd85baf06f66ebdf79c1bdf06c5ee970962c9167731922df6a9b
# Only layout/shape parameters changed; arithmetic and budget denominator preserved.
"""Independent-patch anomaly gate core, r4.

r3 plus exactly one mechanism:

  D) score pooling -- the decision score of a patch is pooled over its spatial
     neighbourhood of patches before it is compared against the threshold:

       pool='none'   (r3 / r2 behaviour)      the patch's own score
       pool='max3'   3x3 neighbourhood, max
       pool='mean3'  3x3 neighbourhood, mean
       pool='mean5'  5x5 neighbourhood, mean

     Rationale (measured, not assumed): pooling raises the interference detection
     rate at any fixed clean firing rate, but only if the threshold is recalibrated
     on the pooled score -- dilating the mask after the fact blows the clean side up
     from 11.86% to 34.36%. The threshold therefore has to be built from pooled
     scores, which is what the r4 gate builder does.

The adapter supplies the valid patch grid (VLA-Arena: full 16x16).
The original RoboTwin layout is recovered with grid=(12,16), valid_start=32.
Padding follows the original scripts exactly:
max-pooling uses torch's implicit -inf padding, mean-pooling replicates the edge.

With `pool='none'` no extra arithmetic is executed, so the operator is bitwise
identical to `patch_intervention_normres_v3` (asserted by test_patch_gate_core_r4).
"""

import torch
import torch.nn.functional as F

NTERMS = 3
POOLS = ('none', 'max3', 'mean3', 'mean5')


def _agg_channels(sq, topk_channels):
    """Reduce a (1,192,D) tensor of squared standardised residuals over channels."""
    if topk_channels is None:
        return sq.mean(-1)
    k = int(topk_channels)
    if not 0 < k <= sq.shape[-1]:
        raise RuntimeError(f'invalid topk_channels {k} for {sq.shape[-1]} channels')
    if k == sq.shape[-1]:
        # identical arithmetic to the plain mean; do not route through topk, whose
        # summation order differs and would break the bitwise r2 equivalence claim.
        return sq.mean(-1)
    return torch.topk(sq, k, dim=-1).values.mean(-1)


def _pool_scores(scores, pool, grid):
    GH, GW = grid
    NVALID = GH * GW
    """Pool a (1,192) decision score over the 12x16 valid-patch grid."""
    if pool == 'none':
        return scores
    if pool not in POOLS:
        raise RuntimeError(f'unknown pool {pool!r}')
    x = scores.reshape(1, 1, GH, GW)
    if pool == 'max3':
        return F.max_pool2d(x, 3, stride=1, padding=1).reshape(1, NVALID)
    if pool == 'mean3':
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1).reshape(1, NVALID)
    return F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='replicate'), 5, stride=1).reshape(1, NVALID)


@torch.inference_mode()
def patch_intervention_normres_v4(y, mu, ref, U, lam, sig2, thresholds, tau, rho,
                                  residual_mean, residual_scale,
                                  cluster_id=0, score_mean=None, score_scale=None,
                                  fusion_alpha=1.0, fusion_beta=0.0, fusion_gamma=0.0,
                                  topk_channels=None, pool='none', eps=1e-6,
                                  valid_start=0, grid=(16, 16)):
    NVALID = grid[0] * grid[1]
    D = y.shape[-1]
    stop = valid_start + NVALID
    if y.ndim != 3 or y.shape[0] != 1 or valid_start < 0 or stop > y.shape[1]:
        raise RuntimeError(f'unexpected feature shape {tuple(y.shape)}')
    if tuple(mu.shape) != (NVALID, D) or tuple(ref.shape) != (NVALID, D):
        raise RuntimeError(f'unexpected mu/ref shape {tuple(mu.shape)} {tuple(ref.shape)}')
    if tuple(residual_mean.shape) != (NVALID, D) or tuple(residual_scale.shape) != (NVALID, D):
        raise RuntimeError('unexpected normalized residual stats shape')
    if thresholds.ndim != 2 or thresholds.shape[1] != NVALID or not torch.isfinite(thresholds).all():
        raise RuntimeError(f'invalid per-patch thresholds {tuple(thresholds.shape)}')
    # r2's score is a non-negative squared residual, so r2 can demand thresholds > 0. The r3/r4
    # score is standardised and/or fused, and its clean q90 can legitimately sit slightly
    # below zero when most of the mass lies just under the mean with a short right tail.
    if (lam <= 0).any() or (residual_scale <= 0).any():
        raise RuntimeError('non-positive eigenvalue or residual scale')
    if sig2 <= 0 or tau <= 0 or not 0 < rho <= 1 or eps <= 0:
        raise RuntimeError('invalid sig2/tau/rho/eps')
    if pool not in POOLS:
        raise RuntimeError(f'unknown pool {pool!r}')
    nt = int(thresholds.shape[0])
    cid = int(cluster_id)
    if not 0 <= cid < nt:
        raise RuntimeError(f'state cluster id {cid} outside [0,{nt})')
    if score_mean is None:
        score_mean = torch.zeros(NVALID, NTERMS, device=y.device, dtype=torch.float32)
    if score_scale is None:
        score_scale = torch.ones(NVALID, NTERMS, device=y.device, dtype=torch.float32)
    if tuple(score_mean.shape) != (NVALID, NTERMS) or tuple(score_scale.shape) != (NVALID, NTERMS):
        raise RuntimeError(f'unexpected score stats shape {tuple(score_mean.shape)}')
    if (score_scale <= 0).any():
        raise RuntimeError('non-positive score scale')

    yv = y[:, valid_start:stop, :].float()
    d = yv - mu
    z = d @ U
    perp = d - z @ U.t()
    rs = residual_scale.clamp_min(eps)
    normalized = (perp - residual_mean) / rs

    # --- evidence terms, each (1,192) -------------------------------------------------
    t_resid = _agg_channels(normalized.square(), topk_channels)
    t_ref = ((yv - ref) / rs).square().mean(-1)
    t_bg = (d / rs).square().mean(-1)
    terms = torch.stack((t_resid, t_ref, t_bg), dim=-1)[0]          # (192,3)

    # --- B: state-conditioned standardisation (identity when conditioning is off) -----
    zt = (terms - score_mean) / score_scale.clamp_min(eps)          # (192,3)

    # --- C: score fusion (residual only when beta=gamma=0) ---------------------------
    scores = fusion_alpha * zt[:, 0]
    if fusion_beta != 0.0:
        scores = scores + fusion_beta * zt[:, 1]
    if fusion_gamma != 0.0:
        scores = scores + fusion_gamma * zt[:, 2]
    scores = scores[None]                                           # (1,192)

    # --- D: spatial pooling of the decision score ------------------------------------
    if pool != 'none':
        scores = _pool_scores(scores, pool, grid)

    mask = scores > thresholds[cid][None]

    # --- edit path: identical to r2/r3 ------------------------------------------------
    sc = (z * z / lam).sum(-1) + (perp * perp / sig2).sum(-1)
    a = -sc
    sd = a.std(1, keepdim=True, correction=0) + 1e-6
    w = torch.exp((a - a.max(1, keepdim=True).values) / (tau * sd))
    candidate_delta = ref[None] + w[..., None] * (yv - ref[None]) - yv
    zeros = torch.zeros_like(candidate_delta)
    delta = torch.where(mask[..., None], candidate_delta, zeros)
    n = delta.abs().sum((-1, -2), keepdim=True)
    den = d.abs().sum((-1, -2), keepdim=True)
    eta = torch.where(n > 1e-12, torch.minimum(torch.ones_like(n), rho * den / n.clamp_min(1e-12)), torch.zeros_like(n))
    out = y.clone()
    edited = yv + eta * delta
    out[:, valid_start:stop, :] = torch.where(mask[..., None], edited, yv).to(y.dtype)

    active = bool(mask.any())
    actual = out[:, valid_start:stop, :].float() - yv
    off = actual[~mask]
    meta = dict(
        gate_active=active,
        gate_raw_active=active,
        patch_active_count=int(mask.sum()),
        patch_total_count=int(mask.numel()),
        patch_mask=mask[0].tolist(),
        patch_scores=scores[0].tolist(),
        patch_residual_scores=t_resid[0].tolist(),
        patch_gate_scores=scores[0].tolist(),
        patch_terms=terms.tolist(),
        patch_terms_z=zt.tolist(),
        patch_thresholds=thresholds[cid].tolist(),
        patch_score_mode='patch_pca_residual_norm2',
        patch_state_cluster=cid,
        patch_score_terms=('resid', 'ref', 'bg'),
        patch_aggregation=('mean' if topk_channels is None else f'topk{int(topk_channels)}'),
        patch_pool=pool,
        fusion_alpha=float(fusion_alpha), fusion_beta=float(fusion_beta), fusion_gamma=float(fusion_gamma),
        off_patch_maxabs=float(off.abs().max()) if off.numel() else 0.0,
        padding_maxabs=(float(torch.cat((out[:, :valid_start, :] - y[:, :valid_start, :], out[:, stop:, :] - y[:, stop:, :]), 1).abs().max()) if y.shape[1] > NVALID else 0.0),
        eta=float(eta.max()),
        delta_l1=float(n.max()),
        denom_l1=float(den.max()),
        applied_delta_l1=float(actual.abs().sum()),
        budget_limit_l1=float(rho * den.max()),
        changed_patch_count=int((actual.abs().sum(-1) > 0).sum()),
        patch_anomaly_score_min=float(scores.min()),
        patch_anomaly_score_max=float(scores.max()),
    )
    return out, meta
