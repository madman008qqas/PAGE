# PAGE

PAGE is a training-free visual intervention for frozen vision-language-action policies. It uses clean demonstrations to build a pose-conditioned reference, detects patch-level deviations with a clean-calibrated exemplar-minimum gate, and edits only gated visual features before the policy action head.

## Method

At each policy query, PAGE retrieves the eight clean demonstration states closest to the current proprioceptive state. For each patch, the gate searches the K=8 reference tensors in a 9×9 local neighbourhood, takes the minimum normalized residual, and applies 3×3 max pooling. Thresholds are fitted with leave-one-out clean calibration and an automatically selected α for a 1.5% clean firing target. Active patches use PAR soft contraction toward the original-position K8 mean, with unit edit strength in the released unbudgeted configuration.

The policy weights remain frozen. Randomized evaluation images and success labels are excluded from calibration.

## Layout

- `src/page/core.py` contains the exemplar-minimum gate and PAR or hard-replacement writeback.
- `src/page/calibration.py` contains clean-only PCA fitting and leave-one-out α calibration.
- `src/page/reference.py` contains state-conditioned K8 retrieval and reference-bank construction.
- `configs/example.yaml` shows the released parameters.
- `docs/method.md` records the implementation contract.
- `docs/results.md` records the benchmark summary and evaluation scope.

## Minimal use

```python
from page import retrieve_k8_bank, patch_intervention_normres_v4

bank, reference, row_ids = retrieve_k8_bank(clean_states, clean_features, query_state, k=8)
repaired, telemetry = patch_intervention_normres_v4(
    query_features, background_mean, reference, basis, eigenvalues, residual_variance,
    thresholds, tau=3.0, residual_mean=residual_mean, residual_scale=residual_scale,
    reference_bank=bank, local_window=9, repair_mode="par",
)
```

The operator accepts torch tensors and is intended to run on CUDA with a benchmark-specific VLA serving adapter. The host adapter supplies the visual feature tensor, clean assets, and current state query.

## Environment

The released utilities require Python 3.10 or newer, NumPy, and PyTorch. Install a PyTorch build matching the target CUDA runtime for closed-loop use. The repository does not bundle benchmark datasets, model weights, or demonstration features.

## Citation

Please cite the PAGE paper when using the method. The paper draft and benchmark-specific evidence are maintained separately from this public code snapshot.
