# PAGE

PAGE is a training-free visual intervention for frozen vision-language-action policies. It uses clean demonstrations to build a state-conditioned reference, detects patch-level deviations with a clean-calibrated gate, and edits only the gated visual features before the policy action head.

The public release contains the reusable operator, clean-data calibration utilities, a state-conditioned K-nearest reference helper, an example configuration, method notes, and a compact benchmark summary. Benchmark-specific launchers, checkpoints, rollout logs, private assets, and machine-specific paths stay outside this repository.

## Method

At each policy query, PAGE retrieves the eight clean demonstration states closest to the current proprioceptive state. Their same-camera patch features form the reference mean. A clean calibration set supplies the background basis, residual statistics, state clusters, and per-patch thresholds. Online scores use a 3×3 maximum-pooled residual gate. The default writeback uses the original K8 patch mean with unit edit strength on active patches.

The implementation keeps the policy weights frozen. It does not use randomized evaluation images or success labels during calibration.

## Layout

- `src/page/core.py` contains the patch scoring, pooling, gating, and writeback operator.
- `src/page/calibration.py` contains clean-data basis and threshold fitting helpers.
- `src/page/reference.py` contains state-conditioned K8 retrieval.
- `configs/example.yaml` shows the frozen PAGE parameters used by the released operator.
- `docs/method.md` records the implementation contract.
- `docs/results.md` records the benchmark summary and evaluation scope.

## Minimal use

```python
import numpy as np
from page import retrieve_k8

reference, row_ids = retrieve_k8(clean_states, clean_features, query_state, k=8)
```

The core operator accepts torch tensors and is intended to run on CUDA together with the host VLA serving stack. The host adapter is benchmark-specific and should supply the model's visual feature tensor, the clean assets, and the current state query.

## Environment

The released utilities require Python 3.10 or newer, NumPy, PyTorch, and SciPy-compatible linear algebra. Install a PyTorch build matching the target CUDA runtime for closed-loop use. The repository does not bundle benchmark datasets, model weights, or demonstration features.

## Citation

Please cite the PAGE paper when using the method. The paper draft and benchmark-specific evidence are maintained separately from this public code snapshot.
