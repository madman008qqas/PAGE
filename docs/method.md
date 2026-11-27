# PAGE implementation contract

PAGE operates on a frozen VLA. The policy weights, action head, language input, and action decoding remain unchanged.

## Inputs

The operator receives a visual feature tensor with shape `(batch, patches, channels)`, a clean reference tensor with the same patch layout, a clean background mean, a rank-64 background basis, eigenvalues, residual mean and scale, cluster-specific score statistics, and cluster-specific thresholds.

State-conditioned retrieval uses the normalized proprioceptive state and the eight nearest clean training rows. Reference features are averaged at the same camera slot and patch position.

## Gate

For each patch, PAGE computes the normalized residual energy in the clean complement space. The gate score is the residual term after state-conditioned standardization. The released configuration applies 3×3 maximum pooling on the patch grid before threshold comparison. Thresholds are fitted from clean data at the 90th percentile and scaled by 1.50.

## Writeback

Let `a_i` be the binary gate and `m_i(q)` the K8 reference mean. The released no-budget writeback is

`y'_i = (1 - a_i) y_i + a_i m_i(q)`

The operator edits only active patches. Inactive patches pass through unchanged.

## Data boundary

Calibration uses clean demonstrations only. Randomized evaluation observations and success labels are excluded from asset fitting. The repository contains no private datasets or checkpoints.
