# PAGE implementation contract

PAGE operates on a frozen vision-language-action policy. Policy weights, language input, and action decoding stay unchanged.

## Clean-only calibration

The calibration set contains clean demonstrations only. We fit a rank-64 clean PCA subspace and retain the raw proprioceptive state for each feature frame. At a query state (q_t), PAGE retrieves the eight nearest clean states and their same-camera feature tensors.

For each clean calibration frame, thresholds are fitted with leave-one-out K=8 retrieval. For patch (p), the exemplar-minimum score is the smallest normalized orthogonal residual over the K references and the valid 9×9 patch neighbourhood

\[
r_{p} = \min_{j\le K,\,u\in N_9(p)}
\left\|\frac{(I-UU^\top)(y_p-f_{j,u})}{\sigma_p}\right\|_2^2.
\]

The score is then 3×3 max-pooled. State clusters use (G=8). The scalar α is selected from the configured grid so the row-weighted clean firing rate is at most 1.5%, and the stored thresholds are α times the clean 90th percentile for each state cluster and patch.

## Online gate and reference

The gate uses the current feature tensor, the K=8 pose-matched clean references, the 9×9 exemplar-minimum score, and 3×3 max pooling. The edit target is always the original-position K=8 mean

\[
m_i(q_t)=\frac{1}{8}\sum_{j=1}^{8} f_{j,i}.
\]

The local 9×9 search changes the gate score only. It does not move the writeback target to a neighbouring patch.

## PAR writeback

The default repair mode is PAR. For an active patch, the reliability weight is computed from the clean PCA deviation and the feature is softly contracted toward the pose-matched mean

\[
\tilde y_i=m_i(q_t)+w_i(y_i-m_i(q_t)),\qquad
 y'_i=(1-a_i)y_i+a_i\tilde y_i.
\]

The released configuration is unbudgeted with unit edit strength η=1. An explicit `repair_mode="hard_replace"` is available only for controlled ablations and sets (w_i=0) while leaving the binary gate unchanged.

## Public API

`page.reference.retrieve_k8_bank` returns the K reference tensors, their original-position mean, and row IDs. `page.core.patch_intervention_normres_v4` accepts that bank through `reference_bank` and defaults to `repair_mode="par"`, `local_window=9`, and unbudgeted writeback. `page.calibration.calibrate_thresholds` performs the clean-only leave-one-out calibration described above.

The repository contains no private datasets, model checkpoints, rollout logs, or benchmark-specific launchers.
