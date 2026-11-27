# PAGE implementation contract

PAGE operates on a frozen vision-language-action policy. Policy weights, language input, and action decoding stay unchanged.

## Clean-only calibration

The calibration set contains clean demonstrations only. We fit a rank-64 clean PCA subspace and retain the raw proprioceptive state for each feature frame. At a query state (`q_t`), PAGE retrieves the eight nearest clean states and their same-camera feature tensors.

For each clean calibration frame, thresholds are fitted with leave-one-out K=8 retrieval using strict same-position matching (`calib_local_window=1`). For patch `p`, the calibrated support score compares the query patch with the same-position patch in each retrieved exemplar, then applies 3×3 max pooling. The scalar α is the smallest 0.05-grid value whose clean query×camera×patch firing rate is at most 1.5%.

## Online gate and reference

At inference time, the query patch stays fixed while the gate searches the valid 9×9 neighbourhood in every K=8 reference frame:

\[
 u_{c,p}=\min_{j\in\mathcal N_8(q),\,u\in\mathcal V_9(p)}
 \operatorname{mean}_{\mathrm{chan}}
 \left(\frac{\operatorname{perp}(y_{c,p})-\operatorname{perp}(f_{j,c,u})}
 {\max(\mathrm{rs}_{c,p},10^{-6})}\right)^2.
\]

`V_9(p)` contains only in-bounds reference positions. The resulting per-patch score is 3×3 max-pooled and compared with the thresholds fitted by strict same-position calibration. The edit target does not move with the search window. It is always the original-position K=8 mean

\[
m_i(q_t)=\frac{1}{8}\sum_{j=1}^{8} f_{j,i}.
\]

## PAR writeback

The default repair mode is PAR. For an active patch, the reliability weight is computed from the clean PCA deviation and the feature is softly contracted toward the pose-matched mean

\[
\tilde y_i=m_i(q_t)+w_i(y_i-m_i(q_t)),\qquad
 y'_i=(1-a_i)y_i+a_i\tilde y_i.
\]

The released configuration is unbudgeted with unit edit strength η=1. An explicit `repair_mode="hard_replace"` is available only for controlled ablations and sets `w_i=0` while leaving the binary gate unchanged.

## Public API

`page.reference.retrieve_k8_bank` returns the K reference tensors, their original-position mean, and row IDs. `page.core.patch_intervention_normres_v4` accepts that bank through `reference_bank` and defaults to `repair_mode="par"`, `local_window=9`, and unbudgeted writeback. `page.calibration.calibrate_thresholds` performs strict same-position clean calibration. `CalibrationConfig.calib_local_window` is 1 for calibration, while `CalibrationConfig.online_local_window` documents the default online window of 9.

The repository contains no private datasets, model checkpoints, rollout logs, or benchmark-specific launchers.
