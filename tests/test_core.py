import unittest

import torch

from page import exemplar_minimum_score, patch_intervention_normres_v4


class TestCoreContract(unittest.TestCase):
    def test_exemplar_minimum_uses_reference_neighbourhood(self):
        grid = (5, 5)
        center = 2 * grid[1] + 2
        shifted = 2 * grid[1] + 3
        reference = torch.zeros(1, 25, 1)
        reference[:, center, 0] = 1.0
        scale = torch.ones(25, 1)

        translated = torch.zeros(25, 1)
        translated[shifted, 0] = 1.0
        translated_score = exemplar_minimum_score(
            translated, reference, scale, grid=grid, local_window=3
        )
        self.assertEqual(float(translated_score[shifted]), 0.0)

        distractor = torch.zeros(25, 1)
        distractor[center, 0] = 1.0
        distractor_score = exemplar_minimum_score(
            distractor, torch.zeros_like(reference), scale, grid=grid, local_window=3
        )
        self.assertGreater(float(distractor_score[center]), 0.5)

    def test_default_is_explicit_par_and_hard_replace_keeps_gate(self):
        torch.manual_seed(3)
        y = torch.randn(1, 16, 8)
        mu = torch.zeros(16, 8)
        ref = torch.randn(16, 8)
        bank = torch.randn(2, 16, 8)
        basis = torch.randn(8, 2)
        basis, _ = torch.linalg.qr(basis)
        lam = torch.ones(2)
        kwargs = dict(
            mu=mu, ref=ref, U=basis, lam=lam, sig2=1.0,
            thresholds=torch.full((1, 16), -1.0), tau=3.0,
            residual_mean=torch.zeros(8), residual_scale=torch.ones(8),
            grid=(4, 4), reference_bank=bank, local_window=3,
        )
        default, default_info = patch_intervention_normres_v4(y, **kwargs)
        par, par_info = patch_intervention_normres_v4(y, repair_mode="par", **kwargs)
        hard, hard_info = patch_intervention_normres_v4(y, repair_mode="hard_replace", **kwargs)
        self.assertTrue(torch.equal(default, par))
        self.assertEqual(default_info["patch_mask"], par_info["patch_mask"])
        self.assertEqual(par_info["patch_mask"], hard_info["patch_mask"])
        self.assertEqual(len(par_info["patch_mask"]), 16)
        self.assertTrue(torch.any((par - hard).abs() > 0))


if __name__ == "__main__":
    unittest.main()
