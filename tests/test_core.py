import unittest

import torch

from page import patch_intervention_normres_v4


class CoreContractTest(unittest.TestCase):
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
