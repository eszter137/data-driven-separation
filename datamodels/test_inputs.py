#!/usr/bin/env python3
"""
Various tests for the utility functions in this project.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

May 2021
"""

import math
import unittest

import torch

import inputs


class InputmodelTests(unittest.TestCase):
    def test_ti_covariance(self):
        D = 500
        xi = 2
        pbc = True

        for p in [2]:
            cov = inputs.trans_inv_var(D, pbc, p=p, xi=xi)

            explicit = torch.zeros(D, D)
            for i in range(D):
                for j in range(D):
                    explicit[i, j] = math.exp(
                        -min(abs(i - j), D - abs(i - j)) ** p / xi**p
                    )
            explicit += 1e-3 * torch.eye(D)

            diff = torch.sum((explicit - cov) ** 2) / torch.sum(explicit ** 2)

            msg = ("Explicit translation-invariant var is incorrect for p=%d" % p,)
            self.assertTrue(diff.item() < 1e-4, msg)

    def test_erfgp_meanstd(self):
        """
        Making sure that the mean, standard deviation of the ErfGP is always (0, 1).
        """
        D = 500
        gains = [0.1, 1, 10]
        cov = inputs.trans_inv_var(D)
        P = 10000

        for gain in gains:
            erfGP = inputs.NLGP("erf", cov, gain=gain)

            xs = erfGP.sample(P)

            mean = torch.mean(xs).item()
            std = torch.std(xs).item()

            msg = "Mean incorrect for erfGP: gain=%g, mean=%g" % (gain, mean)
            self.assertTrue(abs(mean) < 1e-3, msg)
            msg = "Std incorrect for erfGP: gain=%g, std=%g" % (gain, std)
            self.assertTrue(abs(std - 1) < 1e-3, msg)


    def test_erfgp_covariance(self):
        """
        Make sure that the analytical covariance provided by NLGP and the empirical covariance
        are close for the erf function.
        """
        D = 50
        gains = [0.1, 1, 10]
        var = inputs.trans_inv_var(D)
        P = 20000

        for gain in gains:
            erfGP = inputs.NLGP("erf", var, gain=gain)

            xs = erfGP.sample(P)

            var_emp = xs.T @ xs / P

            diff = (torch.sum((var_emp - erfGP.variance())**2) /
                    torch.sum(var_emp**2)).item()
            msg = "variance incorrect for erfGP with gain=%g: diff=%g" % (gain, diff)
            self.assertTrue(diff < 1e-2, msg)


if __name__ == "__main__":
    unittest.main()
