#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the scalar CBF benchmark adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cbf_scalar import load_scalar_cbf
from differential_conic_prefos import EXPONENTIAL, POWER


MIXED_MAP_CBF = """\
VER
3

POWCONES
1 2
2
1
1

OBJSENSE
MIN

VAR
1 1
F 1

CON
7 3
L= 1
EXP 3
@0:POW 3

OBJACOORD
1
0 2

OBJBCOORD
4

ACOORD
3
0 0 1
3 0 1
6 0 1

BCOORD
5
0 -2
1 10
2 1
4 4
5 1
"""


class ScalarCbfTest(unittest.TestCase):
    def test_affine_exp_and_power_domains_become_cone_slacks(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "mixed.cbf"
            filename.write_text(MIXED_MAP_CBF, encoding="ascii")
            model, metadata = load_scalar_cbf(filename)

        self.assertEqual(model.n, 7)
        self.assertEqual(model.A.shape, (7, 7))
        self.assertEqual(metadata["introduced_slack_variables"], 6)
        self.assertEqual(metadata["exponential_cones"], 1)
        self.assertEqual(metadata["power_cones"], 1)
        np.testing.assert_array_equal(model.box_indices, [0])
        self.assertTrue(np.isneginf(model.box_lower[0]))
        self.assertTrue(np.isposinf(model.box_upper[0]))
        np.testing.assert_allclose(model.lower, [2, 10, 1, 0, 4, 1, 0])
        np.testing.assert_allclose(model.upper, model.lower)
        np.testing.assert_allclose(model.c, [2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(model.offset, 4)

        exponential, power = model.cones
        self.assertEqual(exponential.cone_type, EXPONENTIAL)
        np.testing.assert_array_equal(exponential.indices, [3, 2, 1])
        self.assertEqual(power.cone_type, POWER)
        np.testing.assert_array_equal(power.indices, [4, 5, 6])
        self.assertEqual(power.power_alpha, 0.5)

        dense = model.A.toarray()
        np.testing.assert_allclose(dense[0], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(dense[3], [-1, 0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(dense[6], [-1, 0, 0, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
