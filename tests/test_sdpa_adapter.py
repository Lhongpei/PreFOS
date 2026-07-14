#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the sparse SDPA benchmark adapter."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_sdpa_prefos import make_problem, parse_sdpa_sparse


SMALL_SDPA = """\
2
2
{2, -2}
{1.5, -2.0}
0 1 1 1 -3
0 1 2 1 4
1 1 2 2 5
2 2 1 1 6
2 2 2 2 0
"""


class SparseSdpaAdapterTest(unittest.TestCase):
    def test_sdpa_dual_slack_maps_to_affine_cones(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "small.dat-s"
            filename.write_text(SMALL_SDPA, encoding="ascii")
            objective, sizes, blocks, offset, matrix, metadata = (
                parse_sdpa_sparse(filename)
            )

        np.testing.assert_allclose(objective, [-1.5, 2.0])
        self.assertEqual(sizes, [2, -2])
        self.assertEqual(blocks, [(3, 3, 2), (0, 2, 0)])
        np.testing.assert_allclose(offset, [3.0, -4.0 * math.sqrt(2.0), 0, 0, 0])
        np.testing.assert_allclose(
            matrix.toarray(), [[0, 0], [0, 0], [5, 0], [0, 6], [0, 0]]
        )
        self.assertEqual(metadata["source_entries"], 5)
        self.assertEqual(metadata["explicit_zeros"], 1)

        problem, keepalive = make_problem(objective, blocks, offset, matrix)
        self.assertEqual(problem.Q.rows, 2)
        self.assertEqual(problem.Q.cols, 2)
        self.assertEqual(problem.affine_cone_matrix.rows, 5)
        self.assertTrue(keepalive)


if __name__ == "__main__":
    unittest.main()
