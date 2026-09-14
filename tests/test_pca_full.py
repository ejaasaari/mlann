import unittest

import numpy as np

import mlann


class PCAFullTests(unittest.TestCase):
    def test_full_support_and_all_points_routed(self):
        # Only the last coordinate varies. Even a tiny density must retain it.
        # Exercise both sides of the sampling cap and odd/even median splits.
        for n in (32, 300, 301, 640):
            data = np.zeros((n, 64), dtype=np.float32)
            data[:, -1] = np.arange(n) - n / 2
            labels = np.arange(n, dtype=np.uint32)[:, None]
            for density in ('auto', None, 0.001):
                with self.subTest(n=n, density=density):
                    index = mlann.MLANNIndex(data, 'PCAFull')
                    index.build(data, labels, n_trees=1, depth=1, density=density)
                    found = index.ann(data, k=1, votes_required=1e-7)
                    np.testing.assert_array_equal(found[:, 0], np.arange(n))
                    # Median is computed over all rows, including unsampled rows.
                    candidates = index.ann(data[[0, -1]], k=n, votes_required=1e-7)
                    lower = np.sort(candidates[0][candidates[0] >= 0])
                    upper = np.sort(candidates[1][candidates[1] >= 0])
                    self.assertIn(len(lower), (n // 2, (n + 1) // 2))
                    np.testing.assert_array_equal(lower, np.arange(len(lower)))
                    np.testing.assert_array_equal(upper, np.arange(len(lower), n))

    def test_deep_tree_and_distances(self):
        rng = np.random.default_rng(9)
        data = rng.normal(size=(512, 12)).astype(np.float32)
        labels = np.arange(len(data), dtype=np.uint32)[:, None]
        for name in ('PCA', 'PCAFull'):
            index = mlann.MLANNIndex(data, name)
            index.build(data, labels, n_trees=2, depth=8)
            found, distances = index.ann(
                data, k=1, votes_required=1e-7, return_distances=True
            )
            np.testing.assert_array_equal(found[:, 0], np.arange(len(data)))
            np.testing.assert_allclose(distances, 0, atol=1e-6)

    def test_constant_points(self):
        data = np.zeros((320, 8), dtype=np.float32)
        labels = np.arange(len(data), dtype=np.uint32)[:, None]
        index = mlann.MLANNIndex(data, 'PCAFull')
        index.build(data, labels, n_trees=1, depth=8)
        found, distances = index.ann(data[:1], k=1, votes_required=1e-7, return_distances=True)
        self.assertGreaterEqual(found[0, 0], 0)
        self.assertEqual(distances[0, 0], 0)


if __name__ == '__main__':
    unittest.main()
