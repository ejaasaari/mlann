import unittest
import numpy as np
import mlann


class PCACETests(unittest.TestCase):
    def test_factory_and_exact_candidates(self):
        rng = np.random.default_rng(87)
        corpus = rng.normal(size=(64, 8)).astype(np.float32)
        train = rng.normal(size=(32, 8)).astype(np.float32)
        labels = np.argsort(((train[:, None] - corpus[None])**2).sum(axis=2), axis=1)[:, :5].astype(np.uint32)
        for name in ['PCACE']:
            with self.subTest(index=name):
                a = mlann.MLANNIndex(corpus, name)
                a.build(train, labels, n_trees=3, depth=4)
                for metric in [mlann.L2, mlann.IP]:
                    ids, distances = a.ann(train, k=8, votes_required=1e-7, dist=metric, return_distances=True)
                    for q, found, values in zip(train, ids, distances):
                        valid = found >= 0
                        self.assertEqual(len(np.unique(found[valid])), valid.sum())
                        expected = np.linalg.norm(corpus[found[valid]] - q, axis=1) if metric == mlann.L2 else corpus[found[valid]] @ q
                        np.testing.assert_allclose(values[valid], expected, atol=2e-6)
                        if metric == mlann.L2:
                            self.assertTrue(np.all(np.diff(values[valid]) >= -1e-6))
                        else:
                            self.assertTrue(np.all(np.diff(values[valid]) <= 1e-6))

    def test_all_input_dimensions(self):
        # Signal exists only in the last coordinate; density cannot exclude it.
        data = np.zeros((32, 200), dtype=np.float32)
        data[:, -1] = np.arange(32) - 16
        labels = np.arange(32, dtype=np.uint32)[:, None]
        for name in ['PCACE']:
            outputs = []
            for density in ['auto', None, 0.005]:
                index = mlann.MLANNIndex(data, name)
                index.build(data, labels, n_trees=2, depth=1, density=density)
                result = index.ann(data, k=32, votes_required=1e-7)
                self.assertTrue(np.all((result >= 0).sum(axis=1) < 32))
                outputs.append(result)
            for result in outputs[1:]:
                np.testing.assert_array_equal(result, outputs[0])

    def test_reject_duplicate_labels(self):
        data = np.zeros((16, 4), dtype=np.float32)
        index = mlann.MLANNIndex(data, 'PCACE')
        with self.assertRaisesRegex(RuntimeError, 'distinct'):
            index.build(data, np.zeros((16, 2), dtype=np.uint32), n_trees=1, depth=2)


    def test_reproducible_build(self):
        rng = np.random.default_rng(18)
        data = rng.normal(size=(64, 17)).astype(np.float32)
        labels = np.argsort(((data[:, None] - data[None])**2).sum(axis=2), axis=1)[:, :5].astype(np.uint32)
        outputs = []
        for _ in range(2):
            index = mlann.MLANNIndex(data, 'PCACE')
            index.build(data, labels, n_trees=3, depth=4)
            outputs.append(index.ann(data, k=10, votes_required=1e-7))
        np.testing.assert_array_equal(*outputs)


if __name__ == '__main__':
    unittest.main()
