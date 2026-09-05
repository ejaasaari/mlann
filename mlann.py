import numpy as np
import mlannlib

IP = mlannlib.IP
L2 = mlannlib.L2


class MLANNIndex(object):
    """
    An MLANN index object
    """

    def __init__(self, data, index_type="PCA"):
        """
        Initializes an MLANN index object.
        :param data: Input data either as a NxDim numpy ndarray or as a filepath to a binary file containing the data.
        :return:
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or 0 in data.shape:
                raise ValueError("The data matrix should be non-empty and two-dimensional")
            if data.dtype != np.float32:
                raise ValueError("The data matrix should have type float32")
            if not data.flags["C_CONTIGUOUS"] or not data.flags["ALIGNED"]:
                raise ValueError("The data matrix has to be C_CONTIGUOUS and ALIGNED")
            n_samples, dim = data.shape
        elif data is not None:
            raise ValueError("Data must be an ndarray")

        if data is not None:
            self.index_type = index_type.upper()
            self.index = mlannlib.MLANNIndex(data, n_samples, dim, self.index_type)
            self.dim = dim
            self.n_samples = n_samples

        self.built = False

    def _compute_density(self, density):
        if density == "auto":
            return 1.0 / np.sqrt(self.dim)
        if density is None:
            return 1
        if not (0 < density <= 1):
            raise ValueError("Density should be in (0, 1]")
        return density

    def build(self, train, knn, n_trees=None, depth=None, density="auto", b=1, *,
              branching_factor=10, leaf_size=32, label_dim=128, feature_dim=0,
              iterations=2, node_sample_size=1000, seed=42, dist=L2):
        """
        Build a forest from training queries and their neighbor IDs.
        CraftML uses knn.shape[1] labels per query (r), with no leaf pruning.
        Its dist selects cosine centroid routing for IP or Euclidean routing for L2.
        feature_dim=0 preserves features; positive values enable signed feature hashing.
        CraftML depth is a maximum, and leaf_size is a stopping target: unsplittable
        nodes may be larger. node_sample_size limits split fitting, not leaf statistics.
        :param depth: Maximum depth, 0..64 for CraftML; 1..floor(log2(n)) for other trees.
        :param n_trees: The number of trees used in the index.
        :param projection_sparsity: Expected ratio of non-zero components in a projection matrix.
        :param b: Minimum vote threshold for candidates to be included in the linear search phase.
        :return:
        """
        if self.built:
            raise RuntimeError("The index has already been built")

        if self.index_type == "CRAFTML":
            train = self._craft_features(train, matrix=True)
            knn = np.asarray(knn)
            if (knn.ndim != 2 or knn.shape[0] != train.shape[0] or knn.shape[1] == 0
                    or not np.issubdtype(knn.dtype, np.integer)
                    or np.any(knn < 0) or np.any(knn >= self.n_samples)):
                raise ValueError("knn must contain valid integer corpus IDs, one row per query")
            if dist not in (IP, L2):
                raise ValueError("dist must be IP or L2")
            if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= np.iinfo(np.uint32).max:
                raise ValueError("seed must be an integer in [0, 2**32 - 1]")
            if density != "auto" or b != 1:
                raise ValueError("CraftML uses feature_dim and unpruned leaves instead of density/b")
            self.index.build_craftml(
                train, np.ascontiguousarray(knn, dtype=np.uint32),
                10 if n_trees is None else n_trees, 20 if depth is None else depth,
                branching_factor, leaf_size, label_dim, feature_dim,
                iterations, node_sample_size, seed, dist,
            )
            self.dist = dist
            self.built = True
            return

        if n_trees is None or depth is None:
            raise TypeError("n_trees and depth are required for this index")
        density = self._compute_density(density)
        self.index.build(
            train,
            train.shape[0],
            train.shape[1],
            knn,
            knn.shape[0],
            knn.shape[1],
            n_trees,
            depth,
            density,
            b,
        )
        self.built = True

    def ann(self, q, k, votes_required=None, dist=None, return_distances=False, *,
            candidate_budget=None):
        """
        Performs an approximate nearest neighbor query for a single query vector or multiple query vectors
        in parallel. The queries are given as a numpy vector or a numpy matrix where each row contains a query.
        :param q: The query object. Can be either a single query vector or a matrix with one query vector per row.
        :param k: The number of nearest neighbors to be returned.
        :param votes_required: The number of votes an object has to get to be included in the linear search part of the query.
        :param return_distances: Whether the distances are also returned.
        :param candidate_budget: CraftML's maximum shortlist size (B >= k), selected by
                                 descending forest probability, with corpus ID breaking ties.
                                 Alternative: votes_required selects strict probability > tau.
                                 Fewer than k selected IDs triggers exact full-corpus search.
        :return: If return_distances is false, returns a vector or matrix of indices of the approximate
                 nearest neighbors in the original input data for the corresponding query. Otherwise,
                 returns a tuple where the first element contains the nearest neighbors and the second
                 element contains their distances to the query.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        if self.index_type == "CRAFTML":
            q = self._craft_features(q)
            if candidate_budget is not None and votes_required is not None:
                raise ValueError("Specify candidate_budget or votes_required, not both")
            if candidate_budget is None and votes_required is None:
                raise ValueError("Specify candidate_budget or votes_required")
            if candidate_budget is not None and (
                    not isinstance(candidate_budget, (int, np.integer)) or candidate_budget < k):
                raise ValueError("candidate_budget must be an integer >= k")
            return self.index.ann_craftml(
                q, k, -1 if candidate_budget is None else candidate_budget,
                0.0 if votes_required is None else votes_required,
                self.dist if dist is None else dist, return_distances,
            )
        if candidate_budget is not None:
            raise ValueError("candidate_budget is available for CraftML")
        if votes_required is None:
            raise ValueError("votes_required is required")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        return self.index.ann(q, k, votes_required, L2 if dist is None else dist, return_distances)

    def _craft_features(self, q, matrix=False):
        q = np.asarray(q)
        if (q.ndim not in ((2,) if matrix else (1, 2)) or q.shape[-1] != self.dim
                or (matrix and not q.shape[0]) or q.dtype != np.float32
                or not np.isfinite(q).all()):
            raise ValueError("Features must be finite float32 vectors with the corpus dimension")
        return np.require(q, dtype=np.float32, requirements=["C", "A"])

    def candidate_scores(self, q):
        """Return (corpus IDs, probabilities) for one CraftML query, before selection/fallback."""
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        if self.index_type != "CRAFTML":
            raise ValueError("candidate_scores is available for CraftML")
        return self.index.craftml_scores(self._craft_features(q))

    def exact_search(self, q, k, dist=mlannlib.L2, return_distances=False):
        """
        Performs an exact nearest neighbor query for a single query several queries in parallel. The queries are
        given as a numpy matrix where each row contains a query. Useful for measuring accuracy.
        :param q: The query object. Can be either a single query vector or a matrix with one query vector per row.
        :param k: The number of nearest neighbors to return.
        :param return_distances: Whether the distances are also returned.
        :return: If return_distances is false, returns a vector or matrix of indices of the exact
                 nearest neighbors in the original input data for the corresponding query. Otherwise,
                 returns a tuple where the first element contains the nearest neighbors and the second
                 element contains their distances to the query.
        """
        if self.index_type == "CRAFTML":
            q = self._craft_features(q)
            if not 1 <= k <= self.n_samples or dist not in (IP, L2):
                raise ValueError("Invalid k or metric")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        if k < 1:
            raise ValueError("k must be positive")

        return self.index.exact_search(q, k, dist, return_distances)
