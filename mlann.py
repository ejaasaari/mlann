import numpy as np
from os import name as os_name

import mlannlib


class MLANNIndex(object):
    """
    An MLANN index object
    """

    def __init__(self, data, shape=None):
        """
        Initializes an MLANN index object.
        :param data: Input data either as a NxDim numpy ndarray or as a filepath to a binary file containing the data.
        :param shape: Shape of the data as a tuple (N, dim). Needs to be specified only if loading the data from a file.
        :return:
        """
        if isinstance(data, np.ndarray):
            if len(data) == 0 or len(data.shape) != 2:
                raise ValueError("The data matrix should be non-empty and two-dimensional")
            if data.dtype != np.float32:
                raise ValueError("The data matrix should have type float32")
            if not data.flags["C_CONTIGUOUS"] or not data.flags["ALIGNED"]:
                raise ValueError("The data matrix has to be C_CONTIGUOUS and ALIGNED")
            n_samples, dim = data.shape
        elif data is not None:
            raise ValueError("Data must be an ndarray")

        if data is not None:
            self.index = mlannlib.MLANNIndex(data, n_samples, dim)
            self.dim = dim

        self.built = False

    def _compute_density(self, density):
        if density == "auto":
            return 1.0 / np.sqrt(self.dim)
        if density is None:
            return 1
        if not (0 < density <= 1):
            raise ValueError("Density should be in (0, 1]")
        return density

    def build(self, train, knn, depth, n_trees, density="auto"):
        """
        Builds a normal MLANN index.
        :param depth: The depth of the trees; should be in the set {1, 2, ..., floor(log2(n))}.
        :param n_trees: The number of trees used in the index.
        :param projection_sparsity: Expected ratio of non-zero components in a projection matrix.
        :return:
        """
        if self.built:
            raise RuntimeError("The index has already been built")

        density = self._compute_density(density)
        self.index.build(train, train.shape[0], train.shape[1], knn, knn.shape[0], knn.shape[1], n_trees, depth, density)
        self.built = True

    def ann(self, q, k, votes_required, return_distances=False):
        """
        Performs an approximate nearest neighbor query for a single query vector or multiple query vectors
        in parallel. The queries are given as a numpy vector or a numpy matrix where each row contains a query.
        :param q: The query object. Can be either a single query vector or a matrix with one query vector per row.
        :param k: The number of nearest neighbors to be returned.
        :param votes_required: The number of votes an object has to get to be included in the linear search part of the query.
        :param return_distances: Whether the distances are also returned.
        :return: If return_distances is false, returns a vector or matrix of indices of the approximate
                 nearest neighbors in the original input data for the corresponding query. Otherwise,
                 returns a tuple where the first element contains the nearest neighbors and the second
                 element contains their distances to the query.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        return self.index.ann(q, k, votes_required, return_distances)

    def exact_search(self, q, k, return_distances=False):
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
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        if k < 1:
            raise ValueError("k must be positive")

        return self.index.exact_search(q, k, return_distances)
