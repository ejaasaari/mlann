import numpy as np
import mlannlib

IP = mlannlib.IP
L2 = mlannlib.L2


class MLANNIndex(object):
    """
    An MLANN index object
    """

    def __init__(self, data, index_type="SparsePCA"):
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
            if index_type.upper() == "CRAFTML":
                index_type = "CRAFTML"
            self.n_samples = n_samples
            self.index = mlannlib.MLANNIndex(data, n_samples, dim, index_type)
            self.dim = dim
            self.index_type = index_type

        self.built = False

    def _compute_density(self, density):
        if density == "auto":
            return 1.0 / np.sqrt(self.dim)
        if density is None:
            return 1
        if not (0 < density <= 1):
            raise ValueError("Density should be in (0, 1]")
        return density

    def _compute_n_subsample(self, n_subsample):
        if n_subsample is None:
            return 300
        if self.index_type not in ("RF", "PCA", "PLS"):
            raise ValueError("n_subsample is only supported by RF, PCA and PLS")
        if (not isinstance(n_subsample, (int, np.integer))
                or isinstance(n_subsample, (bool, np.bool_)) or n_subsample < 0):
            raise ValueError("n_subsample must be a non-negative integer")
        if self.index_type == "PCA" and n_subsample == 1:
            raise ValueError("PCA n_subsample must be 0 or at least 2; 0 uses all node rows")
        if self.index_type == "PLS" and n_subsample < 2:
            raise ValueError("PLS n_subsample must be at least 2")
        return n_subsample

    def build(
        self, train=None, knn=None, n_trees=None, depth=None, density="auto", b=1,
        top_variance_dims=5, unsupervised=False, *, n_subsample=None, branching_factor=10,
        leaf_size=32, label_dim=128, feature_dim=0, iterations=2,
        node_sample_size=1000, seed=None, dist=L2, n_clusters=None, subspace_dim=None,
    ):
        """
        Builds a normal MLANN index.
        :param unsupervised: Build KD/SparsePCA/PCA/RP directly on the constructor's corpus,
                             with one vote per point in each routed leaf. Omit train and knn;
                             b must be 1. Default False preserves supervised leaf votes.
        :param depth: The depth of the trees; should be in the set {1, 2, ..., floor(log2(n))}.
        :param n_trees: The number of trees used in the index.
        :param seed: CraftML random seed; None uses 42. IVF does not accept a seed.
        :param n_clusters: IVF clusters per member (required for IVF).
        :param subspace_dim: IVF block dimension, default dim / n_trees. Equal disjoint
                             blocks must cover all dimensions: n_trees * subspace_dim == dim.
                             IVF fits all corpus rows to Lloyd convergence; train/knn
                             supply cell neighbor probabilities under dist.
        :param density: Feature density for legacy methods; PLS uses all input dimensions.
        :param b: Minimum vote threshold for candidates to be included in the linear search phase.
        :param top_variance_dims: Number of highest-variance dimensions KD chooses among
                                  at each node, capped at dim; positive integer, default 5.
                                  KD ignores density.
        :param n_subsample: Per-node fitting/split-scoring row cap for RF, PCA and PLS.
                            None uses 300. RF/PCA accept 0 for all node rows; RF also
                            accepts 1. PCA/PLS otherwise require an integer >= 2.
                            Partitioning and leaf votes use all rows. Other indexes reject it.
        :return:
        """
        if self.built:
            raise RuntimeError("The index has already been built")

        n_subsample = self._compute_n_subsample(n_subsample)

        if self.index_type == "IVF":
            if seed is not None:
                raise ValueError("seed is not supported by IVF")
            if unsupervised or depth is not None or density != "auto" or b != 1:
                raise ValueError("IVF uses supervised cell probabilities and no depth/density/b")
            for name, value in (("n_trees", n_trees), ("n_clusters", n_clusters)):
                if (not isinstance(value, (int, np.integer))
                        or isinstance(value, (bool, np.bool_)) or value < 1):
                    raise ValueError(f"{name} must be a positive integer for IVF")
            if subspace_dim is None:
                subspace_dim = self.dim // n_trees
            if (not isinstance(subspace_dim, (int, np.integer))
                    or isinstance(subspace_dim, (bool, np.bool_)) or subspace_dim < 1
                    or n_trees * subspace_dim != self.dim):
                raise ValueError("IVF requires n_trees * subspace_dim == dimension")
            if n_clusters > self.n_samples:
                raise ValueError("n_clusters must be <= corpus size")
            train = self._distribution_features(train, matrix=True)
            knn = np.asarray(knn)
            if (knn.ndim != 2 or knn.shape[0] != train.shape[0] or knn.shape[1] == 0
                    or not np.issubdtype(knn.dtype, np.integer)
                    or np.any(knn < 0) or np.any(knn >= self.n_samples)):
                raise ValueError("knn must contain valid integer corpus IDs, one row per query")
            if dist not in (IP, L2):
                raise ValueError("dist must be IP or L2")
            self.index.build_ivf(
                train, np.ascontiguousarray(knn, dtype=np.uint32),
                n_trees, n_clusters, subspace_dim, dist,
            )
            self.dist = dist
            self.built = True
            return

        if n_clusters is not None or subspace_dim is not None:
            raise ValueError("n_clusters and subspace_dim are only supported by IVF")

        if self.index_type == "CRAFTML":
            if unsupervised:
                raise ValueError("unsupervised is only supported by KD, SparsePCA, PCA and RP")
            train = self._distribution_features(train, matrix=True)
            knn = np.asarray(knn)
            if (knn.ndim != 2 or knn.shape[0] != train.shape[0] or knn.shape[1] == 0
                    or not np.issubdtype(knn.dtype, np.integer)
                    or np.any(knn < 0) or np.any(knn >= self.n_samples)):
                raise ValueError("knn must contain valid integer corpus IDs, one row per query")
            if dist not in (IP, L2):
                raise ValueError("dist must be IP or L2")
            if seed is None:
                seed = 42
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

        if self.index_type == "KD" and (
            not isinstance(top_variance_dims, (int, np.integer)) or top_variance_dims < 1
        ):
            raise ValueError("top_variance_dims must be a positive integer")

        if n_trees is None or depth is None:
            raise TypeError("n_trees and depth are required")
        density = self._compute_density(density)
        if unsupervised:
            if self.index_type not in ("KD", "SparsePCA", "PCA", "RP"):
                raise ValueError("unsupervised is only supported by KD, SparsePCA, PCA and RP")
            if train is not None or knn is not None:
                raise ValueError("Omit train and knn when unsupervised=True; trees use the corpus")
            if b != 1:
                raise ValueError("b must be 1 when unsupervised=True; each leaf member gets one vote")
            self.index.build_unsupervised(n_trees, depth, density, top_variance_dims, n_subsample)
            self.built = True
            return
        if train is None or knn is None:
            raise ValueError("train and knn are required unless unsupervised=True")
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
            top_variance_dims,
            n_subsample,
        )
        self.built = True

    def ann(self, q, k, votes_required=None, dist=None, return_distances=False, *,
            candidate_budget=None):
        """
        Performs an approximate nearest neighbor query for a single query vector or multiple query vectors
        in parallel. The queries are given as a numpy vector or a numpy matrix where each row contains a query.
        :param candidate_budget: CraftML shortlist size (>= k), ranked by probability.
                                 Alternatively, votes_required selects probability > tau;
                                 fewer than k candidates triggers full-corpus exact search.
        :param q: The query object. Can be either a single query vector or a matrix with one query vector per row.
        :param k: The number of nearest neighbors to be returned.
        :param votes_required: Minimum vote threshold for exact reranking. For IVF, this
                               is an inclusive threshold on averaged cell probabilities in
                               [0, 1]. Only IDs with positive support are candidates; missing
                               results are padded with -1 when fewer than k qualify.
        :param return_distances: Whether the distances are also returned.
        :return: If return_distances is false, returns a vector or matrix of indices of the approximate
                 nearest neighbors in the original input data for the corresponding query. Otherwise,
                 returns a tuple where the first element contains the nearest neighbors and the second
                 element contains their distances to the query.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        if self.index_type == "IVF":
            q = self._distribution_features(q)
            if candidate_budget is not None:
                raise ValueError("IVF uses votes_required, not candidate_budget")
            if (not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_))
                    or not 1 <= k <= self.n_samples):
                raise ValueError("k must be an integer in [1, corpus size]")
            if (not isinstance(votes_required, (int, float, np.integer, np.floating))
                    or isinstance(votes_required, (bool, np.bool_))
                    or not np.isfinite(votes_required) or not 0 <= votes_required <= 1):
                raise ValueError("votes_required must be a finite number in [0, 1]")
            return self.index.ann(
                q, k, votes_required, self.dist if dist is None else dist, return_distances,
            )
        if self.index_type == "CRAFTML":
            q = self._distribution_features(q)
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

    def _distribution_features(self, q, matrix=False):
        q = np.asarray(q)
        if (q.ndim not in ((2,) if matrix else (1, 2)) or q.shape[-1] != self.dim
                or (matrix and not q.shape[0]) or q.dtype != np.float32
                or not np.isfinite(q).all()):
            raise ValueError("Features must be finite float32 vectors with the corpus dimension")
        return np.require(q, dtype=np.float32, requirements=["C", "A"])

    def candidate_scores(self, q):
        """Return sparse (corpus IDs, probabilities) for one IVF or CraftML query."""
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        if self.index_type == "IVF":
            return self.index.ivf_scores(self._distribution_features(q))
        if self.index_type != "CRAFTML":
            raise ValueError("candidate_scores is available for CraftML and IVF")
        return self.index.craftml_scores(self._distribution_features(q))

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
        if self.index_type in ("CRAFTML", "IVF"):
            q = self._distribution_features(q)
            if not 1 <= k <= self.n_samples or dist not in (IP, L2):
                raise ValueError("Invalid k or metric")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        if k < 1:
            raise ValueError("k must be positive")

        return self.index.exact_search(q, k, dist, return_distances)
