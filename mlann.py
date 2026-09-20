from __future__ import annotations

import numpy as np
import mlannlib
from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from time import perf_counter

IP = mlannlib.IP
L2 = mlannlib.L2


@dataclass
class AutotuneConfiguration:
    n_trees: int
    depth: int
    votes_required: float
    tuning_recall: float
    latency_seconds: float | None = None
    index_bytes: int = 0
    estimated_latency_seconds: float | None = None
    estimated_candidates: float | None = None
    estimated_votes: float | None = None
    index_bytes_upper_bound: int = 0


@dataclass
class AutotuneResult:
    """A completed run installs an index; target_met reports recall separately."""

    status: str
    target_recall: float
    b: int
    selected: AutotuneConfiguration | None = None
    configurations: list = field(default_factory=list)
    shortlist: list = field(default_factory=list)
    tuning_seconds: float = 0.0
    query_threads: int = 1
    batch_size: int = 1
    stage_seconds: dict = field(default_factory=dict)
    query_counts: dict = field(default_factory=dict)
    target_met: bool = False
    fallback_reason: str | None = None

    @property
    def success(self):
        return self.selected is not None

    @property
    def n_trees(self):
        return None if self.selected is None else self.selected.n_trees

    @property
    def depth(self):
        return None if self.selected is None else self.selected.depth

    @property
    def votes_required(self):
        return None if self.selected is None else self.selected.votes_required

    @property
    def tuning_recall(self):
        return None if self.selected is None else self.selected.tuning_recall

    @property
    def latency_seconds(self):
        return None if self.selected is None else self.selected.latency_seconds

    @property
    def index_bytes(self):
        return None if self.selected is None else self.selected.index_bytes


def _recalls(found, truth):
    # A missing (-1) result is a miss; the denominator is always k.
    return np.array([np.count_nonzero(np.isin(g, f)) / len(g)
                     for f, g in zip(found, truth)], dtype=np.float64)


def _mean_recall(recalls, k):
    # Recover integer hits to avoid rejecting an exact target such as 9/10
    # merely because averaging per-query fractions rounded down by one ULP.
    return float(np.rint(recalls * k).sum() / (len(recalls) * k))


def _positive_integer(name, value):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def _autotune_split(n_rows, minimum_fit_rows, rng):
    """Reserve up to 1,024 calibration rows; use all other rows for fitting."""
    available = n_rows - minimum_fit_rows
    if available < 1:
        raise ValueError("training_queries needs at least one hold-out row in addition "
                         "to the rows required by depth_min")
    tuning_count = min(1024, available, max(1, n_rows // 5)) if minimum_fit_rows else min(1024, available)
    held_out = rng.choice(n_rows, size=tuning_count, replace=False)
    if minimum_fit_rows:
        fit_mask = np.ones(n_rows, dtype=bool)
        fit_mask[held_out] = False
        fitting = np.flatnonzero(fit_mask)
    else:
        fitting = np.empty(0, dtype=np.intp)
    return fitting, held_out


def _autotune_neighbors(truth, rng):
    """Spread a bounded scoring budget across queries, not correlated neighbors.

    Uniform sampling within each exact top-k set estimates recall@k without
    favoring easier neighbor ranks. Small calibration sets retain every neighbor.
    """
    rows, k = truth.shape
    width = min(k, max(1, 25600 // rows))
    if width == k:
        return truth
    sampled = np.empty((rows, width), dtype=np.uint32)
    for q in range(rows):
        sampled[q] = truth[q, rng.choice(k, width, replace=False)]
    return sampled


@dataclass(frozen=True)
class _AutotuneOptions:
    k: int
    target_recall: float | None
    n_trees_max: int
    depth_min: int
    depth_max: int
    density: str | float | None
    b: int
    unsupervised: bool
    dist: int
    memory_budget: int | None
    initial_batch_size: int
    batch_size: int
    timing_repeats: int
    cost_sample_size: int
    timing_sample_size: int


@dataclass(frozen=True)
class _AutotuneTraining:
    fitting: np.ndarray | None
    labels: np.ndarray | None
    calibration: np.ndarray
    depth_max: int
    query_counts: dict


@contextmanager
def _autotune_stage(stages, name):
    started = perf_counter()
    yield
    stages[name] = perf_counter() - started


def _estimate_autotune_costs(master, configurations, queries, sample, k, dist):
    """Return cost-annotated configurations without modifying the input records."""
    triples = np.asarray([(c.n_trees, c.depth, c.votes_required)
                          for c in configurations], dtype=np.float64)
    costs = master.index._estimate_costs(queries, triples, sample, k, dist)
    return [
        replace(config, estimated_votes=votes, estimated_candidates=candidates,
                estimated_latency_seconds=seconds, index_bytes_upper_bound=bound)
        for config, (votes, candidates, seconds, bound) in zip(configurations, costs)
    ]


def _measure_autotune_latency(deployed, queries, selected, options):
    """Time only the selected index, after one warmup batch."""
    count = min(options.timing_sample_size, len(queries))
    batches = [queries[i:min(i + options.batch_size, count)]
               for i in range(0, count, options.batch_size)]
    if options.batch_size == 1:
        batches = [q[0] for q in batches]
    deployed.ann(batches[0], options.k, selected.votes_required, options.dist, False)
    started = perf_counter()
    for _ in range(options.timing_repeats):
        for batch in batches:
            deployed.ann(batch, options.k, selected.votes_required, options.dist, False)
    return (perf_counter() - started) / (count * options.timing_repeats)


class AutotuneProfile:
    """Reusable maximum forest and empirical recall/cost frontier.

    subset() creates an independent index without tree fitting, exact search,
    calibration or cost estimation. close() frees the retained maximum forest;
    previously created subsets remain usable. Memory budgets apply to each
    deployed subset, not to this profile or to all simultaneously live subsets.
    """

    def __init__(self, master, configurations, timing_queries, options, query_counts, stages):
        master.index._enable_view_cache()
        self._master = master
        self._configurations = tuple(replace(c) for c in configurations)
        self._timing_queries = timing_queries[:options.timing_sample_size].copy()
        self._options = options
        self.query_counts = dict(query_counts)
        self.stage_seconds = dict(stages)
        self.tuning_seconds = 0.0

    def optimal_parameters(self):
        """Return independent records, ordered by increasing calibration recall."""
        return [replace(c) for c in self._configurations]

    @property
    def closed(self):
        return self._master is None

    def close(self):
        self._master = None

    def cache_info(self):
        """Report the active depth's shared payload bytes and cumulative trees built."""
        if self.closed:
            raise RuntimeError("The autotuning profile is closed")
        return self._master.index._view_cache_info()

    def clear_cache(self):
        """Release cached payloads without invalidating any existing subset."""
        if self.closed:
            raise RuntimeError("The autotuning profile is closed")
        self._master.index._clear_view_cache()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("The autotuning profile is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def subset(self, target_recall, *, measure_latency=False):
        """Select the fastest modeled configuration reaching target_recall.

        If unreachable, return the highest-recall configuration. The target is
        assessed on calibration queries only. Optional latency measurement never
        changes the configuration or threshold; by default no queries are run.
        """
        if self.closed:
            raise RuntimeError("The autotuning profile is closed")
        if not np.isfinite(target_recall) or not 0 < target_recall <= 1:
            raise ValueError("Require 0 < target_recall <= 1")
        started = perf_counter()
        feasible = [c for c in self._configurations if c.tuning_recall >= target_recall]
        chosen = min(feasible, key=lambda c: (c.estimated_latency_seconds, c.index_bytes_upper_bound)) \
            if feasible else self._configurations[-1]
        selected = replace(chosen)
        options = self._options
        stages = {}
        with _autotune_stage(stages, "materialization"):
            deployed = MLANNIndex._materialize_autotune(self._master, selected, options.memory_budget)
        if measure_latency:
            with _autotune_stage(stages, "selected_measurement"):
                selected.latency_seconds = _measure_autotune_latency(
                    deployed, self._timing_queries, selected, options)
        target_met = selected.tuning_recall >= target_recall
        result = AutotuneResult(
            status="observed" if target_met else "recall_below_target",
            target_recall=target_recall, b=options.b, selected=selected,
            configurations=[selected], shortlist=[selected],
            tuning_seconds=perf_counter() - started,
            query_threads=deployed._query_threads() if options.batch_size > 1 else 1,
            batch_size=options.batch_size, stage_seconds=stages,
            query_counts=dict(self.query_counts), target_met=target_met,
            fallback_reason=None if target_met else "frontier_target_unreachable",
        )
        # Native views share immutable payloads, own their routing projections,
        # and retain the corpus independently of the profile.
        index = MLANNIndex(None)
        index._data = self._master._data
        index.index_type = self._master.index_type
        index.n_samples, index.dim = self._master.n_samples, self._master.dim
        index.index, index.dist, index.votes_required = deployed, options.dist, selected.votes_required
        index.autotune_result, index.built = result, True
        return index


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
            self._data = data

        self.built = False

    def autotune(
        self, training_queries, knn=None, *,
        k, target_recall=None, n_trees_max, depth_max, depth_min=1,
        density="auto", b=1,
        unsupervised=False, dist=L2, memory_budget=None,
        initial_batch_size=16, batch_size=1, timing_repeats=1, random_state=0,
        cost_sample_size=4096, timing_sample_size=32,
    ):
        """Build one maximum forest; select and materialize one configuration.

        Supply representative training_queries and their corpus-neighbor labels.
        Up to 1,024 rows are reserved for calibration, excluded from supervised
        fitting. Unsupervised trees use the whole corpus and require no labels.
        random_state controls calibration/cost sampling, not tree construction.

        target_recall=None returns an AutotuneProfile retaining the maximum
        forest and full empirical recall/cost frontier. Call profile.subset(r)
        (or self.subset(r)) repeatedly without rebuilding or recalibrating.
        The profile itself is not a queryable index; close it to release memory.

        Recall is estimated from uniformly sampled exact top-k neighbors, with
        at most 25,600 query-neighbor pairs. Query cost is also sampled.
        There is no verification or post-selection threshold relaxation.
        target_met describes calibration recall, not a guarantee on unseen data.
        Recall misses still return a usable index with measured query latency.

        memory_budget bounds owned deployed storage, excluding the corpus,
        tuning workspace and query scratch. Invalid inputs, impossible budgets
        and resource failures raise. See README.md for sampling/timing details.
        """
        started = perf_counter()
        options = _AutotuneOptions(
            k=k, target_recall=target_recall, n_trees_max=n_trees_max,
            depth_min=depth_min, depth_max=depth_max, density=density, b=b,
            unsupervised=unsupervised, dist=dist, memory_budget=memory_budget,
            initial_batch_size=initial_batch_size, batch_size=batch_size,
            timing_repeats=timing_repeats, cost_sample_size=cost_sample_size,
            timing_sample_size=timing_sample_size,
        )
        stages = {}
        with _autotune_stage(stages, "validation"):
            queries, labels = self._validate_autotune_input(training_queries, knn, options)
            rng = np.random.default_rng(random_state)
            training = self._prepare_autotune_training(queries, labels, options, rng)
            tuning, query_counts = training.calibration, training.query_counts
            del queries, labels
        with _autotune_stage(stages, "structure_build"):
            master = self._build_autotune_master(training, options)
            del training  # Native construction has copied fitting rows/labels.
        if target_recall is None:
            profile = self._create_autotune_profile(master, tuning, options, rng, query_counts, stages)
            profile.tuning_seconds = perf_counter() - started
            self._autotune_profile = profile
            return profile
        with _autotune_stage(stages, "recall_calibration"):
            truth, configurations, fallback_reason = self._calibrate_autotune(master, tuning, options, rng)
        with _autotune_stage(stages, "cost_model"):
            selected, configurations, budget_reason, timing_queries = self._select_autotune(
                master, tuning, truth, configurations, options, rng)
            fallback_reason = budget_reason or fallback_reason
        query_threads = master.index._query_threads() if batch_size > 1 else 1
        with _autotune_stage(stages, "materialization"):
            deployed = self._materialize_autotune(master, selected, memory_budget)
            del master  # Release maximum-forest storage before timing queries.
        with _autotune_stage(stages, "selected_measurement"):
            selected.latency_seconds = _measure_autotune_latency(
                deployed, timing_queries, selected, options)

        target_met = selected.tuning_recall >= target_recall
        result = AutotuneResult(
            status="observed" if target_met else "recall_below_target",
            target_recall=target_recall, b=b, selected=selected,
            configurations=configurations, shortlist=[selected],
            tuning_seconds=perf_counter() - started, query_threads=query_threads,
            batch_size=batch_size, stage_seconds=stages, query_counts=query_counts,
            target_met=target_met, fallback_reason=fallback_reason,
        )
        # Commit only a fully constructed and timed index to the public object.
        self.index, self.dist, self.votes_required = deployed, dist, selected.votes_required
        self.autotune_result = result
        self.built = True
        return result

    def subset(self, target_recall, *, measure_latency=False):
        """Create an index from an autotune(target_recall=None) profile."""
        profile = getattr(self, "_autotune_profile", None)
        if profile is None:
            raise RuntimeError("Call autotune with target_recall=None before subsetting")
        return profile.subset(target_recall, measure_latency=measure_latency)

    def _create_autotune_profile(self, master, tuning, options, rng, query_counts, stages):
        queries = np.ascontiguousarray(tuning[rng.permutation(len(tuning))])
        sample = np.ascontiguousarray(rng.choice(
            self.n_samples, size=min(options.cost_sample_size, self.n_samples),
            replace=False), dtype=np.uint32)
        with _autotune_stage(stages, "exact_ground_truth"):
            truth = np.ascontiguousarray(master.exact_search(queries, options.k, options.dist), dtype=np.uint32)
            truth = _autotune_neighbors(truth, rng)
        with _autotune_stage(stages, "recall_cost_frontier"):
            entries = master.index._calibrate_frontier(
                queries, truth, options.depth_min, sample,
                min(options.initial_batch_size, len(queries)), options.dist,
                options.memory_budget or 0, options.k)
        if not entries:
            raise ValueError("memory_budget is too small for any conservative index storage bound")
        configurations = [AutotuneConfiguration(
            t, d, v, r, estimated_votes=votes, estimated_candidates=candidates,
            estimated_latency_seconds=seconds, index_bytes_upper_bound=bound)
            for t, d, v, r, votes, candidates, seconds, bound in entries]
        return AutotuneProfile(master, configurations, queries, options, query_counts, stages)

    def _validate_autotune_input(self, training_queries, knn, options):
        if self.built or getattr(self, "_autotune_profile", None) is not None:
            raise RuntimeError("The index has already been built")
        if self.index_type not in ("KD", "RP", "SparsePCA", "PCA", "RF", "PLS"):
            raise ValueError("Autotuning supports KD, RP, SparsePCA, PCA, RF and PLS only")
        for name in ("k", "n_trees_max", "depth_min", "depth_max", "b",
                     "initial_batch_size", "batch_size", "timing_repeats",
                     "cost_sample_size", "timing_sample_size"):
            _positive_integer(name, getattr(options, name))
        if options.k > self.n_samples or options.dist not in (IP, L2):
            raise ValueError("Invalid k or distance measure")
        if not np.isfinite(self._data).all():
            raise ValueError("Corpus must be finite")
        if options.target_recall is not None and (
                not np.isfinite(options.target_recall) or not 0 < options.target_recall <= 1):
            raise ValueError("Require 0 < target_recall <= 1")
        if options.memory_budget is not None:
            _positive_integer("memory_budget", options.memory_budget)
        training_queries = self._distribution_features(training_queries, matrix=True)
        if options.unsupervised:
            if self.index_type in ("RF", "PLS") or knn is not None or options.b != 1:
                raise ValueError("Unsupervised tuning requires KD/RP/PCA, no knn and b=1")
            rows = self.n_samples
        else:
            knn = np.asarray(knn)
            if (knn.ndim != 2 or knn.shape[0] != len(training_queries) or knn.shape[1] == 0
                    or not np.issubdtype(knn.dtype, np.integer)
                    or np.any(knn < 0) or np.any(knn >= self.n_samples)):
                raise ValueError("knn must contain valid integer corpus IDs, one row per query")
            for first in range(0, len(knn), 16384):
                sorted_labels = np.sort(knn[first:first + 16384], axis=1)
                if np.any(sorted_labels[:, 1:] == sorted_labels[:, :-1]):
                    raise ValueError("knn rows must have distinct labels")
            del sorted_labels
            knn = np.ascontiguousarray(knn, dtype=np.uint32)
            rows = len(training_queries)
        if not options.depth_min <= options.depth_max <= min(29, int(np.floor(np.log2(rows)))):
            raise ValueError("Require 1 <= depth_min <= depth_max <= min(29, floor(log2(training rows)))")
        # Validate build options before allocating the master.
        self._compute_density(options.density)
        return training_queries, knn

    def _prepare_autotune_training(self, queries, labels, options, rng):
        fitting_rows, tuning_rows = _autotune_split(
            len(queries), 0 if options.unsupervised else 1 << options.depth_min, rng)
        calibration = np.ascontiguousarray(queries[tuning_rows])
        if options.unsupervised:
            return _AutotuneTraining(
                None, None, calibration, options.depth_max,
                dict(fitting=self.n_samples, calibration=len(calibration)))
        return _AutotuneTraining(
            np.ascontiguousarray(queries[fitting_rows]),
            np.ascontiguousarray(labels[fitting_rows]), calibration,
            min(options.depth_max, len(fitting_rows).bit_length() - 1),
            dict(fitting=len(fitting_rows), calibration=len(calibration)))

    def _build_autotune_master(self, training, options):
        master = MLANNIndex(self._data, self.index_type)
        master.index._enable_tuning(True)
        master.build(
            training.fitting, training.labels, n_trees=options.n_trees_max,
            depth=training.depth_max, density=options.density, b=options.b,
            unsupervised=options.unsupervised)
        return master

    def _autotune_fallback(self, master, tuning, truth, trees, options):
        threshold = float(np.finfo(np.float32).tiny) if self.index_type in ("RF", "PLS") else 1.0
        recalls = master.index._predict_recall(
            tuning, truth, trees, options.depth_min, threshold)
        return AutotuneConfiguration(
            trees, options.depth_min, threshold, _mean_recall(recalls, truth.shape[1]))

    def _calibrate_autotune(self, master, tuning, options, rng):
        truth = np.ascontiguousarray(
            master.exact_search(tuning, options.k, options.dist), dtype=np.uint32)
        truth = _autotune_neighbors(truth, rng)
        configurations = [
            AutotuneConfiguration(*values)
            for values in master.index._calibrate(
                tuning, truth, options.depth_min, options.target_recall)
        ]
        if configurations:
            return truth, configurations, None
        fallback = self._autotune_fallback(master, tuning, truth, options.n_trees_max, options)
        return truth, [fallback], "calibration_target_unreachable"

    def _select_autotune(self, master, tuning, truth, configurations, options, rng):
        timing_queries = np.ascontiguousarray(tuning[rng.permutation(len(tuning))])
        cost_queries = timing_queries[:options.initial_batch_size]
        sample = np.ascontiguousarray(rng.choice(
            self.n_samples, size=min(options.cost_sample_size, self.n_samples),
            replace=False), dtype=np.uint32)
        configurations = _estimate_autotune_costs(
            master, configurations, cost_queries, sample, options.k, options.dist)
        feasible = [c for c in configurations if options.memory_budget is None or
                    c.index_bytes_upper_bound <= options.memory_budget]
        if feasible:
            selected = min(feasible, key=lambda c: (
                c.estimated_latency_seconds, c.index_bytes_upper_bound))
            return selected, configurations, None, timing_queries

        # Keep the same samples when costing a smaller, lower-recall fallback.
        compact = self._autotune_fallback(master, tuning, truth, 1, options)
        compact, = _estimate_autotune_costs(
            master, [compact], cost_queries, sample, options.k, options.dist)
        if compact.index_bytes_upper_bound > options.memory_budget:
            raise ValueError("memory_budget is too small for the conservative storage "
                             "bound of a single tree at depth_min")
        return (compact, configurations + [compact],
                "recall_configuration_exceeds_memory_budget", timing_queries)

    @staticmethod
    def _materialize_autotune(master, selected, memory_budget):
        deployed = master.index._make_view(selected.n_trees, selected.depth)
        selected.index_bytes = deployed._index_bytes()
        if memory_budget is not None and selected.index_bytes > memory_budget:
            raise RuntimeError("Materialized index exceeded its conservative size bound")
        return deployed

    def _compute_density(self, density):
        if density == "auto":
            return 1.0 / np.sqrt(self.dim)
        if density is None:
            return 1
        if not (0 < density <= 1):
            raise ValueError("Density should be in (0, 1]")
        return density

    def build(
        self, train=None, knn=None, n_trees=None, depth=None, density="auto", b=1,
        unsupervised=False, *, branching_factor=10,
        leaf_size=32, label_dim=128, feature_dim=0, iterations=2,
        node_sample_size=1000, seed=None, dist=L2,
    ):
        """
        Builds a normal MLANN index.
        :param unsupervised: Build KD/SparsePCA/PCA/RP directly on the constructor's corpus,
                             with one vote per point in each routed leaf. Omit train and knn;
                             b must be 1. Default False preserves supervised leaf votes.
        :param depth: The depth of the trees; should be in the set {1, 2, ..., floor(log2(n))}.
        :param n_trees: The number of trees used in the index.
        :param seed: CraftML random seed; None uses 42.
        :param density: Feature density; "auto" uses 1/sqrt(dim), None uses 1.
                        KD chooses among max(1, floor(density * dim)) highest-variance
                        dimensions per node; density=1 includes every dimension.
                        PLS uses all input dimensions.
        :param b: Minimum raw label count retained in a node (fixed build-time pruning).
        RF scores at most 400 sampled rows per node. PCA fits at most 100;
        PLS fits and scores at most 100. These caps are fixed. Smaller nodes
        use all their rows. Partitioning and leaf votes always use all rows.
        :return:
        """
        if self.built or getattr(self, "_autotune_profile", None) is not None:
            raise RuntimeError("The index has already been built")

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
            self.index.build_unsupervised(n_trees, depth, density)
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
        :param votes_required: Minimum vote threshold for exact reranking.
        :param return_distances: Whether the distances are also returned.
        :return: If return_distances is false, returns a vector or matrix of indices of the approximate
                 nearest neighbors in the original input data for the corresponding query. Otherwise,
                 returns a tuple where the first element contains the nearest neighbors and the second
                 element contains their distances to the query.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
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
            votes_required = getattr(self, "votes_required", None)
            if votes_required is None:
                raise ValueError("votes_required is required")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        return self.index.ann(q, k, votes_required, getattr(self, "dist", L2) if dist is None else dist, return_distances)

    def _distribution_features(self, q, matrix=False):
        q = np.asarray(q)
        if (q.ndim not in ((2,) if matrix else (1, 2)) or q.shape[-1] != self.dim
                or (matrix and not q.shape[0]) or q.dtype != np.float32
                or not np.isfinite(q).all()):
            raise ValueError("Features must be finite float32 vectors with the corpus dimension")
        return np.require(q, dtype=np.float32, requirements=["C", "A"])

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
            q = self._distribution_features(q)
            if not 1 <= k <= self.n_samples or dist not in (IP, L2):
                raise ValueError("Invalid k or metric")
        if q.dtype != np.float32:
            raise ValueError("The query matrix should have type float32")

        if k < 1:
            raise ValueError("k must be positive")

        return self.index.exact_search(q, k, dist, return_distances)
