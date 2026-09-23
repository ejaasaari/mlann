#define PY_SSIZE_T_CLEAN

#include <Eigen/Dense>
#include <climits>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Python.h"
#include "index/craftml.h"
#include "index/kd.h"
#include "index/pca.h"
#include "index/rf.h"
#include "index/rp.h"
#include "numpy/arrayobject.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> UIntRowMatrix;

typedef struct {
    PyObject_HEAD MLANN* index;
    PyArrayObject* py_data;
    int n;
    int dim;
} mlannIndex;

static PyObject* MLANN_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    mlannIndex* self = reinterpret_cast<mlannIndex*>(type->tp_alloc(type, 0));

    if (self != NULL) {
        self->index = NULL;
        self->py_data = NULL;
    }

    return reinterpret_cast<PyObject*>(self);
}

static int MLANN_init(mlannIndex* self, PyObject* args) {
    PyArrayObject* py_data;
    int n, dim;
    const char* index_type;

    if (!PyArg_ParseTuple(args, "O!iis", &PyArray_Type, &py_data, &n, &dim, &index_type))
        return -1;

    if (n <= 0 || dim <= 0 || PyArray_NDIM(py_data) != 2 || PyArray_TYPE(py_data) != NPY_FLOAT32 ||
        !PyArray_ISCARRAY_RO(py_data) || !PyArray_ISNOTSWAPPED(py_data) ||
        PyArray_DIM(py_data, 0) != n || PyArray_DIM(py_data, 1) != dim) {
        PyErr_SetString(
            PyExc_ValueError,
            "Corpus must be an aligned contiguous float32 matrix of shape (n, dim)"
        );
        return -1;
    }

    float* data = reinterpret_cast<float*>(PyArray_DATA(py_data));
    self->py_data = py_data;
    Py_XINCREF(self->py_data);

    self->n = n;
    self->dim = dim;

    if (strcmp(index_type, "RP") == 0)
        self->index = new RP(data, n, dim);
    else if (strcmp(index_type, "CRAFTML") == 0 || strcmp(index_type, "CraftML") == 0)
        self->index = new CraftML(data, n, dim);
    else if (strcmp(index_type, "KD") == 0)
        self->index = new KD(data, n, dim);
    else if (strcmp(index_type, "PCA") == 0)
        self->index = new PCA(data, n, dim);
    else if (strcmp(index_type, "RF") == 0)
        self->index = new RF(data, n, dim);
    else {
        PyErr_Format(PyExc_ValueError, "Unrecognized index type '%s'.", index_type);
        return -1;
    }

    return 0;
}

static PyObject* build(mlannIndex* self, PyObject* args) {
    PyArrayObject* train_data;
    int n_train, dim_train;

    PyArrayObject* knn_data;
    int n_knn, dim_knn;

    int n_trees, depth, b;
    float density;

    if (!PyArg_ParseTuple(
            args,
            "O!iiO!iiiifi",
            &PyArray_Type,
            &train_data,
            &n_train,
            &dim_train,
            &PyArray_Type,
            &knn_data,
            &n_knn,
            &dim_knn,
            &n_trees,
            &depth,
            &density,
            &b
        ))
        return NULL;

    Eigen::Map<const UIntRowMatrix> knn(
        reinterpret_cast<uint32_t*>(PyArray_DATA(knn_data)), n_knn, dim_knn
    );
    Eigen::Map<const RowMatrix> train(
        reinterpret_cast<float*>(PyArray_DATA(train_data)), n_train, dim_train
    );

    PyThreadState* _save = PyEval_SaveThread();
    try {
        self->index->grow(n_trees, depth, knn, train, density, b);
        PyEval_RestoreThread(_save);
    } catch (const std::exception& e) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* build_unsupervised(mlannIndex* self, PyObject* args) {
    int n_trees, depth;
    float density;
    if (!PyArg_ParseTuple(args, "iif", &n_trees, &depth, &density))
        return NULL;

    PyThreadState* _save = PyEval_SaveThread();
    try {
        self->index->grow_unsupervised(n_trees, depth, density);
        PyEval_RestoreThread(_save);
    } catch (const std::exception& e) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static void mlann_dealloc(mlannIndex* self) {
    if (self->index) {
        delete self->index;
        self->index = NULL;
    }

    Py_XDECREF(self->py_data);
    self->py_data = NULL;

    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* ann(mlannIndex* self, PyObject* args) {
    PyArrayObject* v;
    int k, dim, n, return_distances;
    Distance dist;
    float elect;

    if (!PyArg_ParseTuple(args, "O!ifii", &PyArray_Type, &v, &k, &elect, &dist, &return_distances))
        return NULL;

    float* indata = reinterpret_cast<float*>(PyArray_DATA(v));
    PyObject* nearest;

    if (PyArray_NDIM(v) == 1) {
        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int* outdata = reinterpret_cast<int*>(PyArray_DATA((PyArrayObject*) nearest));

        if (return_distances) {
            PyObject* distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float* out_distances =
                reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*) distances));
            Py_BEGIN_ALLOW_THREADS;
            self->index->query(indata, k, elect, outdata, dist, out_distances);
            Py_END_ALLOW_THREADS;

            PyObject* out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            Py_BEGIN_ALLOW_THREADS;
            self->index->query(indata, k, elect, outdata, dist);
            Py_END_ALLOW_THREADS;
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int* outdata = reinterpret_cast<int*>(PyArray_DATA((PyArrayObject*) nearest));

        if (return_distances) {
            PyObject* distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float* distances_out =
                reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*) distances));

            Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < n; ++i) {
                const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
                const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
                self->index->query(
                    indata + query_offset,
                    k,
                    elect,
                    outdata + output_offset,
                    dist,
                    distances_out + output_offset
                );
            }
            Py_END_ALLOW_THREADS;

            PyObject* out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < n; ++i) {
                const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
                const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
                self->index->query(indata + query_offset, k, elect, outdata + output_offset, dist);
            }
            Py_END_ALLOW_THREADS;
            return nearest;
        }
    }
}

static PyObject* exact_search(mlannIndex* self, PyObject* args) {
    PyArrayObject* v;
    int k, n, dim, return_distances;
    Distance dist;

    if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &dist, &return_distances))
        return NULL;

    float* indata = reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*) v));
    PyObject* nearest;

    if (PyArray_NDIM(v) == 1) {
        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int* outdata = reinterpret_cast<int*>(PyArray_DATA((PyArrayObject*) nearest));

        if (return_distances) {
            PyObject* distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float* out_distances =
                reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*) distances));
            Py_BEGIN_ALLOW_THREADS;
            self->index->exact_knn(indata, k, outdata, dist, out_distances);
            Py_END_ALLOW_THREADS;

            PyObject* out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            Py_BEGIN_ALLOW_THREADS;
            self->index->exact_knn(indata, k, outdata, dist);
            Py_END_ALLOW_THREADS;
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int* outdata = reinterpret_cast<int*>(PyArray_DATA((PyArrayObject*) nearest));

        if (return_distances) {
            PyObject* distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float* distances_out =
                reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*) distances));

            Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < n; ++i) {
                const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
                const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
                self->index->exact_knn(
                    indata + query_offset,
                    k,
                    outdata + output_offset,
                    dist,
                    distances_out + output_offset
                );
            }
            Py_END_ALLOW_THREADS;

            PyObject* out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < n; ++i) {
                const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
                const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
                self->index->exact_knn(indata + query_offset, k, outdata + output_offset, dist);
            }
            Py_END_ALLOW_THREADS;
            return nearest;
        }
    }
}

static bool craft_array(PyArrayObject* array, int type, int ndim, int columns = -1) {
    if (PyArray_TYPE(array) != type || PyArray_NDIM(array) != ndim || !PyArray_ISCARRAY_RO(array) ||
        !PyArray_ISNOTSWAPPED(array) || (columns >= 0 && PyArray_DIM(array, ndim - 1) != columns)) {
        PyErr_SetString(PyExc_ValueError, "Invalid array dtype, shape, alignment, or layout");
        return false;
    }
    return true;
}

static PyObject* build_craftml(mlannIndex* self, PyObject* args) {
    PyArrayObject *train, *knn;
    CraftMLOptions options;
    int distance;
    if (!PyArg_ParseTuple(
            args,
            "O!O!iiiiiiii",
            &PyArray_Type,
            &train,
            &PyArray_Type,
            &knn,
            &options.n_trees,
            &options.max_depth,
            &options.branching_factor,
            &options.leaf_size,
            &options.label_dim,
            &options.feature_dim,
            &options.iterations,
            &distance
        ))
        return nullptr;
    auto* index = dynamic_cast<CraftML*>(self->index);
    if (!index) {
        PyErr_SetString(PyExc_TypeError, "Expected CraftML index");
        return nullptr;
    }
    if (!craft_array(train, NPY_FLOAT32, 2, self->dim) || !craft_array(knn, NPY_UINT32, 2))
        return nullptr;
    options.distance = static_cast<Distance>(distance);
    PyThreadState* state = PyEval_SaveThread();
    try {
        index->build(
            Eigen::Map<const UIntRowMatrix>(
                static_cast<uint32_t*>(PyArray_DATA(knn)), PyArray_DIM(knn, 0), PyArray_DIM(knn, 1)
            ),
            Eigen::Map<const RowMatrix>(
                static_cast<float*>(PyArray_DATA(train)),
                PyArray_DIM(train, 0),
                PyArray_DIM(train, 1)
            ),
            options
        );
    } catch (const std::exception& error) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    Py_RETURN_NONE;
}

static PyObject* ann_craftml(mlannIndex* self, PyObject* args) {
    PyArrayObject* queries;
    int k, distance, return_distances;
    int budget;
    float threshold;
    if (!PyArg_ParseTuple(
            args,
            "O!iifii",
            &PyArray_Type,
            &queries,
            &k,
            &budget,
            &threshold,
            &distance,
            &return_distances
        ))
        return nullptr;
    auto* index = dynamic_cast<CraftML*>(self->index);
    if (!index) {
        PyErr_SetString(PyExc_TypeError, "Expected CraftML index");
        return nullptr;
    }
    const int ndim = PyArray_NDIM(queries);
    if ((ndim != 1 && ndim != 2) || k <= 0 || k > self->n || (budget != -1 && budget < k)) {
        PyErr_SetString(PyExc_ValueError, "Invalid query shape, k, or candidate budget");
        return nullptr;
    }
    if (!craft_array(queries, NPY_FLOAT32, ndim, self->dim))
        return nullptr;
    if (index->empty() || distance != index->distance() ||
        (budget == -1 && (!std::isfinite(threshold) || threshold < 0 || threshold > 1))) {
        PyErr_SetString(PyExc_ValueError, "Unbuilt index, mismatched metric, or invalid threshold");
        return nullptr;
    }
    const npy_intp n = ndim == 1 ? 1 : PyArray_DIM(queries, 0);
    npy_intp shape[2] = {n, k};
    PyObject* nearest = PyArray_SimpleNew(ndim, ndim == 1 ? shape + 1 : shape, NPY_INT);
    if (!nearest)
        return nullptr;
    PyObject* distances = return_distances
                              ? PyArray_SimpleNew(ndim, ndim == 1 ? shape + 1 : shape, NPY_FLOAT32)
                              : nullptr;
    if (return_distances && !distances) {
        Py_DECREF(nearest);
        return nullptr;
    }
    const float* input = static_cast<float*>(PyArray_DATA(queries));
    int* output = static_cast<int*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(nearest)));
    float* scores =
        distances ? static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(distances)))
                  : nullptr;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for if (n > 1)
#endif
    for (npy_intp i = 0; i < n; ++i) {
        try {
            index->search(
                input + i * self->dim,
                k,
                budget,
                threshold,
                output + i * k,
                static_cast<Distance>(distance),
                scores ? scores + i * k : nullptr
            );
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(craftml_query_error)
#endif
            {
                if (!error)
                    error = std::current_exception();
            }
        }
    }
    Py_END_ALLOW_THREADS;
    if (error) {
        Py_DECREF(nearest);
        Py_XDECREF(distances);
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
        return nullptr;
    }
    if (distances)
        return Py_BuildValue("NN", nearest, distances);
    return nearest;
}

static PyObject* enable_tuning(mlannIndex* self, PyObject* args) {
    int structure_only = 0;
    if (!PyArg_ParseTuple(args, "|p", &structure_only))
        return nullptr;
    if (!(dynamic_cast<KD*>(self->index) || dynamic_cast<RP*>(self->index) ||
          dynamic_cast<PCA*>(self->index) || dynamic_cast<RF*>(self->index) ||
          dynamic_cast<CraftML*>(self->index))) {
        PyErr_SetString(PyExc_ValueError, "Autotuning supports KD, RP, PCA, RF and CRAFTML only");
        return nullptr;
    }
    try {
        self->index->enable_tuning(structure_only);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* tuning_view_impl(mlannIndex* self, PyObject* args, bool timing) {
    int trees, depth;
    if (!PyArg_ParseTuple(args, "ii", &trees, &depth))
        return nullptr;
    std::unique_ptr<MLANN> view;
    PyThreadState* state = PyEval_SaveThread();
    try {
        view = timing ? self->index->make_timing_view(trees, depth)
                      : self->index->make_view(trees, depth);
    } catch (const std::exception& e) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    auto* result = reinterpret_cast<mlannIndex*>(Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0));
    if (!result)
        return nullptr;
    result->index = view.release();
    result->py_data = self->py_data;
    Py_INCREF(result->py_data);
    result->n = self->n;
    result->dim = self->dim;
    return reinterpret_cast<PyObject*>(result);
}

static PyObject* tuning_view(mlannIndex* self, PyObject* args) {
    return tuning_view_impl(self, args, false);
}

static PyObject* timing_view(mlannIndex* self, PyObject* args) {
    return tuning_view_impl(self, args, true);
}

static PyObject* calibrate(mlannIndex* self, PyObject* args) {
    PyArrayObject *queries, *truth;
    int min_depth;
    double target;
    float fixed_threshold = 0;
    if (!PyArg_ParseTuple(
            args,
            "O!O!id|f",
            &PyArray_Type,
            &queries,
            &PyArray_Type,
            &truth,
            &min_depth,
            &target,
            &fixed_threshold
        ))
        return nullptr;
    if (!craft_array(queries, NPY_FLOAT32, 2, self->dim) || !craft_array(truth, NPY_UINT32, 2))
        return nullptr;
    std::vector<MLANN::Calibration> configurations;
    PyThreadState* state = PyEval_SaveThread();
    try {
        configurations = self->index->calibrate(
            Eigen::Map<const RowMatrix>(
                static_cast<float*>(PyArray_DATA(queries)), PyArray_DIM(queries, 0), self->dim
            ),
            Eigen::Map<const UIntRowMatrix>(
                static_cast<uint32_t*>(PyArray_DATA(truth)),
                PyArray_DIM(truth, 0),
                PyArray_DIM(truth, 1)
            ),
            min_depth,
            target,
            fixed_threshold
        );
    } catch (const std::exception& e) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    PyObject* result = PyList_New(configurations.size());
    if (!result)
        return nullptr;
    for (size_t i = 0; i < configurations.size(); ++i) {
        const auto& c = configurations[i];
        PyObject* item = Py_BuildValue("iifd", c.trees, c.depth, c.threshold, c.recall);
        if (!item) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* calibrate_frontier(mlannIndex* self, PyObject* args) {
    PyArrayObject *queries, *truth, *sample;
    int min_depth, cost_queries, dist;
    int query_k = 0;
    float fixed_threshold = 0;
    unsigned long long budget;
    if (!PyArg_ParseTuple(
            args,
            "O!O!iO!iiK|if",
            &PyArray_Type,
            &queries,
            &PyArray_Type,
            &truth,
            &min_depth,
            &PyArray_Type,
            &sample,
            &cost_queries,
            &dist,
            &budget,
            &query_k,
            &fixed_threshold
        ))
        return nullptr;
    if (!craft_array(queries, NPY_FLOAT32, 2, self->dim) || !craft_array(truth, NPY_UINT32, 2) ||
        !craft_array(sample, NPY_UINT32, 1))
        return nullptr;
    const auto* ids = static_cast<uint32_t*>(PyArray_DATA(sample));
    std::vector<uint32_t> samples(ids, ids + PyArray_SIZE(sample));
    std::vector<MLANN::FrontierConfiguration> frontier;
    PyThreadState* state = PyEval_SaveThread();
    try {
        frontier = self->index->calibrate_frontier(
            Eigen::Map<const RowMatrix>(
                static_cast<float*>(PyArray_DATA(queries)), PyArray_DIM(queries, 0), self->dim
            ),
            Eigen::Map<const UIntRowMatrix>(
                static_cast<uint32_t*>(PyArray_DATA(truth)),
                PyArray_DIM(truth, 0),
                PyArray_DIM(truth, 1)
            ),
            min_depth,
            samples,
            cost_queries,
            static_cast<Distance>(dist),
            size_t(budget),
            query_k,
            fixed_threshold
        );
    } catch (const std::exception& e) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    PyObject* result = PyList_New(frontier.size());
    if (!result)
        return nullptr;
    for (size_t i = 0; i < frontier.size(); ++i) {
        const auto& entry = frontier[i];
        const auto& c = entry.configuration;
        const auto& cost = entry.cost;
        PyObject* item = Py_BuildValue(
            "iifddddK",
            c.trees,
            c.depth,
            c.threshold,
            c.recall,
            cost.votes,
            cost.candidates,
            cost.seconds,
            static_cast<unsigned long long>(cost.bytes_upper_bound)
        );
        if (!item) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* index_bytes(mlannIndex* self, PyObject*) {
    return PyLong_FromSize_t(self->index->index_bytes());
}

static PyObject* enable_view_cache(mlannIndex* self, PyObject*) {
    try {
        self->index->enable_view_cache();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* clear_view_cache(mlannIndex* self, PyObject*) {
    PyThreadState* state = PyEval_SaveThread();
    self->index->clear_view_cache();
    PyEval_RestoreThread(state);
    Py_RETURN_NONE;
}

static PyObject* view_cache_info(mlannIndex* self, PyObject*) {
    const auto info = self->index->view_cache_info();
    return Py_BuildValue(
        "{s:i,s:i,s:K,s:K}",
        "depth",
        info.depth,
        "trees",
        info.trees,
        "payload_bytes",
        static_cast<unsigned long long>(info.bytes),
        "trees_built",
        static_cast<unsigned long long>(info.trees_built)
    );
}

static PyObject* estimate_costs(mlannIndex* self, PyObject* args) {
    PyArrayObject *queries, *configurations, *sample;
    int k, dist;
    if (!PyArg_ParseTuple(
            args,
            "O!O!O!ii",
            &PyArray_Type,
            &queries,
            &PyArray_Type,
            &configurations,
            &PyArray_Type,
            &sample,
            &k,
            &dist
        ))
        return nullptr;
    if (!craft_array(queries, NPY_FLOAT32, 2, self->dim) ||
        !craft_array(configurations, NPY_DOUBLE, 2, 3) || !craft_array(sample, NPY_UINT32, 1))
        return nullptr;
    std::vector<MLANN::Calibration> configs;
    const auto* data = static_cast<double*>(PyArray_DATA(configurations));
    for (npy_intp i = 0; i < PyArray_DIM(configurations, 0); ++i) {
        if (!std::isfinite(data[3 * i]) || !std::isfinite(data[3 * i + 1]) || data[3 * i] < 1 ||
            data[3 * i] > INT_MAX || data[3 * i + 1] < 1 ||
            data[3 * i + 1] > (dynamic_cast<CraftML*>(self->index) ? 64 : 29)) {
            PyErr_SetString(PyExc_ValueError, "Invalid cost configuration");
            return nullptr;
        }
        configs.push_back({int(data[3 * i]), int(data[3 * i + 1]), float(data[3 * i + 2]), 0.});
    }
    const auto* ids = static_cast<uint32_t*>(PyArray_DATA(sample));
    std::vector<uint32_t> samples(ids, ids + PyArray_SIZE(sample));
    std::vector<MLANN::CostEstimate> costs;
    PyThreadState* state = PyEval_SaveThread();
    try {
        costs = self->index->estimate_costs(
            Eigen::Map<const RowMatrix>(
                static_cast<float*>(PyArray_DATA(queries)), PyArray_DIM(queries, 0), self->dim
            ),
            configs,
            samples,
            k,
            static_cast<Distance>(dist)
        );
    } catch (const std::exception& e) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    PyObject* result = PyList_New(costs.size());
    if (!result)
        return nullptr;
    for (size_t i = 0; i < costs.size(); ++i) {
        const auto& c = costs[i];
        PyObject* item = Py_BuildValue(
            "dddK",
            c.votes,
            c.candidates,
            c.seconds,
            static_cast<unsigned long long>(c.bytes_upper_bound)
        );
        if (!item) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* predict_recall(mlannIndex* self, PyObject* args) {
    PyArrayObject *queries, *truth;
    int trees, depth;
    float threshold;
    if (!PyArg_ParseTuple(
            args,
            "O!O!iif",
            &PyArray_Type,
            &queries,
            &PyArray_Type,
            &truth,
            &trees,
            &depth,
            &threshold
        ))
        return nullptr;
    if (!craft_array(queries, NPY_FLOAT32, 2, self->dim) || !craft_array(truth, NPY_UINT32, 2))
        return nullptr;
    std::vector<double> recalls;
    PyThreadState* state = PyEval_SaveThread();
    try {
        recalls = self->index->predict_recall(
            Eigen::Map<const RowMatrix>(
                static_cast<float*>(PyArray_DATA(queries)), PyArray_DIM(queries, 0), self->dim
            ),
            Eigen::Map<const UIntRowMatrix>(
                static_cast<uint32_t*>(PyArray_DATA(truth)),
                PyArray_DIM(truth, 0),
                PyArray_DIM(truth, 1)
            ),
            trees,
            depth,
            threshold
        );
    } catch (const std::exception& e) {
        PyEval_RestoreThread(state);
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    PyEval_RestoreThread(state);
    npy_intp size = recalls.size();
    PyObject* result = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (result)
        std::copy(
            recalls.begin(),
            recalls.end(),
            static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)))
        );
    return result;
}

static PyObject* query_threads(mlannIndex*, PyObject*) {
#ifdef _OPENMP
    return PyLong_FromLong(omp_get_max_threads());
#else
    return PyLong_FromLong(1);
#endif
}

static PyMethodDef MLANNMethods[] = {
    {"_enable_tuning", (PyCFunction) enable_tuning, METH_VARARGS, "Retain builder memberships"},
    {"_estimate_costs",
     (PyCFunction) estimate_costs,
     METH_VARARGS,
     "Sample query work without materializing forests"},
    {"_make_timing_view", (PyCFunction) timing_view, METH_VARARGS, "Create a native timing view"},
    {"_make_view",
     (PyCFunction) tuning_view,
     METH_VARARGS,
     "Materialize a fixed forest prefix/depth"},
    {"_calibrate", (PyCFunction) calibrate, METH_VARARGS, "Calibrate positive thresholds"},
    {"_calibrate_frontier",
     (PyCFunction) calibrate_frontier,
     METH_VARARGS,
     "Calibrate reusable recall-cost frontier"},
    {"_predict_recall", (PyCFunction) predict_recall, METH_VARARGS, "Per-query candidate recall"},
    {"_index_bytes", (PyCFunction) index_bytes, METH_NOARGS, "Owned deployed index storage"},
    {"_enable_view_cache",
     (PyCFunction) enable_view_cache,
     METH_NOARGS,
     "Share immutable tuning payloads"},
    {"_clear_view_cache",
     (PyCFunction) clear_view_cache,
     METH_NOARGS,
     "Release strongly cached payloads"},
    {"_view_cache_info",
     (PyCFunction) view_cache_info,
     METH_NOARGS,
     "Tuning payload cache statistics"},
    {"_query_threads", (PyCFunction) query_threads, METH_NOARGS, "OpenMP query thread limit"},
    {"build_craftml", (PyCFunction) build_craftml, METH_VARARGS, "Build a CraftML forest"},
    {"ann_craftml", (PyCFunction) ann_craftml, METH_VARARGS, "Search a CraftML forest"},
    {"ann", (PyCFunction) ann, METH_VARARGS, "Return approximate nearest neighbors"},
    {"exact_search", (PyCFunction) exact_search, METH_VARARGS, "Return exact nearest neighbors"},
    {"build", (PyCFunction) build, METH_VARARGS, "Build the index"},
    {"build_unsupervised",
     (PyCFunction) build_unsupervised,
     METH_VARARGS,
     "Build trees on corpus points with unit leaf votes"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MLANNIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) "mlann.MLANNIndex", /* tp_name*/
    sizeof(mlannIndex),                                /* tp_basicsize*/
    0,                                                 /* tp_itemsize*/
    (destructor) mlann_dealloc,                        /* tp_dealloc*/
    0,                                                 /* tp_print*/
    0,                                                 /* tp_getattr*/
    0,                                                 /* tp_setattr*/
    0,                                                 /* tp_compare*/
    0,                                                 /* tp_repr*/
    0,                                                 /* tp_as_number*/
    0,                                                 /* tp_as_sequence*/
    0,                                                 /* tp_as_mapping*/
    0,                                                 /* tp_hash */
    0,                                                 /* tp_call*/
    0,                                                 /* tp_str*/
    0,                                                 /* tp_getattro*/
    0,                                                 /* tp_setattro*/
    0,                                                 /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "MLANN index object",                              /* tp_doc */
    0,                                                 /* tp_traverse */
    0,                                                 /* tp_clear */
    0,                                                 /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    0,                                                 /* tp_iter */
    0,                                                 /* tp_iternext */
    MLANNMethods,                                      /* tp_methods */
    0,                                                 /* tp_members */
    0,                                                 /* tp_getset */
    0,                                                 /* tp_base */
    0,                                                 /* tp_dict */
    0,                                                 /* tp_descr_get */
    0,                                                 /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    (initproc) MLANN_init,                             /* tp_init */
    0,                                                 /* tp_alloc */
    MLANN_new,                                         /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mlannlib",     /* m_name */
    "",             /* m_doc */
    -1,             /* m_size */
    module_methods, /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_mlannlib(void) {
    PyObject* m;
    if (PyType_Ready(&MLANNIndexType) < 0)
        return NULL;

    m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

    import_array();

    Py_INCREF(&MLANNIndexType);
    PyModule_AddObject(m, "MLANNIndex", reinterpret_cast<PyObject*>(&MLANNIndexType));

    PyModule_AddIntConstant(m, "IP", IP);
    PyModule_AddIntConstant(m, "L2", L2);

    return m;
}
