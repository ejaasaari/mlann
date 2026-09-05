#define PY_SSIZE_T_CLEAN

#include <sys/stat.h>
#include <sys/types.h>

#include <Eigen/Dense>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "Python.h"
#include "numpy/arrayobject.h"
#include "craftml.h"
#include "rf-class-depth.h"
#include "rf-pca.h"
#include "rf-rp.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> UIntRowMatrix;

typedef struct {
  PyObject_HEAD MLANN *index;
  PyArrayObject *py_data;
  float *data;
  int n;
  int dim;
} mlannIndex;

static PyObject *MLANN_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  mlannIndex *self = reinterpret_cast<mlannIndex *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->index = NULL;
    self->data = NULL;
    self->py_data = NULL;
  }

  return reinterpret_cast<PyObject *>(self);
}

static int MLANN_init(mlannIndex *self, PyObject *args) {
  PyArrayObject *py_data;
  int n, dim;
  const char *index_type;

  if (!PyArg_ParseTuple(args, "O!iis", &PyArray_Type, &py_data, &n, &dim, &index_type)) return -1;

  if (n <= 0 || dim <= 0 || PyArray_NDIM(py_data) != 2 || PyArray_TYPE(py_data) != NPY_FLOAT32 ||
      !PyArray_ISCARRAY_RO(py_data) || !PyArray_ISNOTSWAPPED(py_data) ||
      PyArray_DIM(py_data, 0) != n || PyArray_DIM(py_data, 1) != dim) {
    PyErr_SetString(PyExc_ValueError, "Corpus must be an aligned contiguous float32 matrix of shape (n, dim)");
    return -1;
  }

  float *data = reinterpret_cast<float *>(PyArray_DATA(py_data));
  self->py_data = py_data;
  Py_XINCREF(self->py_data);

  self->n = n;
  self->dim = dim;

  if (strcmp(index_type, "RP") == 0)
    self->index = new RFRP(data, n, dim);
  else if (strcmp(index_type, "PCA") == 0)
    self->index = new RFPCA(data, n, dim);
  else if (strcmp(index_type, "CRAFTML") == 0 || strcmp(index_type, "CraftML") == 0)
    self->index = new CraftML(data, n, dim);
  else
    self->index = new RFClass(data, n, dim);

  return 0;
}

static PyObject *build(mlannIndex *self, PyObject *args) {
  PyArrayObject *train_data;
  int n_train, dim_train;

  PyArrayObject *knn_data;
  int n_knn, dim_knn;

  int n_trees, depth, b;
  float density;

  if (!PyArg_ParseTuple(args, "O!iiO!iiiifi", &PyArray_Type, &train_data, &n_train, &dim_train,
                        &PyArray_Type, &knn_data, &n_knn, &dim_knn, &n_trees, &depth, &density, &b))
    return NULL;

  Eigen::Map<const UIntRowMatrix> knn(reinterpret_cast<uint32_t *>(PyArray_DATA(knn_data)), n_knn,
                                      dim_knn);
  Eigen::Map<const RowMatrix> train(reinterpret_cast<float *>(PyArray_DATA(train_data)), n_train,
                                    dim_train);

  PyThreadState *_save = PyEval_SaveThread();
  try {
    self->index->grow(n_trees, depth, knn, train, density, b);
    PyEval_RestoreThread(_save);
  } catch (const std::exception &e) {
    PyEval_RestoreThread(_save);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

static void mlann_dealloc(mlannIndex *self) {
  if (self->data) {
    delete[] self->data;
    self->data = NULL;
  }

  if (self->index) {
    delete self->index;
    self->index = NULL;
  }

  Py_XDECREF(self->py_data);
  self->py_data = NULL;

  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *ann(mlannIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, dim, n, return_distances;
  Distance dist;
  float elect;

  if (!PyArg_ParseTuple(args, "O!ifii", &PyArray_Type, &v, &k, &elect, &dist, &return_distances))
    return NULL;

  float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));
      Py_BEGIN_ALLOW_THREADS;
      self->index->query(indata, k, elect, outdata, dist, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
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
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *distances_out = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < n; ++i) {
        const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
        const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
        self->index->query(indata + query_offset, k, elect, outdata + output_offset, dist,
                           distances_out + output_offset);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
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

static PyObject *exact_search(mlannIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, n, dim, return_distances;
  Distance dist;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &dist, &return_distances))
    return NULL;

  float *indata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));
      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_knn(indata, k, outdata, dist, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
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
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *distances_out = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < n; ++i) {
        const size_t query_offset = static_cast<size_t>(i) * static_cast<size_t>(dim);
        const size_t output_offset = static_cast<size_t>(i) * static_cast<size_t>(k);
        self->index->exact_knn(indata + query_offset, k, outdata + output_offset, dist,
                               distances_out + output_offset);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
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

static bool craft_array(PyArrayObject *array, int type, int ndim, int columns = -1) {
  if (PyArray_TYPE(array) != type || PyArray_NDIM(array) != ndim ||
      !PyArray_ISCARRAY_RO(array) || !PyArray_ISNOTSWAPPED(array) ||
      (columns >= 0 && PyArray_DIM(array, ndim - 1) != columns)) {
    PyErr_SetString(PyExc_ValueError, "Invalid array dtype, shape, alignment, or layout");
    return false;
  }
  return true;
}

static PyObject *build_craftml(mlannIndex *self, PyObject *args) {
  PyArrayObject *train, *knn;
  CraftMLOptions options;
  int distance;
  if (!PyArg_ParseTuple(args, "O!O!iiiiiiiiIi", &PyArray_Type, &train, &PyArray_Type, &knn,
                        &options.n_trees, &options.max_depth, &options.branching_factor,
                        &options.leaf_size, &options.label_dim, &options.feature_dim,
                        &options.iterations, &options.node_sample_size, &options.seed, &distance))
    return nullptr;
  auto *index = dynamic_cast<CraftML *>(self->index);
  if (!index) { PyErr_SetString(PyExc_TypeError, "Expected CraftML index"); return nullptr; }
  if (!craft_array(train, NPY_FLOAT32, 2, self->dim) || !craft_array(knn, NPY_UINT32, 2))
    return nullptr;
  options.distance = static_cast<Distance>(distance);
  PyThreadState *state = PyEval_SaveThread();
  try {
    index->build(Eigen::Map<const UIntRowMatrix>(static_cast<uint32_t *>(PyArray_DATA(knn)),
                                                PyArray_DIM(knn, 0), PyArray_DIM(knn, 1)),
                 Eigen::Map<const RowMatrix>(static_cast<float *>(PyArray_DATA(train)),
                                             PyArray_DIM(train, 0), PyArray_DIM(train, 1)), options);
  } catch (const std::exception &error) {
    PyEval_RestoreThread(state);
    PyErr_SetString(PyExc_ValueError, error.what());
    return nullptr;
  }
  PyEval_RestoreThread(state);
  Py_RETURN_NONE;
}

static PyObject *ann_craftml(mlannIndex *self, PyObject *args) {
  PyArrayObject *queries;
  int k, budget, distance, return_distances;
  float threshold;
  if (!PyArg_ParseTuple(args, "O!iifii", &PyArray_Type, &queries, &k, &budget, &threshold,
                        &distance, &return_distances)) return nullptr;
  auto *index = dynamic_cast<CraftML *>(self->index);
  if (!index) { PyErr_SetString(PyExc_TypeError, "Expected CraftML index"); return nullptr; }
  const int ndim = PyArray_NDIM(queries);
  if ((ndim != 1 && ndim != 2) || k <= 0 || k > self->n || (budget != -1 && budget < k)) {
    PyErr_SetString(PyExc_ValueError, "Invalid query shape, k, or candidate budget");
    return nullptr;
  }
  if (!craft_array(queries, NPY_FLOAT32, ndim, self->dim)) return nullptr;
  if (index->empty() || distance != index->distance() ||
      (budget == -1 && (!std::isfinite(threshold) || threshold < 0 || threshold > 1))) {
    PyErr_SetString(PyExc_ValueError, "Unbuilt index, mismatched metric, or invalid threshold");
    return nullptr;
  }
  const npy_intp n = ndim == 1 ? 1 : PyArray_DIM(queries, 0);
  npy_intp shape[2] = {n, k};
  PyObject *nearest = PyArray_SimpleNew(ndim, ndim == 1 ? shape + 1 : shape, NPY_INT);
  if (!nearest) return nullptr;
  PyObject *distances = return_distances ?
      PyArray_SimpleNew(ndim, ndim == 1 ? shape + 1 : shape, NPY_FLOAT32) : nullptr;
  if (return_distances && !distances) { Py_DECREF(nearest); return nullptr; }
  const float *input = static_cast<float *>(PyArray_DATA(queries));
  int *output = static_cast<int *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(nearest)));
  float *scores = distances ?
      static_cast<float *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(distances))) : nullptr;
  std::exception_ptr error;
  Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for if(n > 1)
#endif
  for (npy_intp i = 0; i < n; ++i) {
    try {
      index->search(input + i * self->dim, k, budget, threshold, output + i * k,
                     static_cast<Distance>(distance), scores ? scores + i * k : nullptr);
    } catch (...) {
#ifdef _OPENMP
#pragma omp critical(craftml_query_error)
#endif
      { if (!error) error = std::current_exception(); }
    }
  }
  Py_END_ALLOW_THREADS;
  if (error) {
    Py_DECREF(nearest);
    Py_XDECREF(distances);
    try { std::rethrow_exception(error); }
    catch (const std::exception &e) { PyErr_SetString(PyExc_ValueError, e.what()); }
    return nullptr;
  }
  if (distances) return Py_BuildValue("NN", nearest, distances);
  return nearest;
}

static PyObject *craftml_scores(mlannIndex *self, PyObject *args) {
  PyArrayObject *q;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &q)) return nullptr;
  auto *index = dynamic_cast<CraftML *>(self->index);
  if (!index) { PyErr_SetString(PyExc_TypeError, "Expected CraftML index"); return nullptr; }
  if (!craft_array(q, NPY_FLOAT32, 1, self->dim)) return nullptr;
  std::vector<CraftML::LabelScore> prediction;
  PyThreadState *state = PyEval_SaveThread();
  try { prediction = index->predict(static_cast<float *>(PyArray_DATA(q))); }
  catch (const std::exception &error) {
    PyEval_RestoreThread(state);
    PyErr_SetString(PyExc_ValueError, error.what());
    return nullptr;
  }
  PyEval_RestoreThread(state);
  npy_intp size = prediction.size();
  PyObject *ids = PyArray_SimpleNew(1, &size, NPY_UINT32);
  PyObject *scores = PyArray_SimpleNew(1, &size, NPY_FLOAT32);
  if (!ids || !scores) { Py_XDECREF(ids); Py_XDECREF(scores); return nullptr; }
  auto *id_data = static_cast<uint32_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ids)));
  auto *score_data = static_cast<float *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(scores)));
  for (size_t i = 0; i < prediction.size(); ++i) {
    id_data[i] = prediction[i].id;
    score_data[i] = prediction[i].score;
  }
  return Py_BuildValue("NN", ids, scores);
}

static PyMethodDef MLANNMethods[] = {
    {"build_craftml", (PyCFunction)build_craftml, METH_VARARGS, "Build a CraftML forest"},
    {"ann_craftml", (PyCFunction)ann_craftml, METH_VARARGS, "Search a CraftML forest"},
    {"craftml_scores", (PyCFunction)craftml_scores, METH_VARARGS, "Sparse corpus-ID probabilities"},
    {"ann", (PyCFunction)ann, METH_VARARGS, "Return approximate nearest neighbors"},
    {"exact_search", (PyCFunction)exact_search, METH_VARARGS, "Return exact nearest neighbors"},
    {"build", (PyCFunction)build, METH_VARARGS, "Build the index"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MLANNIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) "mlann.MLANNIndex", /* tp_name*/
    sizeof(mlannIndex),                                /* tp_basicsize*/
    0,                                                 /* tp_itemsize*/
    (destructor)mlann_dealloc,                         /* tp_dealloc*/
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
    (initproc)MLANN_init,                              /* tp_init */
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
  PyObject *m;
  if (PyType_Ready(&MLANNIndexType) < 0) return NULL;

  m = PyModule_Create(&moduledef);

  if (m == NULL) return NULL;

  import_array();

  Py_INCREF(&MLANNIndexType);
  PyModule_AddObject(m, "MLANNIndex", reinterpret_cast<PyObject *>(&MLANNIndexType));

  PyModule_AddIntConstant(m, "IP", IP);
  PyModule_AddIntConstant(m, "L2", L2);

  return m;
}
