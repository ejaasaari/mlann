#define PY_SSIZE_T_CLEAN

#include <sys/stat.h>
#include <sys/types.h>

#include <Eigen/Dense>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "Python.h"
#include "numpy/arrayobject.h"
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

  float *data = reinterpret_cast<float *>(PyArray_DATA(py_data));
  self->py_data = py_data;
  Py_XINCREF(self->py_data);

  self->n = n;
  self->dim = dim;

  if (strcmp(index_type, "RP") == 0)
    self->index = new RFRP(data, n, dim);
  else if (strcmp(index_type, "PCA") == 0)
    self->index = new RFPCA(data, n, dim);
  else
    self->index = new RFClass(data, n, dim);

  return 0;
}

static PyObject *build(mlannIndex *self, PyObject *args) {
  PyArrayObject *train_data;
  int n_train, dim_train;

  PyArrayObject *knn_data;
  int n_knn, dim_knn;

  int n_trees, depth;
  float density;

  if (!PyArg_ParseTuple(args, "O!iiO!iiiif", &PyArray_Type, &train_data, &n_train, &dim_train,
                        &PyArray_Type, &knn_data, &n_knn, &dim_knn, &n_trees, &depth, &density))
    return NULL;

  Eigen::Map<const UIntRowMatrix> knn(reinterpret_cast<uint32_t *>(PyArray_DATA(knn_data)), n_knn,
                                      dim_knn);
  Eigen::Map<const RowMatrix> train(reinterpret_cast<float *>(PyArray_DATA(train_data)), n_train,
                                    dim_train);

  try {
    Py_BEGIN_ALLOW_THREADS;
    self->index->grow(n_trees, depth, knn, train, density);
    Py_END_ALLOW_THREADS;

  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
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
        self->index->query(indata + i * dim, k, elect, outdata + i * k, dist, distances_out + i * k);
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
        self->index->query(indata + i * dim, k, elect, outdata + i * k, dist);
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

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &dist, &return_distances)) return NULL;

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
        self->index->exact_knn(indata + i * dim, k, outdata + i * k, dist, distances_out + i * k);
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
        self->index->exact_knn(indata + i * dim, k, outdata + i * k, dist);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

static PyMethodDef MLANNMethods[] = {
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
