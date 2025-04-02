#define PY_SSIZE_T_CLEAN

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "Python.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <Eigen/Dense>

#include "numpy/arrayobject.h"
#include "rf-class-depth.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;

typedef struct {
  PyObject_HEAD MLANN *index;
  PyObject *py_data;
  float *data;
  bool mmap;
  int n;
  int dim;
  int k;
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

float *read_memory(char *file, int n, int dim) {
  FILE *fd;
  if ((fd = fopen(file, "rb")) == NULL) {
    return NULL;
  }

  float *data = new float[n * dim];

  if (data == NULL) {
    fclose(fd);
    return NULL;
  }

  int read = fread(data, sizeof(float), n * dim, fd);
  fclose(fd);

  if (read != n * dim) {
    delete[] data;
    return NULL;
  }

  return data;
}

#ifndef _WIN32
float *read_mmap(char *file, int n, int dim) {
  FILE *fd;
  if ((fd = fopen(file, "rb")) == NULL) return NULL;

  float *data;

  if ((data = reinterpret_cast<float *>(
#ifdef MAP_POPULATE
           mmap(0, n * dim * sizeof(float), PROT_READ, MAP_SHARED | MAP_POPULATE, fileno(fd),
                0))) == MAP_FAILED) {
#else
           mmap(0, n * dim * sizeof(float), PROT_READ, MAP_SHARED, fileno(fd), 0))) == MAP_FAILED) {
#endif
    return NULL;
  }

  fclose(fd);
  return data;
}
#endif

static int MLANN_init(mlannIndex *self, PyObject *args) {
  PyObject *py_data;
  int n, dim, mmap;

  if (!PyArg_ParseTuple(args, "Oiii", &py_data, &n, &dim, &mmap)) return -1;

  float *data;
  if (PyUnicode_Check(py_data)) {
    char *file = PyBytes_AsString(py_data);

    struct stat sb;
    if (stat(file, &sb) != 0) {
      PyErr_SetString(PyExc_IOError, strerror(errno));
      return -1;
    }

    if (sb.st_size != static_cast<unsigned>(sizeof(float) * dim * n)) {
      PyErr_SetString(PyExc_ValueError, "Size of the input is not N x dim");
      return -1;
    }

#ifndef _WIN32
    data = mmap ? read_mmap(file, n, dim) : read_memory(file, n, dim);
#else
    data = read_memory(file, n, dim);
#endif

    if (data == NULL) {
      PyErr_SetString(PyExc_IOError, "Unable to read data from file or allocate memory for it");
      return -1;
    }

    self->mmap = mmap;
    self->data = data;
  } else {
    data = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)py_data));
    self->py_data = py_data;
    Py_XINCREF(self->py_data);
  }

  self->n = n;
  self->dim = dim;
  self->index = new MLANN(data, dim, n);

  return 0;
}

static PyObject *build(mlannIndex *self, PyObject *args) {
  PyObject *train_data;
  int n_train, dim_train;

  PyObject *knn_data;
  int n_knn, dim_knn;

  int n_trees, depth;
  float density;

  if (!PyArg_ParseTuple(args, "OiiOiiiif", &train_data, &n_train, &dim_train, &knn_data, &n_knn,
                        &dim_knn, &n_trees, &depth, &density))
    return NULL;

  Eigen::Map<const Eigen::MatrixXi> knn(
      reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)knn_data)), n_knn, dim_knn);
  Eigen::Map<const Eigen::MatrixXf> train(
      reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)train_data)), n_train, dim_train);

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
#ifndef _WIN32
    if (self->mmap)
      munmap(self->data, self->n * self->dim * sizeof(float));
    else
#endif
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
  int k, elect, dim, n, return_distances;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &elect, &return_distances))
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
      self->index->query(indata, k, elect, outdata, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->query(indata, k, elect, outdata);
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
        self->index->query(indata + i * dim, k, elect, outdata + i * k, distances_out + i * k);
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
        self->index->query(indata + i * dim, k, elect, outdata + i * k);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

static PyObject *exact_search(mlannIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, n, dim, return_distances;

  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &v, &k, &return_distances)) return NULL;

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
      self->index->exact_knn(indata, k, outdata, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_knn(indata, k, outdata);
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
        self->index->exact_knn(indata + i * dim, k, outdata + i * k, distances_out + i * k);
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
        self->index->exact_knn(indata + i * dim, k, outdata + i * k);
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

  return m;
}
