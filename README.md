# MLANN

Approximate nearest neighbor search library implementing the [Multilabel Classification Framework](https://proceedings.neurips.cc/paper_files/paper/2022/file/e8752f3e51f33a2e06daf044c40ce412-Paper-Conference.pdf) (NeurIPS '22). This is a research library and will not offer state-of-the-art performance in most scenarios. However, it can be useful in extreme out-of-distribution (OOD) settings or in maximum inner product search (MIPS) where a small portion of queries have the highest inner products with most queries.

An [extended version](https://www.jmlr.org/papers/volume25/23-0286/23-0286.pdf) of the paper was published in Journal of Machine Learning Research (JMLR).

The original code used in the paper is available [here](https://github.com/vioshyvo/a-multilabel-classification-framework).

## Getting started

Install the Python module with `pip install git+https://github.com/ejaasaari/mlann`

> [!TIP]
> On macOS, it is recommended to use the Homebrew version of Clang as the compiler:

```shell script
brew install llvm libomp
CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install git+https://github.com/ejaasaari/mlann
```

An example for indexing and querying a dataset using MLANN is provided below:

```python
import mlann
import numpy as np
from sklearn.datasets import fetch_openml  # scikit-learn is used only for loading the data

k = 10
training_k = 50  # should be equal or larger to k
n_trees = 10  # increase for higher recall, slower search
depth = 6  # increase for lower recall, faster search
voting_threshold = 5  # increase for lower recall, faster search
dist = mlann.IP  # or mlann.L2

# for RF index, the voting threshold should be a probability:
# voting_threshold = 0.000005

X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = np.ascontiguousarray(X, dtype=np.float32)

data = X[:30_000]
training_data = X[30_000:60_000]

q = X[-1]

index = mlann.MLANNIndex(data, "PCA")  # one of RP, PCA, or RF
knn = index.exact_search(training_data, training_k, dist=dist)  # required for training

index.build(training_data, knn, n_trees, depth)

print('Exact:      ', index.exact_search(q, k, dist=dist))
print('Approximate:', index.ann(q, k, voting_threshold, dist=dist))
```

The following distances are available: `L2`, `IP`. Cosine distance can be used with `IP` by normalizing vectors.

The following index types are available:
- `RF`: random forest
- `RP`: random projection tree
- `PCA`: PCA tree

On most datasets, `RF` will likely provide the best query performance but can be slower to build. `RP` will likely be the fastest to build while offering the worst query performance, and `PCA` is a compromise between the two.

Building an MLANN index requires a training set of queries and their k nearest neighbors. If no separate training set is available, the database vectors can be used also as the training set. The k nearest neighbors can be computed e.g. by using

```index.exact_search(training_data, training_k, dist=dist)```

If this is too slow, the following can be tried:

- Sample a smaller training set
- Use a different approximate nearest neighbor library to search for approximate nearest neighbors instead
- If available, use a GPU to compute the nearest neighbors (with e.g. [cuVS](https://docs.rapids.ai/api/cuvs/nightly/python_api/neighbors_brute_force/))

## Citation

If you use the library in an academic context, please consider citing the following paper:

> Hyvönen, V., Jääsaari, E., and Roos, T. "A Multilabel Classification Framework for Approximate Nearest Neighbor Search." Advances in Neural Information Processing Systems 35 (2022): 35741-35754.

~~~~
@article{hyvonen2022multilabel,
  title={A Multilabel Classification Framework for Approximate Nearest Neighbor Search},
  author={Hyv{\"o}nen, Ville and J{\"a}{\"a}saari, Elias and Roos, Teemu},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35741--35754},
  year={2022}
}
~~~~

## License

MLANN is available under the MIT License (see [LICENSE](LICENSE)). Note that third-party libraries in the [cpp/lib](cpp/lib) folder may be distributed under other open source licenses (see [licenses](licenses)).
