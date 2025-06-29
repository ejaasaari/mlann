import mlann
import numpy as np
from sklearn.datasets import fetch_openml  # scikit-learn is used only for loading the data

k = 10
training_k = 50
n_trees = 10
depth = 6
voting_threshold = 5
dist = mlann.IP  # or mlann.L2

# for RF index, the voting threshold should be a probability:
# voting_threshold = 0.000005

X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = np.ascontiguousarray(X, dtype=np.float32)

data = X[:30_000]
training_data = X[30_000:60_000]

q = X[-1]

index = mlann.MLANNIndex(data, "PCA")  # one of RP, PCA, or RF
knn = index.exact_search(training_data, training_k, dist=dist)

index.build(training_data, knn, n_trees, depth)

print('Exact:      ', index.exact_search(q, k, dist=dist))
print('Approximate:', index.ann(q, k, voting_threshold, dist=dist))
