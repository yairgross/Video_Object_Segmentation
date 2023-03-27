import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as ds

def dcdpmeans()


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    d = X.shape[1]
    m = X.shape[0]
    u = [np.random.rand(1, d) for j in range(k)]
    c = np.zeros(m)

    for step in range(t):
        sizes = np.zeros(k)
        sums = [np.zeros(d) for j in range(k)]
        for xi, i in zip(X, range(m)):
            min_dis = math.inf
            for uj, j in zip(u, range(k)):
                dis = np.linalg.norm(xi - uj)
                if dis < min_dis:
                    min_dis = dis
                    c[i] = j
            ind = int(c[i])
            sizes[ind] += 1
            sums[ind] += xi
        u = [sums[j] / sizes[j] for j in range(k)]

    c = np.array([i + 1 for i in c])
    return np.reshape(c, (c.shape[0], 1))

def test_kmeans():
    # Generating data
    N = 1000
    x,y = ds.make_blobs(n_samples=N, n_features=2, centers=5, random_state=42)
    ks = [1, 50, 100, 200, 500, 1000]
    for k in ks:
        clusters = kmeans(x, k, 100)
        # Generating k random colors for each cluster
        title = f"Test of k means implementation with k={k}"
        plt.title(title)
        plt.scatter(x[:, 0], x[:, 1], c=clusters, s=1)
        plt.show()





if __name__ == '__main__':
    test_kmeans()
