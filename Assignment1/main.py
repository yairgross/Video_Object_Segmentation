import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as ds

def converged(clusters):
    return False


def dcdpmeans(x, l, t):
    k = 1
    N, d = np.shape(x)
    centroids = [np.sum(x, axis=0) / N]
    z = np.ones(N)
    for step in range(t):
        print(step)
    #while not converged(): # TODO: implement
        sizes = np.zeros(k)
        sums = [np.zeros(d) for j in range(k)]
        jmax = -1
        dmax = -1
        for i in range(N):
            min_centroid_dis = math.inf
            for centroid, j in zip(centroids, range(k)):
                dis = np.linalg.norm(x[i] - centroid)
                if dis < min_centroid_dis:
                    min_dis = dis
                    z[i] = j
            if min_centroid_dis > dmax:
                jmax = i
                dmax = min_centroid_dis
        if dmax > l:
            k += 1
            centroids.append(x[jmax])
            z[jmax] = k-1
        for m in range(k):
            sum = np.zeros(d)
            indecies = np.array(np.where(z == m))
            size = len(indecies)
            for index in indecies[0]:
                sum += x[index]
            centroids[m] = sum / size
    return np.reshape(z, (z.shape[0], 1))


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


def test_dcdpmeans():
    # Generating data
    N = 1000
    x,y = ds.make_blobs(n_samples=N, n_features=2, centers=5, random_state=42)
    ls = [1, 2, 5, 10]
    for l in ls:
        clusters = dcdpmeans(x, l, 400)
        # Generating k random colors for each cluster
        title = f"Test of DC-DP means implementation with lambda={l}"
        plt.title(title)
        plt.scatter(x[:, 0], x[:, 1], c=clusters, s=1)
        plt.show()


if __name__ == '__main__':
    test_dcdpmeans()
