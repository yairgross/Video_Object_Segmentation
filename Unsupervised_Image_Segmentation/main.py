import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.datasets as ds
from PIL import Image


# task 1

def kmeans(X, k):
    """
    :param X: numpy array of size (N, d) containing the samples
    :param k: the number of clusters
    :return: a column vector of length N, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned;
             an array containing the centroids
    """

    N, d = np.shape(X)
    centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
    clustering = np.zeros(N)
    oldc = np.ones(N)
    step = 0
    while not np.array_equal(clustering, oldc):
        # print(step)
        step += 1
        oldc = np.copy(clustering)
        sizes = np.zeros(k)
        sums = [np.zeros(d) for j in range(k)]
        for xi, i in zip(X, range(N)):
            min_dis = math.inf
            for centroid, j in zip(centroids, range(k)):
                dis = np.linalg.norm(xi - centroid)
                if dis < min_dis:
                    min_dis = dis
                    clustering[i] = j
            ind = int(clustering[i])
            sizes[ind] += 1
            sums[ind] += xi
        centroids = [sums[j] / sizes[j] for j in range(k)]
    return np.reshape(clustering, (clustering.shape[0], 1)), centroids


def dcdpmeans(x, l):
    """
        :param x: numpy array of size (N, d) containing the samples
        :param l: lambda - the distance parameter to identify need of initializing a new cluster
        :return: a column vector of length N, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned;
                 an array containing the centroids
        """
    k = 1
    N, d = np.shape(x)
    centroids = [np.sum(x, axis=0) / N]
    clustering = np.zeros(N)
    oldc = np.ones(N)
    step = 0
    # Testing convergence by comparing membership of examples to the clusters
    while not np.array_equal(oldc, clustering):
        # print(step)
        step += 1
        oldc = np.copy(clustering)
        jmax = -1
        dmax = -1
        for i in range(N):
            min_centroid_dis = math.inf
            for centroid, j in zip(centroids, range(k)):
                dis = np.linalg.norm(x[i] - centroid)
                if dis < min_centroid_dis:
                    min_centroid_dis = dis
                    clustering[i] = j
            if min_centroid_dis > dmax:
                jmax = i
                dmax = min_centroid_dis
        if dmax > l:
            k += 1
            centroids.append(x[jmax])
            clustering[jmax] = k - 1
        for m in range(k):
            sum = np.zeros(d)
            indecies = np.array(np.where(clustering == m))
            size = len(indecies[0])
            for index in indecies[0]:
                sum += x[index]
            centroids[m] = sum / size
    return np.reshape(clustering, (clustering.shape[0], 1)), centroids


# Task 2

def test_kmeans():
    # Generating data
    N = 1000
    x, y = ds.make_blobs(n_samples=N, n_features=2, centers=5, shuffle=True)
    ks = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    for k in ks:
        clustering, centroids = kmeans(x, k)

        title = f"Test of k means implementation with k={k}"
        plt.title(title)
        plt.scatter(x[:, 0], x[:, 1], c=clustering, s=1, cmap="gist_rainbow")
        plt.show()


def test_dcdpmeans():
    # Generating data
    N = 1000
    x, y = ds.make_blobs(n_samples=N, n_features=2, centers=5, shuffle=True)
    ls = [1, 2, 3, 5, 10, 20]
    for l in ls:
        clustering = dcdpmeans(x, l)

        title = f"Test of DC-DP means implementation with lambda={l}"
        plt.title(title)
        plt.scatter(x[:, 0], x[:, 1], c=clustering[0], s=1, cmap="gist_rainbow")
        plt.show()


# Task 3

def mandrill_kmeans():
    image = Image.open('mandrill.jpg')
    image.show()
    img_array = np.array(image)
    rows, cols, rgb = img_array.shape
    x = img_array.reshape(rows * cols, rgb)
    ks = [2, 3, 5, 10, 30, 50, 100]
    for k in ks:
        clusters, centroids = kmeans(x, k)
        # Coloring each pixel in img_array by their corresponding centroid
        new_img_array = np.copy(img_array)
        for i in range(rows):
            for j in range(cols):
                new_img_array[i][j] = centroids[int(clusters[i * cols + j])]
        new_img = Image.fromarray(new_img_array)
        new_img.show()


def mandrill_dcdpmeans():
    image = Image.open('mandrill.jpg')
    image.show()
    img_array = np.array(image)
    rows, cols, rgb = img_array.shape
    x = img_array.reshape(rows * cols, rgb)
    ls = [250, 200, 150, 100, 50]
    for l in ls:
        clustering, centroids = dcdpmeans(x, l)
        # Coloring each pixel in img_array by their corresponding centroid
        new_img_array = np.copy(img_array)
        for i in range(rows):
            for j in range(cols):
                new_img_array[i][j] = centroids[int(clustering[i * cols + j])]
        new_img = Image.fromarray(new_img_array)
        new_img.show()


if __name__ == '__main__':
    test_kmeans()
    # test_dcdpmeans()
    # mandrill_kmeans()
    #mandrill_dcdpmeans()
