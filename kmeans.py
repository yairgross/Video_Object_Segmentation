# This is a sample Python script.
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np


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



if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
