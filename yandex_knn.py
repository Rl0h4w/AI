import numpy as np


class KNearestNeighbor:

    def __init__(self):
        self.y_train = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                diff = X[i, :] - self.X_train[j, :]
                squared_diff = diff ** 2
                sum_squared_diff = np.sum(squared_diff)
                dists[i, j] = np.sqrt(sum_squared_diff)
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            diff = self.X_train - X[i, :]
            squared_diff = diff ** 2
            sum_squared_diff = np.sum(squared_diff, axis=1)
            dists[i, :] = np.sqrt(sum_squared_diff)
        return dists

    def compute_distances_no_loops(self, X):
        dists = np.sqrt(
            -2 * np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:,
                                                                                       np.newaxis])
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
