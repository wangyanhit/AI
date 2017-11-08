import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.m = n_clusters
        self.max_iter = max_iter

    def init_centroids(self):
        data_min, data_max = np.min(self.data, axis=0), np.max(self.data, axis=0)
        self.centroids = np.random.random((self.m, self.l)) * (data_max - data_min) + data_min

    def assignment(self):
        for i in range(self.n):
            self.labels_[i] = np.argmin(np.sqrt(np.sum((self.data[i, :] - self.centroids)**2, axis=1)))

    def update(self):
        for i in range(self.m):
            if True in (self.labels_ == i):
                self.centroids[i] = np.mean(self.data[self.labels_ == i, :], axis=0)


    def fit(self, X):
        self.data = X
        self.l = np.shape(self.data)[1]
        self.n = np.shape(self.data)[0]
        self.init_centroids()
        self.labels_ = np.random.randint(self.m, size=self.n)
        cnt = 1
        while True:
            pre_centroids = self.centroids.copy()
            self.assignment()
            self.update()
            # End condition judgement
            if np.sum(np.abs(pre_centroids - self.centroids)) == 0:
                print('After {} iterations k means converges.'.format(cnt))
                break
            if cnt > self.max_iter:
                print('Reach max iteration number..')
                break
            cnt += 1