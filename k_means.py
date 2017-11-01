import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self):
        data = np.loadtxt("realdata.txt")
        self.data = data
        self.l = np.shape(data)[1] - 1
        self.n = np.shape(data)[0]
        self.b = np.zeros(self.n)
        self.b.astype(int)
        self.color_map = {0: 'r', 1: 'b', 2: 'g', 3: 'k'}

        #plt.xlabel('Length')
        #plt.ylabel('Width')
        #plt.scatter(self.data[:, 1], self.data[:, 2], color=self.color_map[3])
        #plt.show()

    def init_centroids(self, m):
        self.m = int(m)
        data_min, data_max = np.min(self.data[:, 1:], axis=0), np.max(self.data[:, 1:], axis=0)
        self.centroids = np.random.random((m, self.l)) * (data_max - data_min) + data_min

    def assignment(self):
        for i in range(self.n):
            self.b[i] = np.argmin(np.sqrt(np.sum((self.data[i, 1:] - self.centroids)**2, axis=1)))

    def update(self):
        for i in range(self.m):
            self.centroids[i] = np.mean(self.data[self.b == i, 1:], axis=0)

    def cluster(self, m):
        self.init_centroids(m)
        cnt = 1
        while True:
            pre_centroids = self.centroids.copy()
            self.assignment()
            self.update()
            # End condition judgement
            if np.sum(np.abs(pre_centroids - self.centroids)) == 0:
                print('After {} iterations k means converges.'.format(cnt))

            plt.xlabel('Length')
            plt.ylabel('Width')
            handles = []
            s1 = plt.scatter(self.data[self.b == 0, 1], self.data[self.b == 0, 2],
                             color='r', label="Cluter1", marker='o')
            handles.append(s1)
            s2 = plt.scatter(self.data[self.b == 1, 1], self.data[self.b == 1, 2],
                             color='k', label="Cluter2", marker='^')
            handles.append(s2)
            if m == 3:
                s3 = plt.scatter(self.data[self.b == 2, 1], self.data[self.b == 2, 2],
                                 color='b', label="Cluter3", marker='+')
                handles.append(s3)
            plt.legend(handles=handles)
            plt.title('Iteration {}'.format(cnt))
            plt.show()
            cnt += 1


k_means = KMeans()
k_means.cluster(2)