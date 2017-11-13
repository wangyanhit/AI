import  numpy as np


class LogisticRegression():
    def __init__(self, learning_rate=0.01, max_iter=200, lamb=0.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lamb = lamb

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape[0], X.shape[1] + 1
        one = np.ones((self.m, 1))
        self.w = np.random.normal(loc=0.0, scale=1.0, size=(self.n, 1))
        self.X = np.concatenate((one, X), axis=1)
        self.y = y.reshape((self.m, 1))
        for i in range(self.max_iter):
            z = self.X.dot(self.w)
            sig = self.sigmoid(z)
            d_sig_z = sig * (1 - sig)
            d_j_sig = -self.y / (sig + 0.0001) + (1 - self.y) / (1 - sig + 0.0001)
            d_z_w = self.X
            gradient = np.sum(d_j_sig * d_sig_z * d_z_w, axis=0) / self.m
            gradient = gradient.reshape((self.n, 1))
            gradient += self.lamb / self.m * self.w
            self.w -= self.learning_rate * gradient

    def predict(self, X):
        one = np.ones((X.shape[0], 1))
        one_X = np.concatenate((one, X), axis=1)
        pred = np.ravel(self.sigmoid(one_X.dot(self.w))) > 0.5
        pred = pred * 1.0
        return pred

