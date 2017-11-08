from time import time
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster.supervised import check_clusterings, contingency_matrix
from k_means import KMeans

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
# print(X.shape) (1797, 64)
# print(y.shape) #(1797, )
np.random.seed(0)

# Redefine cluster
def def_cluster(y, labels):
    labels_new = labels.copy()
    for i in range(10):
        counts = np.bincount(y[labels == i])
        if counts.shape[0] > 0:
            labels_new[labels == i] = np.argmax(counts)
    return labels_new

def fowlkes_mallows_score(labels_true, labels_pred, sparse=False):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples, = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return tk / np.sqrt(np.float32(pk) * qk) if tk != 0. else 0.

# K-means

print("K-means:")
clustering = KMeans(n_clusters=10, max_iter=500)
clustering.fit(X)
print(clustering.labels_[:40])
labels = def_cluster(y, clustering.labels_)
# onfusion matrix
cfs_mat = confusion_matrix(y, labels)
print("confusion matrix:")
print(cfs_mat)
# Fowlkes–Mallows index
fm_index = fowlkes_mallows_score(y, labels)
print("Fowlkes–Mallows index:")
print(fm_index)


# Agglomerative clustering with Ward linkage
print("Agglomerative clustering with Ward linkage:")
clustering = AgglomerativeClustering(linkage='ward', n_clusters=10)
clustering.fit(X)
labels = def_cluster(y, clustering.labels_)
# onfusion matrix
cfs_mat = confusion_matrix(y, labels)
print("confusion matrix:")
print(cfs_mat)
# Fowlkes–Mallows index
fm_index = fowlkes_mallows_score(y, labels)
print("Fowlkes–Mallows index:")
print(fm_index)

# AffinityPropagation
print("AffinityPropagation:")
af = AffinityPropagation(preference=-80000).fit(X)
labels = def_cluster(y, clustering.labels_)
# confusion matrix
cfs_mat = confusion_matrix(y, labels)
print("confusion matrix:")
print(cfs_mat)
print(cfs_mat.shape)
# Fowlkes–Mallows index
fm_index = fowlkes_mallows_score(y, labels)
print("Fowlkes–Mallows index:")
print(fm_index)
