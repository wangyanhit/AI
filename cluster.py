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
        labels_new[labels == i] = np.argmax(counts)
    return labels_new

#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    # normalize data into [0, 1]
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    color_handle = []
    for i in range(10):
        color_patch = mpatches.Patch(color=plt.cm.spectral(i / 10.), label='cluster-'+str(i))
        color_handle.append(color_patch)
    plt.legend(handles=color_handle)

    plt.xticks([])
    plt.yticks([])

    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
def fowlkes_mallows_score(labels_true, labels_pred, sparse=False):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples, = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return tk / np.sqrt(np.float32(pk) * qk) if tk != 0. else 0.

# 2D embedding of the digits dataset
print("Computing embedding")
# manifold.SpectralEmbedding -- Spectral embedding for non-linear dimensionality reduction
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")
# print(X_red.shape) (1797, 2)

# K-means


# Agglomerative clustering with Ward linkage
'''
print("Agglomerative clustering with Ward linkage:")
clustering = AgglomerativeClustering(linkage='ward', n_clusters=10)
clustering.fit(X_red)
labels = def_cluster(y, clustering.labels_)
plot_clustering(X_red, X, labels, 'Agglomerative clustering with Ward linkage')
plt.show()
# onfusion matrix
cfs_mat = confusion_matrix(y, labels)
print("confusion matrix:")
print(cfs_mat)
# Fowlkes–Mallows index
fm_index = fowlkes_mallows_score(y, labels)
print("Fowlkes–Mallows index:")
print(fm_index)
'''

# AffinityPropagation
print("AffinityPropagation:")
af = AffinityPropagation(preference=-50).fit(X_red)
labels = af.labels_ # def_cluster(y, clustering.labels_)
#plot_clustering(X_red, X, labels, 'AffinityPropagation')
#plt.show()
# confusion matrix
print(af.cluster_centers_indices_)
#print(y[:30], labels[:30])
cfs_mat = confusion_matrix(y, labels)
print("confusion matrix:")
print(cfs_mat)
print(cfs_mat.shape)
# Fowlkes–Mallows index
fm_index = fowlkes_mallows_score(y, labels)
print("Fowlkes–Mallows index:")
print(fm_index)