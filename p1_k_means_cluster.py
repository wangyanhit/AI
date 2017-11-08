import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans

k_means = KMeans(2)
X = np.loadtxt("realdata.txt")[:, 1:]
k_means.fit(X)
labels = k_means.labels_

plt.xlabel('Length')
plt.ylabel('Width')
handles = []
s1 = plt.scatter(X[labels == 0, 0], X[labels == 0, 1],
                 color='r', label="Cluter1", marker='o')
handles.append(s1)
s2 = plt.scatter(X[labels == 1, 0], X[labels == 1, 1],
                 color='k', label="Cluter2", marker='^')
handles.append(s2)

plt.legend(handles=handles)
plt.title('K-means')
plt.show()