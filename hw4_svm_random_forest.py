from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from hw4_get_data import get_data, split_into_train_test_set
from sklearn.metrics import f1_score

X, y = get_data()
X_train, y_train, X_test, y_test = split_into_train_test_set(X, y, 0.8)

# SVM with linear kernel
print('SVM with Linear kernel')
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f = f1_score(y_test, pred)
print('f-measure: {}'.format(f))

# SVM with RBF kernel
print('SVM with RBF kernel')
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f = f1_score(y_test, pred)
print('f-measure: {}'.format(f))

# Random forest
print('Random forest')
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f = f1_score(y_test, pred)
print('f-measure: {}'.format(f))