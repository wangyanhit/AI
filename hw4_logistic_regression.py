from hw4_get_data import get_data, split_into_train_test_set
from logistic_regression import LogisticRegression
from sklearn.metrics import f1_score


X, y = get_data()
X_train, y_train, X_test, y_test = split_into_train_test_set(X, y, 0.8)

clf = LogisticRegression(learning_rate=0.1, max_iter=20000, lamb=0.01)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
f = f1_score(y_test, pred)
print('f-measure: {}'.format(f))