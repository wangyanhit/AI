import numpy as np
from collections import OrderedDict

data = []
with open('realdata1\chronic_kidney_disease_full.arff', 'rb') as f:
    lines = [x.strip() for x in f.readlines()]
    data_start = False
    for line in lines:
        if data_start:
            if line == b'':
                break
            data.append(line.decode('utf-8').replace('\t', '').replace(' ', '').split(','))
        if line == b'@data':
            data_start = True

numerical_features = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def get_raw_data(data, num_features):
    X = []
    y = []
    i = 0
    ct = OrderedDict({'age': [], 'bp': [], 'sg': [],
                            'al': [], 'su': [], 'rbc': [],
                            'pc': [], 'pcc': [], 'ba': [],
                            'bgr': [], 'bu': [], 'sc': [],
                            'sod': [], 'pot': [], 'hemo': [],
                            'pcv': [], 'wc': [], 'rc': [],
                            'htn': [], 'dm': [], 'cad': [],
                            'appet': [], 'pe': [], 'ane': []})
    for sample in data:
        j = 0
        sample_bias = 0
        sample_raw = []
        for c in ct:
            if j not in num_features:
                # fix wrong format
                while sample[j + sample_bias] is '':
                    sample_bias += 1
                if sample[j + sample_bias] not in ct[c] and sample[j + sample_bias] is not '?':
                    ct[c].append(sample[j + sample_bias])
                    sample_raw.append(sample[j + sample_bias])
                else:
                    sample_raw.append(sample[j + sample_bias])
            else:
                sample_raw.append(sample[j + sample_bias])
            j += 1
        X.append(sample_raw)
        y.append(sample[j + sample_bias])
        i += 1
    return X, y, ct


def complete_missing_data(X_raw, ct, num_features):
    # get the lens of category and initialize the means of every feature
    lens = OrderedDict({})
    means = OrderedDict({})
    num = OrderedDict({})

    for c in ct:
        # numerical
        if len(ct[c]) is 0:
            lens[c] = 1
            means[c] = 0
        # binary
        elif len(ct[c]) is 2:
            lens[c] = 1
            means[c] = 0
        # multi-class
        else:
            lens[c] = len(ct[c])
            means[c] = np.zeros(len(ct[c]))
        num[c] = 0
    i = 0
    for x in X_raw:
        j = 0
        for c in ct:
            # numerical feature
            if x[j] != '?':
                num[c] += 1
                if j in num_features:
                    means[c] += np.float32(x[j])
                    X_raw[i][j] = np.float32(x[j])
                # binary feature
                elif len(ct[c]) == 2:
                    X_raw[i][j] = ct[c].index(x[j])
                    means[c] += X_raw[i][j]
                # multi-class feature
                else:
                    tmp = np.zeros(lens[c])
                    tmp[ct[c].index(x[j])] = 1
                    means[c] += tmp
                    X_raw[i][j] = tmp
            j += 1
        i += 1
    for c in ct:
        means[c] /= num[c]

    # complete missing feature
    for i in range(len(X_raw)):
        j = 0
        for c in ct:
            if X_raw[i][j] is '?':
                X_raw[i][j] = means[c]
            j += 1
    # change into numpy array
    X_new = np.array([[]])
    for x in X_raw:
        x_new = np.array([])
        for j in range(24):
            if np.shape(x[j]) == ():
                x_new = np.concatenate((x_new, np.array([x[j]])), axis=0)
            else:
                x_new = np.concatenate((x_new, x[j]), axis=0)
        x_new = x_new.reshape((1, -1))
        if X_new.shape == (1, 0):
            X_new = x_new
        else:
            X_new = np.concatenate((X_new, x_new), axis=0)

    return X_new


def normalize_data(X):
    X -= np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X /= std
    return X


def get_data():
    X_raw, y_raw, category = get_raw_data(data, numerical_features)
    X = complete_missing_data(X_raw, category, numerical_features)
    X = normalize_data(X)

    y = np.array(y_raw) == 'notckd'
    y = y.astype(np.float32)
    return X, y


def split_into_train_test_set(X, y, per_train):
    n_train = np.int(y.shape[0] * per_train)
    p = np.random.permutation(y.shape[0])

    return X[p][:n_train], y[p][:n_train], X[p][n_train:], y[p][n_train:]
