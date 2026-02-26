import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski

data = pd.read_csv("ml_dataset.csv")

X = data[['alpha','beta','beta_alpha_ratio','delta','gamma','rms','theta','variance']].values
labels = data['label'].values

le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def dot(X, y):
    return [X[i] * y[i] for i in range(len(X))]

def norm(X, y):
    d = X - y
    return np.sqrt(np.sum((d[:-1] - d[1:]) ** 2))

def euclid(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_custom(Xt, yt, Xs, k):
    preds = []
    for x in Xs:
        d = [(euclid(x, Xt[i]), yt[i]) for i in range(len(Xt))]
        d.sort(key=lambda z: z[0])
        preds.append(Counter([l for _, l in d[:k]]).most_common(1)[0][0])
    return np.array(preds)

def knn_sklearn(Xt, yt, Xs, k):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(Xt, yt)
    return m.predict(Xs)

def matrix_inv(Xt, yt, Xs):
    Xt = np.c_[np.ones(Xt.shape[0]), Xt]
    Xs = np.c_[np.ones(Xs.shape[0]), Xs]
    w = np.linalg.inv(Xt.T @ Xt) @ Xt.T @ yt
    return np.round(Xs @ w).astype(int)

def metrics(y_true, y_pred, beta=1):
    cm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP) if TP+FP else 0
    rec = TP/(TP+FN) if TP+FN else 0
    f = (1+beta**2)*prec*rec/((beta**2)*prec+rec) if prec+rec else 0
    return cm, acc, prec, rec, f

def mean(X):
    return np.sum(X, axis=0)/X.shape[0]

def std(X):
    m = mean(X)
    return np.sqrt(np.sum((X-m)**2, axis=0)/X.shape[0])

def minkowski_vs_p(X, Y, pmax):
    return [(np.sum(np.abs(X-Y)**p))**(1/p) for p in range(1, pmax+1)]

def minkowski_compare(X, Y, pmax):
    return [( (np.sum(np.abs(X-Y)**p))**(1/p), minkowski(X,Y,p)) for p in range(1,pmax+1)]

k = 3
pred_c = knn_custom(X_train, y_train, X_test, k)
pred_s = knn_sklearn(X_train, y_train, X_test, k)
pred_m = matrix_inv(X_train, y_train, X_test)

mc = metrics(y_test, pred_c)
ms = metrics(y_test, pred_s)
mm = metrics(y_test, pred_m)

X1 = X[y==0]
X2 = X[y==1]

centroid_dist = np.linalg.norm(mean(X1)-mean(X2))

hist_mean = np.mean(data['alpha'].values)
hist_var = np.var(data['alpha'].values)

mink_manual = minkowski_vs_p(data['alpha'].values, data['beta'].values, 10)
mink_compare = minkowski_compare(data['alpha'].values, data['beta'].values, 10)

print(mc[1:], ms[1:], mm[1:])
print(centroid_dist, hist_mean, hist_var)
print(mink_manual)
print(mink_compare)