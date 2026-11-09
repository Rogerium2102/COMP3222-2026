import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lab6 import add_bias, plot_boundary


def sigmoid(z): return 1 / (1 + np.pow(np.e, -z))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)

ACTIVATIONS = { # name: (f, f’(a, z), targets for class 0/1, threshold)
    "sigmoid": (sigmoid, lambda a, z: a * (1 - a), (0, 1), 0.5),
    "tanh": (tanh, lambda a, z: 1 - a**2, (-1, 1), 0.0),
    "relu": (relu, lambda a, z: (z > 0).astype(float), (0, 1), 0.5),
}

def neuron_fit(X, t, act, eta=0.1, epochs=3000):
    f, df, (lo, hi), _ = ACTIVATIONS[act]
    target = np.where(t == 1, hi, lo) # targets in this activation’s range
    Xb = add_bias(X)
    w = np.random.default_rng(0).normal(0, 0.1, Xb.shape[1]) # small random start
    for epoch in range(epochs):
        z = Xb @ w
        a = f(z)
        w = w - eta * Xb.T @ ((a - target) * df(a, z)) / len(t)
    return w

def neuron_predict(w, X, act):
    f, _, _, threshold = ACTIVATIONS[act]
    return (f(add_bias(X) @ w) > threshold).astype(int)

rng = np.random.default_rng(0) # use this seed
C = np.array([[2, 1], [1, 2]]) # shared covariance
data0 = rng.multivariate_normal(mean=[0, 3], cov=C, size=100) # class 0
data1 = rng.multivariate_normal(mean=[3, 0], cov=C, size=100) # class 1
X2 = np.vstack([data0, data1])
t2 = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]

X_tr, X_va, t_tr, t_va = train_test_split(X2, t2, test_size=0.3,
    stratify=t2, random_state=0)

for name, active_func in ACTIVATIONS.items():
    print("FUNCTION: ", name)
    start_time = time.perf_counter()
    weights = neuron_fit(X_tr, t_tr, name)
    end_time = time.perf_counter()
    pred = neuron_predict(weights, X_va, name)
    accuracy = accuracy_score(t_va, pred)
    print("Accuracy: ", accuracy)
    print("Time: ", end_time - start_time)
    plot_boundary(lambda P: neuron_predict(weights, P, name), X_tr, t_tr)

