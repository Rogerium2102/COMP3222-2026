import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
t = np.array([0, 0, 0, 1]) # AND targets
def plot_boundary(predict, X, t, title=""):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    zz = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#ffcccc", "#cce0ff"])
    plt.scatter(X[t == 0, 0], X[t == 0, 1], c="red", marker="o", label="Class 0")
    plt.scatter(X[t == 1, 0], X[t == 1, 1], c="blue", marker="x", s=80, label="Class 1")
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$"); plt.title(title); plt.legend(); plt.show()

def plot_mistakes(mistakes_per_epoch):
    n_epochs = len(mistakes_per_epoch)
    x_points = np.array(range(n_epochs))
    y_points = mistakes_per_epoch
    plt.xlabel("n_epochs")
    plt.ylabel("mistakes")
    plt.title("Mistakes vs Epochs")
    plt.plot(x_points, y_points)
    plt.legend()
    plt.show()

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X]) # adds the x0 = 1 column

def perceptron_fit(X, t, eta=1.0, w_init=None, max_epochs=100):
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1]) if w_init is None else np.array(w_init, dtype=float)
    mistakes_per_epoch = []
    for epoch in range(max_epochs):
        mistakes = 0
        for x, target in zip(Xb, t):
            y = 1 if np.dot(w, x) > 0 else 0
            if y != target:
                w = w - eta * (y - target) * x
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0: # TODO: stop when converged
            break
    return w, mistakes_per_epoch

def perceptron_predict(w, X):
    return (add_bias(X) @ w > 0).astype(int)

final_w, mistakes_per_epoch = perceptron_fit(X, t, eta=0.01)
print(*final_w, sep = ", ")
print(f"Epochs: {len(mistakes_per_epoch)} of {mistakes_per_epoch}")
plot_boundary(lambda P: perceptron_predict(final_w, P), X, t)
plot_mistakes(mistakes_per_epoch)

