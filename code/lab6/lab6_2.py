from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from lab6 import plot_boundary, perceptron_fit, perceptron_predict, plot_mistakes
import numpy as np

rng = np.random.default_rng(0) # use this seed
C = np.array([[2, 1], [1, 2]]) # shared covariance
data0 = rng.multivariate_normal(mean=[0, 3], cov=C, size=100) # class 0
data1 = rng.multivariate_normal(mean=[3, 0], cov=C, size=100) # class 1
X2 = np.vstack([data0, data1])
t2 = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]

X_tr, X_va, t_tr, t_va = train_test_split(X2, t2, test_size=0.3,
    stratify=t2, random_state=0)

clf = Perceptron(random_state=0)
clf.fit(X_tr, t_tr)

final_w, mistakes_per_epoch = perceptron_fit(X_tr, t_tr)

accuracy_valid = clf.score(X_va, t_va)
accuracy_train = clf.score(X_tr, t_tr)
print(f"Accuracy on validation: {accuracy_valid}")
print(f"Accuracy on training: {accuracy_train}")
plot_boundary(clf.predict, X_tr, t_tr)
plot_boundary(lambda P: perceptron_predict(final_w, P), X_tr, t_tr)
plot_mistakes(mistakes_per_epoch)
