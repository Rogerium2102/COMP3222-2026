from unittest.mock import patch

import pytest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ensemble_selection import GreedyEnsembleSelector

class PerfectPredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))

class PoorPredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))

@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=2,
        random_state=42,
    )
    return X, y

@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y

@pytest.fixture
def estimators():
    return [
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ]

def test_init_supported_metrics(estimators):
    selector_acc = GreedyEnsembleSelector(estimators, metric="accuracy")
    assert selector_acc.metric == "accuracy"

    selector_loss = GreedyEnsembleSelector(estimators, metric="log_loss")
    assert selector_loss.metric == "log_loss"

def test_init_invalid_metric(estimators):
    with pytest.raises(ValueError, match="Metric roc_auc is not supported."):
        GreedyEnsembleSelector(estimators, metric="roc_auc")

def test_not_fitted_error(estimators, binary_data):
    X, _ = binary_data
    selector = GreedyEnsembleSelector(estimators)
    with pytest.raises(NotFittedError):
        selector.predict(X)

    with pytest.raises(NotFittedError):
        selector.predict_proba(X)