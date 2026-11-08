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

def test_fit_and_predict_binary(binary_data, estimators):
    X, y = binary_data
    selector = GreedyEnsembleSelector(
        estimators, cv=3, metric="accuracy", random_state=42
    )
    selector.fit(X, y)

    assert hasattr(selector, "classes_")
    assert hasattr(selector, "selected_indices_")
    assert hasattr(selector, "selected_names_")
    assert hasattr(selector, "estimators_")

    assert len(selector.selected_indices_) >= 1
    assert len(selector.estimators_) == len(selector.selected_indices_)
    assert len(selector.selected_names_) == len(selector.selected_indices_)

    preds=selector.predict(X)
    probas = selector.predict_proba(X)
    assert preds.shape == (len(X),)
    assert probas.shape == (len(X), 2)

    np.testing.assert_allclose(probas.sum(axis=1), np.ones(len(X)), atol=1e-6)

def test_fit_multiclass_log_loss(multiclass_data, estimators):
    X, y = multiclass_data
    selector = GreedyEnsembleSelector(estimators, metric="log_loss", cv=3, random_state=42)
    selector.fit(X, y)

    preds = selector.predict(X)
    probas = selector.predict_proba(X)

    assert len(np.unique(preds)) <= 3
    assert probas.shape == (len(X), 3)
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(len(X)), atol=1e-6)

def test_max_ensemble_size_constraint(binary_data, estimators):
    X, y = binary_data
    max_size = 2
    selector = GreedyEnsembleSelector(
        estimators, cv=3, max_ensemble_size=max_size, metric="accuracy"
    )
    selector.fit(X, y)

    assert len(selector.selected_indices_) <= max_size
    assert len(selector.estimators_) <= max_size

@patch("ensemble_selection.generate_oof_probas")
def test_greedy_selection_accuracy(mock_generate_oof, binary_data):
    X, y = binary_data
    n_samples = X.shape[0]
    perfect_oof = np.zeros((n_samples, 2))
    perfect_oof[np.arange(n_samples), y] = 1.0

    bad_oof = np.zeros((n_samples, 2))
    bad_oof[np.arange(n_samples),1 - y] = 1.0

    mock_generate_oof.return_value = (
        np.array([bad_oof, perfect_oof]),
        np.array([0.0,1.0])
    )

    estimators = [("bad model", PoorPredictor()), ("good model", PerfectPredictor())]
    selector = GreedyEnsembleSelector(estimators, metric="accuracy", cv=3)
    selector.fit(X, y)

    assert selector.selected_indices_ == [1]
    assert selector.selected_names_ == ["good model"]

@patch("ensemble_selection.generate_oof_probas")
def test_handling_raw_oof_structures(mock_generate_oof, binary_data):
    X, y = binary_data
    n_samples = X.shape[0]
    oof_item_tuple = (np.random.uniform(size=(n_samples, 2)), "extra_metadata")
    oof_item_2d_truncated = np.random.uniform(size=(n_samples, 1))

    mock_generate_oof.return_value = (
        [oof_item_tuple, oof_item_2d_truncated],
        np.array([0.5, 0.5])
    )

    estimators = [("model1", LogisticRegression()), ("model2", LogisticRegression())]
    selector = GreedyEnsembleSelector(estimators, metric="accuracy", cv=3)

    selector.fit(X, y)
    assert len(selector.selected_indices_) > 0

def test_invalid_estimators_format_raises_error(binary_data):
    X, y = binary_data
    invalid_estimators = [
        LogisticRegression(random_state=42),
        DecisionTreeClassifier(random_state=42),
    ]
    selector = GreedyEnsembleSelector(invalid_estimators, cv=3)
    with pytest.raises(TypeError):
        selector.fit(X, y)