from statistics import LinearRegression

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pool import generate_oof_probas



def make_basic_data():
    X, y =  make_classification(
        n_samples=150,
        n_features=5,
        n_classes=3,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    return X, y

def make_basic_estimators():
    return [
        ('LR',LogisticRegression(max_iter=200, random_state=42)),
        ('RF',RandomForestClassifier(n_estimators=15,random_state=42)),
    ]

@pytest.fixture
def setup_basic_with_estimator():
    X, y = make_basic_data()
    estimators = make_basic_estimators()
    return X, y, estimators

def test_basic(setup_basic_with_estimator):
    cv_folds = 5
    X,y,estimators = setup_basic_with_estimator
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv_folds, random_state=42)
    n_estimators = len(estimators)
    n_cases = X.shape[0]
    n_classes = len(np.unique(y))

    assert oof_probas.shape == (n_estimators, n_cases, n_classes)
    assert scores.shape == (n_estimators,)
    assert np.allclose(oof_probas.sum(axis=2), 1.0)
    assert np.all(scores > 0)

def test_random_state_string(setup_basic_with_estimator):
    cv_folds = 5
    X, y, estimators = setup_basic_with_estimator
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds, random_state="Hello"))

def test_works_with_normal_lists():
    estimators = make_basic_estimators()
    X = [[1,2,3],[3,4,5],[6,2,5],[7,2,8],[3,7,2],[4,6,8]]
    y = [0,1,2,1,2,0]
    cv_folds = 2
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv_folds, random_state=42)
    n_estimators = len(estimators)
    n_cases = len(X)
    n_classes = len(np.unique(y))
    assert oof_probas.shape == (n_estimators, n_cases, n_classes)

def test_works_with_estimator_tuple():
    estimators = [('Hello', LogisticRegression(random_state=42)), ('Bye', RandomForestClassifier(n_estimators=15, random_state=42))]
    X, y = make_basic_data()
    cv_folds = 5
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv_folds, random_state=42)
    n_cases = len(X)
    n_classes = len(np.unique(y))
    n_estimators = len(estimators)
    assert oof_probas.shape == (n_estimators, n_cases, n_classes)

def test_bad_esimators_list():
    estimators = [RandomForestClassifier(n_estimators=15, random_state=42), LogisticRegression(random_state=42)]
    X, y = make_basic_data()
    cv_folds = 5
    pytest.raises(TypeError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_bad_cv_folds(setup_basic_with_estimator):
    cv_folds = 1
    X, y, estimators = setup_basic_with_estimator
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_duplicate_estimator_name():
    cv_folds = 5
    estimators = [('Hello', LogisticRegression(random_state=42)),('Bye', RandomForestClassifier(n_estimators=15, random_state=42)),('Bye', LogisticRegression(random_state=42))]
    X, y, = make_basic_data()
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_extreme_estimator_format():
    estimators = [('Hello', LogisticRegression(random_state=42),3567, ('Bye', RandomForestClassifier(n_estimators=15, random_state=42),("Hi",[])),
     ('Bye', LogisticRegression(random_state=42),12,()))]
    X, y, = make_basic_data()
    cv_folds = 5
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_bad_estimator_good_format(setup_basic_with_estimator):
    cv_folds = 5
    X, y, estimators = setup_basic_with_estimator
    estimators.append(('Bad', 13))
    assert pytest.raises(TypeError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_bad_type_cv_folds(setup_basic_with_estimator):
    cv_folds = "HI"
    X, y, estimators = setup_basic_with_estimator
    assert pytest.raises(TypeError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_different_X_y_lengths():
    cv_folds = 5
    X = [[1,2,3],[3,4,5],[6,2,5],[7,2,8],[3,7,2],[4,6,8]]
    y = [0,1,2,1,2,0,4]
    estimators = make_basic_estimators()
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))