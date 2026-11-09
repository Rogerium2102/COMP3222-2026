import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pool import generate_oof_probas, search_for_classifier, is_valid_classifier

class CustomClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        return np.ones((n_samples, n_classes)) / n_classes

class InvalidClassifier:
    pass

@pytest.fixture
def make_binary_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42,
    )
    return X, y

@pytest.fixture
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

@pytest.fixture
def make_basic_estimators():
    return [
        ('LR',LogisticRegression(max_iter=200, random_state=42)),
        ('RF',RandomForestClassifier(n_estimators=15,random_state=42)),
    ]

@pytest.fixture
def setup_basic_with_estimator(make_basic_estimators, make_basic_data):
    X, y = make_basic_data
    estimators = make_basic_estimators
    return X, y, estimators

def test_random_state_string(setup_basic_with_estimator):
    cv_folds = 5
    X, y, estimators = setup_basic_with_estimator
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds, random_state="Hello"))

def test_works_with_normal_lists(make_basic_estimators):
    estimators = make_basic_estimators
    X = [[1,2,3],[3,4,5],[6,2,5],[7,2,8],[3,7,2],[4,6,8]]
    y = [0,1,2,1,2,0]
    cv_folds = 2
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv_folds, random_state=42)
    n_estimators = len(estimators)
    n_cases = len(X)
    n_classes = len(np.unique(y))
    assert oof_probas.shape == (n_estimators, n_cases, n_classes)

def test_works_with_estimator_tuple(make_basic_data):
    estimators = [('Hello', LogisticRegression(random_state=42)), ('Bye', RandomForestClassifier(n_estimators=15, random_state=42))]
    X, y = make_basic_data
    cv_folds = 5
    oof_probas, scores = generate_oof_probas(estimators, X, y, cv_folds, random_state=42)
    n_cases = len(X)
    n_classes = len(np.unique(y))
    n_estimators = len(estimators)
    assert oof_probas.shape == (n_estimators, n_cases, n_classes)

def test_bad_esimators_list(make_basic_data):
    estimators = [RandomForestClassifier(n_estimators=15, random_state=42), LogisticRegression(random_state=42)]
    X, y = make_basic_data
    cv_folds = 5
    pytest.raises(TypeError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_bad_cv_folds(setup_basic_with_estimator):
    cv_folds = 1
    X, y, estimators = setup_basic_with_estimator
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_duplicate_estimator_name(make_basic_data):
    cv_folds = 5
    estimators = [('Hello', LogisticRegression(random_state=42)),('Bye', RandomForestClassifier(n_estimators=15, random_state=42)),('Bye', LogisticRegression(random_state=42))]
    X, y, = make_basic_data
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_extreme_estimator_format(make_basic_data):
    estimators = [('Hello', LogisticRegression(random_state=42),3567, ('Bye', RandomForestClassifier(n_estimators=15, random_state=42),("Hi",[])),
     ('Bye', LogisticRegression(random_state=42),12,()))]
    X, y, = make_basic_data
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

def test_different_X_y_lengths(make_basic_estimators):
    cv_folds = 5
    X = [[1,2,3],[3,4,5],[6,2,5],[7,2,8],[3,7,2],[4,6,8]]
    y = [0,1,2,1,2,0,4]
    estimators = make_basic_estimators
    assert pytest.raises(ValueError, lambda: generate_oof_probas(estimators, X, y, cv_folds))

def test_search_for_classifier_valid(make_basic_estimators):
    result = search_for_classifier(make_basic_estimators)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "LR" in result and "RF" in result
    assert isinstance(result["LR"], LogisticRegression)

def test_search_for_classifier_not_iterable():
    with pytest.raises(TypeError, match="must be a list of tuples"):
        search_for_classifier(12345)


def test_search_for_classifier_invalid_tuple_type():
    with pytest.raises(TypeError, match="must be a list of tuples"):
        search_for_classifier(["lr", LogisticRegression()])


def test_search_for_classifier_invalid_tuple_length():
    with pytest.raises(ValueError, match="must be a list of tuples of form"):
        search_for_classifier([("lr", LogisticRegression(), "extra")])

def test_search_for_classifier_duplicate_names():
    duplicate_data = [
        ("model", LogisticRegression()),
        ("model", RandomForestClassifier()),
    ]
    with pytest.raises(ValueError, match="Duplicate classifier name"):
        search_for_classifier(duplicate_data)

def test_is_valid_classifier():
    assert is_valid_classifier(LogisticRegression()) is True
    assert is_valid_classifier(CustomClassifier()) is True
    assert is_valid_classifier(InvalidClassifier()) is False
    assert is_valid_classifier("not_an_estimator") is False

def test_generate_oof_probas_output_shapes_binary(make_binary_data, make_basic_estimators):
    X, y = make_binary_data
    cv = 5
    estimators = make_basic_estimators
    oof_probas, scores = generate_oof_probas(
        estimators, X, y, cv, random_state=42
    )

    n_estimators = len(estimators)
    n_samples = len(X)
    n_classes = len(np.unique(y))

    assert oof_probas.shape == (n_estimators, n_samples, n_classes)
    assert scores.shape == (n_estimators,)
    assert np.all(scores >= 0)

def test_generate_oof_probas_multiclass(make_basic_data, make_basic_estimators):
    X, y = make_basic_data
    estimators = make_basic_estimators
    oof_probas, scores = generate_oof_probas(
        estimators, X, y, cv=5, random_state=42
    )
    n_estimators = len(estimators)
    n_samples = len(X)
    n_classes = len(np.unique(y))
    assert oof_probas.shape == (n_estimators, n_samples, n_classes)
    for i in range(n_estimators):
        np.testing.assert_allclose(
            oof_probas[i].sum(axis = 1), np.ones(n_samples), atol = 1e-6
        )

def test_generate_oof_probas_reproducability(make_binary_data, make_basic_estimators):
    X, y = make_binary_data
    oof1, scores1 = generate_oof_probas(make_basic_estimators, X, y, cv=5, random_state=42)
    oof2, scores2 = generate_oof_probas(make_basic_estimators, X, y, cv=5, random_state=42)

    np.testing.assert_array_equal(oof1, oof2)
    np.testing.assert_array_equal(scores1, scores2)

def test_generate_oof_probas_mismatched_length(make_basic_estimators):
    X = np.ones((10, 2))
    y = np.ones(8)
    with pytest.raises(ValueError, match="X and y must have the same length"):
        generate_oof_probas(make_basic_estimators, X, y, cv=5, random_state=42)

@pytest.mark.parametrize("invalid_cv, match_err", [
    ("five",TypeError),
    (1.5, TypeError),
    (1, ValueError),
    (0, ValueError),
    (-3, ValueError),
])
def test_generate_oof_probas_invalid_cv(make_binary_data, make_basic_estimators, invalid_cv, match_err):
    X, y = make_binary_data
    with pytest.raises(match_err):
        generate_oof_probas(make_basic_estimators, X, y, cv=invalid_cv)

def test_generate_oof_probas_invalid_estimator(make_binary_data):
    X, y = make_binary_data
    invalid_estimators = [("bad_model", InvalidClassifier())]
    with pytest.raises(TypeError, match="Not a valid classifier or estimator"):
        generate_oof_probas(invalid_estimators, X, y, cv=5)
