from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, is_classifier, ClassifierMixin, BaseEstimator
from collections.abc import Iterable

def search_for_classifier(data) -> Dict[str, ClassifierMixin]:
    """
    Given classifiers of some form this function will create a dictionary of name, classifier pairs. It will detect issues
    where classifiers came in an invalid format

    Args:
        Data - list of classifiers in some format

    Returns:
        Dictionary with classifier name as key and classifier object as value

    Raises:
        TypeError - where the data types are not correct
        ValueError - where the values of those types are not correct
    """
    classifierPairing = {}
    if not isinstance(data, Iterable):
        raise TypeError("data must be a list of tuples of form (str, classifier)")
    for dataTuple in data:
        if not isinstance(dataTuple, Tuple):
            raise TypeError("data must be a list of tuples of form (str, classifier)")
        if len(dataTuple) != 2:
            raise ValueError("data must be a list of tuples of form (str, classifier)")
        name, clf = dataTuple
        if not classifierPairing.__contains__(name):
            classifierPairing[name] = clf
        else:
            raise ValueError("Duplicate classifier name")
    return classifierPairing

def is_valid_classifier(classifier) -> bool:
    return (isinstance(classifier, ClassifierMixin) and is_classifier(classifier)) or isinstance(classifier, BaseEstimator)




def generate_oof_probas(
        estimators,
        X,
        y,
        cv=5,
        random_state=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates the Out-Of-Fold Probabilities of data on a series of estimators.
    Args:
        estimators - A list of classifiers generate OOF probabilities on.
        X - feature matrix
        y - target vector
        cv - Cross-validation k (defines the number of splits in the k-fold)
        random_state - random_state for each estimator
    Returns:
        OOF Probabilities
        How each classifier scored in log_loss metric
    Raises:
        ValueError - where the values are incorrect
        TypeError - When CV is not an integer or a TypeError occurred in search_for_classifier.
    """
    estimators_searched = search_for_classifier(estimators)
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if not isinstance(cv, int):
        raise TypeError("cv must be an integer")
    elif cv < 2:
        raise ValueError("cv must be >= 2")

    n_estimators = len(estimators_searched)
    n_cases = X.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    oof_probas = np.zeros((n_estimators, n_cases, n_classes))
    scores = np.zeros(n_estimators)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for i, base_estimator in enumerate(estimators_searched.values()):
        if not is_valid_classifier(base_estimator):
            raise TypeError("Not a valid classifier or estimator")
        for train_idx, test_idx in skf.split(X,y):
            clf = clone(base_estimator)
            clf.fit(X[train_idx], y[train_idx])
            fold_probas = clf.predict_proba(X[test_idx])
            for j, cls in enumerate(clf.classes_):
                class_idx = np.where(classes == cls)[0][0]
                oof_probas[i, test_idx, class_idx] = fold_probas[:, j]
        scores[i] = log_loss_normalised(y, oof_probas[i], classes)
    return oof_probas, scores

def log_loss_normalised(y, probs, classes):
    """
    Returns the normalised log loss metric

    Args:
        y - target vector
        probs - probability vector
        classes - class labels

    Returns:
        log loss metric (Float)
    """
    blended_probs = probs / np.sum(probs, axis=1, keepdims=True)
    return log_loss(y, blended_probs, labels=classes)