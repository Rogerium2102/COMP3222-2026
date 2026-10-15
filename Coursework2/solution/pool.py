from typing import List

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, is_classifier, ClassifierMixin, BaseEstimator
from collections.abc import Mapping, Iterable

def search_for_classifier(data) -> List[ClassifierMixin]:
    classifiers = []
    if isinstance(data, BaseEstimator) and is_classifier(data):
        classifiers.append(data)
    elif isinstance(data, Mapping):
        for value in data.values():
            classifiers.extend(search_for_classifier(value))
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        for item in data:
            classifiers.extend(search_for_classifier(item))
    return classifiers


def generate_oof_probas(
        estimators,
        X,
        y,
        cv=5,
        random_state=None
) -> tuple[np.ndarray, np.ndarray]:
    estimators_searched = search_for_classifier(estimators)
    X = np.asarray(X)
    y = np.asarray(y)

    n_estimators = len(estimators_searched)
    n_cases = X.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    oof_probas = np.zeros((n_estimators, n_cases, n_classes))
    scores = np.zeros(n_estimators)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for i, base_estimator in enumerate(estimators_searched):
        for train_idx, test_idx in skf.split(X,y):
            clf = clone(base_estimator)
            clf.fit(X[train_idx], y[train_idx])
            fold_probas = clf.predict_proba(X[test_idx])
            for j, cls in enumerate(clf.classes_):
                class_idx = np.where(classes == cls)[0][0]
                oof_probas[i, test_idx, class_idx] = fold_probas[:, j]
        scores[i] = log_loss(y, oof_probas[i], labels=classes)
    return oof_probas, scores