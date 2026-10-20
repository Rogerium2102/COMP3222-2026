from encodings import unicode_escape
from typing import Callable, Any

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.validation import check_is_fitted

from pool import generate_oof_probas


class GreedyEnsembleSelector(ClassifierMixin, BaseEstimator):

    def __init__(
            self,
            estimators,
            cv=5,
            max_ensemble_size=None,
            metric="accuracy",
            random_state=None
    ):
        self.estimators = estimators
        self.cv = cv
        self.max_ensemble_size = max_ensemble_size
        self.metric = metric
        self.random_state = random_state

    classes_ = None
    selected_indices_ = None
    selected_name_ = None
    selection_scores_ = None
    estimators_ = None

    def determine_metric_(self) -> Callable[[Any, Any], float | int]:
        metric_assignment = {
            'accuracy' : accuracy_score,
            'log_loss' : log_loss
        }
        metric_hold = self.metric.lower()
        metric_function = metric_assignment.get(metric_hold)
        if metric_function is None:
            raise ValueError("Unsupported metric: {}".format(self.metric))
        return metric_function

    def fit(self, X, y):
        oof_probas, scores = generate_oof_probas(self.estimators, X, y, self.cv, self.random_state)
        self.classes_ = np.unique(y)
        metric_function = self.determine_metric_()
        n_estimators = len(self.estimators)
        max_size = (
            self.max_ensemble_size if self.max_ensemble_size is not None else n_estimators
        )
        def evaluate_subset(indices):
            blended_probs = np.mean(oof_probas[indices], axis=0)
            return metric_function(y, blended_probs)
        best_initial_idx = None
        best_score = -np.inf

        for i in range(n_estimators):
            score = evaluate_subset([i])
            if score > best_score:
                best_score = score
                best_initial_idx = i
        selected_indices = [best_initial_idx]
        selected_scores = [best_score]
        while len(selected_indices) < max_size:
            best_candidate = None
            best_candidate_score = best_score

            for i in range(n_estimators):
                if i in selected_indices:
                    continue
                candidate_indices = selected_indices + [i]
                candidate_score = evaluate_subset(candidate_indices)

                if candidate_score > best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate = i
            if best_candidate is None:
                break

            selected_indices.append(best_candidate)
            selected_scores.append(best_score)
            best_score = best_candidate_score

        self.selection_scores_ = selected_scores
        self.selected_indices_ = selected_indices
        self.selected_name_ = [self.estimators[i][0] for i in selected_indices]
        self.estimators_ = [
            clone(self.estimators[i][1]).fit(X, y) for i in selected_indices
        ]
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=['classes_','estimators_'])
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, attributes='estimators_')
        probas = [estimator.predict_proba(X) for estimator in self.estimators_]
        return np.mean(probas, axis=0)
