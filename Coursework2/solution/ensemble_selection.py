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
        self.supported_metric = ['accuracy','log_loss']
        if self.metric not in self.supported_metric:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def fit(self, X, y):
        self.has_fit_ = True
        y = np.asarray(y).ravel() ## 1D Array
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        oof_raw, _ = generate_oof_probas(
            self.estimators, X, y, self.cv, self.random_state
        )

        standardized_oof = []
        for item in oof_raw:
            arr = item[0] if isinstance(item, tuple) else item
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] != n_classes:
                full_arr = np.zeros((len(y), n_classes), dtype=np.float64)
                full_arr[:, :arr.shape[1]] = arr
                standardized_oof.append(full_arr)
            else:
                standardized_oof.append(arr)
        oof_matrix = np.array(standardized_oof, dtype=np.float64)
        n_estimators = len(self.estimators)
        is_accuracy = (self.metric == "accuracy")
        max_size = (
            self.max_ensemble_size
            if self.max_ensemble_size is not None
            else n_estimators
        )

        def evaluate_subset(indices):
            blended_probs = np.mean(oof_matrix[indices], axis=0)
            if is_accuracy:
                preds = self.classes_[np.argmax(blended_probs, axis = 1)]
                return accuracy_score(y, preds)
            else:
                eps = 1e-15
                blended_probs = np.clip(blended_probs, eps, 1 - eps)
                blended_probs = blended_probs / np.sum(blended_probs, axis=1, keepdims=True)
                return log_loss(y, blended_probs, labels=self.classes_)

        def has_improved(current, best):
            return (is_accuracy and current > best) or (not is_accuracy and current < best)

        best_initial_idx = None
        best_score = -np.inf if is_accuracy else np.inf
        for i in range(n_estimators):
            score = evaluate_subset([i])
            if has_improved(score, best_score):
                best_score = score
                best_initial_idx = i
        selected_indices = [best_initial_idx]
        selected_scores = [best_score]

        while len(selected_indices) < max_size and len(selected_indices) < n_estimators:
            best_candidate = None
            best_candidate_score = best_score

            for i in range(n_estimators):
                if i in selected_indices:
                    continue
                candidate_indices = selected_indices + [i]
                candidate_score = evaluate_subset(candidate_indices)
                if has_improved(candidate_score, best_candidate_score):
                    best_candidate_score = candidate_score
                    best_candidate = i

            if best_candidate is None:
                break

            selected_indices.append(best_candidate)
            selected_scores.append(best_candidate_score)
            best_score = best_candidate_score

        self.selected_indices_ = selected_indices
        self.selected_scores_ = selected_scores

        self.selected_names_ = [
            self.estimators[i][0] if isinstance(self.estimators[i], tuple) else str(self.estimators[i])
            for i in selected_indices
        ]

        self.estimators_ = []
        for i in selected_indices:
            raw_estimator = (
                self.estimators[i][1]
                if isinstance(self.estimators[i], tuple)
                else self.estimators[i]
            )
            fitted_model = clone(raw_estimator).fit(X,y)
            self.estimators_.append(fitted_model)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=['estimators_','classes_'])
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis = 1)]


    def predict_proba(self, X):
        check_is_fitted(self, attributes=['estimators_','classes_'])

        raw_probas = [estimator.predict_proba(X) for estimator in self.estimators_]
        probas_3d = np.array(raw_probas, dtype=np.float64)
        avg_probas = np.mean(probas_3d, axis=0)

        eps = 1e-15
        avg_probas = np.clip(avg_probas, eps, 1 - eps)
        avg_probas = avg_probas / np.sum(avg_probas, axis=1, keepdims=True)
        return avg_probas