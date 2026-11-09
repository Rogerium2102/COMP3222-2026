from encodings import unicode_escape
from typing import Callable, Any

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.utils.validation import check_is_fitted

from pool import generate_oof_probas, log_loss_normalised

class GreedyEnsembleSelector(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            estimators,
            cv=5,
            max_ensemble_size=None,
            metric="accuracy",
            random_state=None
    ):
        """
        My implementation of Greedy Ensemble Selection
        Args:
        estimators - a list of classifiers
        cv - Cross-validation (default: 5)
        max_ensemble_size - the maximum number of estimators that the selector can pick (default: None (Infinite))
        metric - The metric to evaluate classifiers on (default: accuracy)
        random_state - The random state to use in each classifier.

        Supported metrics:
            accuracy - (REQUESTED)
            roc_auc - Good for sets with class imbalance.
            f1_score - Good for multiclass problems by balancing precision and recall.
            log_loss - (REQUESTED)

        """
        self.estimators = estimators
        self.cv = cv
        self.max_ensemble_size = max_ensemble_size
        self.metric = metric
        self.random_state = random_state
        self.supported_metrics = ["accuracy","log_loss","roc_auc","f1_score"]
        if self.metric not in self.supported_metrics:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def fit(self, X, y):
        """
        Fits the Ensemble Selector

        Args:
            X - feature matrix
            y - target vector

        Returns:
            self

        Raises:
            TypeError
            ValueError
        """
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
        is_higher_good = self.metric in ["accuracy","roc_auc","f1_score"]
        max_size = (
            self.max_ensemble_size
            if self.max_ensemble_size is not None
            else n_estimators
        )

        def evaluate_subset(indices):
            """
            Calculates the score for the chosen metric

            Args:
                indices - A list of indices to filter by.

            Returns:
                float | int - The score for the chosen metric
                None - Impossible event
            """
            blended_probs = np.mean(oof_matrix[indices], axis=0)
            if self.metric == "accuracy":
                preds = self.classes_[np.argmax(blended_probs, axis = 1)]
                return accuracy_score(y, preds)
            elif self.metric == "log_loss":
                eps = 1e-15
                blended_probs = np.clip(blended_probs, eps, 1 - eps)
                return log_loss_normalised(y, blended_probs, self.classes_)
            elif self.metric == "roc_auc":
                if n_classes == 2:
                    return roc_auc_score(y, blended_probs[:,1])
                else:
                    return roc_auc_score(y, blended_probs, multi_class='ovr', labels=self.classes_)
            elif self.metric == "f1_score":
                preds = self.classes_[np.argmax(blended_probs, axis = 1)]
                avg_method = 'macro' if n_classes > 2 else 'binary'
                return f1_score(y, preds, average=avg_method)
            return None

        def has_improved(current, best):
            """
            Determines if the score has improved over the best score
            Args:
                current (Float) - the current score
                best (Float) - the best score

            Returns:
                Boolean (True | False)
            """
            return (is_higher_good and current > best) or (not is_higher_good and current < best)

        best_initial_idx = None
        best_score = -np.inf if is_higher_good else np.inf
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
        """
        Predicts a vector based on the given feature matrix

        Args:
            X - feature matrix

        Returns:
              (np.array) - Predicition Matrix

        Raises:
              NotFittedError - When fit hasn't been called
        """
        check_is_fitted(self, attributes=['estimators_','classes_'])
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis = 1)]


    def predict_proba(self, X):
        """
        Predicts a vector based on the given feature matrix and returns the probability that it belongs to a class.

        Args:
            X - feature matrix

        Returns:
              (np.array) - Probabilities that it belongs to a class

        Raises:
              NotFittedError - When fit hasn't been called
        """
        check_is_fitted(self, attributes=['estimators_','classes_'])

        raw_probas = [estimator.predict_proba(X) for estimator in self.estimators_]
        probas_3d = np.array(raw_probas, dtype=np.float64)
        avg_probas = np.mean(probas_3d, axis=0)

        eps = 1e-15
        avg_probas = np.clip(avg_probas, eps, 1 - eps)
        avg_probas = avg_probas / np.sum(avg_probas, axis=1, keepdims=True)
        return avg_probas