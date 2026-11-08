import csv

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from ensemble_selection import GreedyEnsembleSelector
from part3_setup import (CV, DATASETS, RANDOM_STATE, load_dataset,
                         make_pool, make_stacking)
from sklearn.base import clone
import numpy as np
from pool import generate_oof_probas

estimator_pool = make_pool()
results = {}
for setName in DATASETS:
    print(f"Working on {setName}")
    X_train, y_train, X_test, y_test = load_dataset(setName)
    oof_probas, score = generate_oof_probas(estimator_pool, X_train, y_train, CV, RANDOM_STATE)
    fitted_pool = []
    full_test_probas = []
    classes = np.unique(y_train)
    for name, estimator in estimator_pool:
        model = clone(estimator)
        model.fit(X_train, y_train)
        fitted_pool.append(model)
        full_test_probas.append(model.predict_proba(X_test))

    oof_accuracies = [
        accuracy_score(y_train, classes[np.argmax(oof, axis=1)])
        for oof in oof_probas
    ]
    best_idx = int(np.argmax(oof_accuracies))
    pred_1 = fitted_pool[best_idx].predict(X_test)

    avg_test_probas = np.mean(full_test_probas, axis=0)
    pred_2 = classes[np.argmax(avg_test_probas, axis=1)]

    stacking = make_stacking()
    stacking.fit(X_train, y_train)
    pred_3 = stacking.predict(X_test)

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    selector_acc = GreedyEnsembleSelector(estimator_pool, cv=CV, random_state=RANDOM_STATE, metric='accuracy')
    selector_acc.fit(X_train, y_train)
    pred_4 = selector_acc.predict(X_test)

    selector_log = GreedyEnsembleSelector(estimator_pool, cv=CV, random_state=RANDOM_STATE, metric='log_loss')
    selector_log.fit(X_train, y_train)
    pred_5 = selector_log.predict(X_test)

    results[setName] = {
        "best_single": accuracy_score(y_test, pred_1),
        "full_pool": accuracy_score(y_test, pred_2),
        "stacking": accuracy_score(y_test, pred_3),
        "greedy_accuracy": accuracy_score(y_test, pred_4),
        "greedy_log_loss": accuracy_score(y_test, pred_5),
        "n_selected_accuracy": len(selector_acc.selected_indices_),
        "n_selected_log_loss": len(selector_log.selected_indices_),
        "oof_accuracy": selector_acc.selected_scores_[-1],
        "chosen_classifier_names_accuray": selector_acc.selected_names_,
        "chosen_classifier_names_log_loss": selector_log.selected_names_,
    }

print(f"{'Dataset':<20} | {'Best Single':<12} | {'Full-Pool':<12} | {'Stacking':<12} | {'Greedy (Acc)':<12} | {'Greedy (Loss)':<12}")
print("-" * 95)
for dataset, metrics in results.items():
    print(f"{dataset:<20} | {metrics['best_single']:<12.4f} | {metrics['full_pool']:<12.4f} | {metrics['stacking']:<12.4f} | {metrics['greedy_accuracy']:<12.4f} | {metrics['greedy_log_loss']:<12.4f}")

print(f"{'Dataset':<20} {'Classifiers chosen (Accuracy)':<30} | {'Classifiers chosen (Log Loss)':<30}")
for dataset, metrics in results.items():
    print(f"{dataset:<20}",end=" | ")
    print(*metrics['chosen_classifier_names_accuray'], sep=", ", end=" | ")
    print(*metrics['chosen_classifier_names_log_loss'], sep=", ", end=" |\n")


print("Writing results.csv...")

header = [
    "dataset","best_single","full_pool","stacking","greedy_accuracy",
    "greedy_log_loss", "n_selected_accuracy","n_selected_log_loss","oof_accuracy"
]
with open("./../results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for dataset, metrics in results.items():
        row = [
            dataset,
            f"{metrics['best_single']:.4f}",
            f"{metrics['full_pool']:.4f}",
            f"{metrics['stacking']:.4f}",
            f"{metrics['greedy_accuracy']:.4f}",
            f"{metrics['greedy_log_loss']:.4f}",
            metrics['n_selected_accuracy'],
            metrics['n_selected_log_loss'],
            f"{metrics['oof_accuracy']:.4f}"
        ]
        writer.writerow(row)
print("Completed results.csv!")

