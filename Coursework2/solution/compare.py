from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from ensemble_selection import GreedyEnsembleSelector, GreedyEnsembleSelectorAI
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
        "Best Single Classifier": accuracy_score(y_test, pred_1),
        "Full-Pool Ensemble": accuracy_score(y_test, pred_2),
        "Stacking Ensemble": accuracy_score(y_test, pred_3),
        "Greedy (Accuracy)": accuracy_score(y_test, pred_4),
        "Greedy (Log Loss)": accuracy_score(y_test, pred_5),
    }

print(f"{'Dataset':<20} | {'Best Single':<12} | {'Full-Pool':<12} | {'Stacking':<12} | {'Greedy (Acc)':<12} | {'Greedy (Loss)':<12}")
print("-" * 95)
for dataset, metrics in results.items():
    print(f"{dataset:<20} | {metrics['Best Single Classifier']:<12.4f} | {metrics['Full-Pool Ensemble']:<12.4f} | {metrics['Stacking Ensemble']:<12.4f} | {metrics['Greedy (Accuracy)']:<12.4f} | {metrics['Greedy (Log Loss)']:<12.4f}")


