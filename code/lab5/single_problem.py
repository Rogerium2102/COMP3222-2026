"""Lab 5, Task 1: comparing classifiers on a single problem (student starter)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, precision_score, recall_score, f1_score, \
    roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from aeon.classification.sklearn import RotationForestClassifier

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_breast_cancer_data():
    """Return X, y and class names, with malignant as the positive class 1.

    The scikit-learn loader codes malignant as 0 and benign as 1, so the labels
    are recoded here. class_names[label] gives the name of each label.
    """
    data = load_breast_cancer()
    y = 1 - data.target
    return data.data, y, ["benign", "malignant"]


def load_iris_data():
    """Return X, y and class names for Iris (labels 0, 1 and 2)."""
    data = load_iris()
    return data.data, data.target, list(data.target_names)


def make_classifiers(random_state=RANDOM_STATE):
    """Return the three classifiers to compare, keyed by a short name.

    Put preprocessing and your classifier in a Pipeline if it needs any, so the
    preprocessing is fitted on training data only. bakeoff.py reuses this function.
    """
    return {
        "RandF": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "RotF": RotationForestClassifier(n_estimators=100, random_state=random_state),
        # TODO: Replace this stand-in with a classifier you implemented, such as
        #       your BasicEnsembleClassifier from Lab 3.
        "Yours": DecisionTreeClassifier(random_state=random_state),
    }


def single_split(y, random_state=RANDOM_STATE):
    """Return train and test row indices for a stratified 70/30 split.

    The row indices are the case identifiers to save with your predictions.
    """
    ids = np.arange(len(y))
    return train_test_split(ids, test_size=0.3, stratify=y, random_state=random_state)


def make_folds(random_state=RANDOM_STATE):
    """The 5-fold stratified cross-validation used in 1.2 and the nested extension."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def class_column(model, label):
    """Return the predict_proba column of label, using the order in model.classes_."""
    return int(np.flatnonzero(model.classes_ == label)[0])


def labelled_confusion_matrix(y_true, y_pred, class_names):
    """Return the confusion matrix as a DataFrame with actual classes as rows."""
    labels = np.arange(len(class_names))
    counts = pd.crosstab(pd.Categorical(y_true, categories=labels),
                         pd.Categorical(y_pred, categories=labels), dropna=False)
    counts.index = [f"actual {name}" for name in class_names]
    counts.columns = [f"predicted {name}" for name in class_names]
    return counts


def save_predictions(path, ids, y_true, y_pred, proba, class_names):
    """Save case ids, true labels, predictions and class probabilities to a CSV."""
    table = pd.DataFrame({"case_id": ids, "true": y_true, "predicted": y_pred})
    for k, name in enumerate(class_names):
        table[f"p_{name}"] = proba[:, k]
    table.to_csv(path, index=False)


def compute_metrics(y_true, y_pred, proba):
    """Return a dict of the 1.1 measures for one classifier.

    proba has one column per class, in label order 0, 1, ... For both problems
    return accuracy, balanced accuracy and log loss. With two classes, also return
    precision, recall, F1 and AUROC for the positive class 1. With more classes,
    return macro-averaged precision, recall and F1 and macro one-versus-rest AUROC.
    See L5.1 slides 31 and 32.
    """
    n_classes = proba.shape[1]
    acurracy = accuracy_score(y_true, y_pred)
    bal_acurr = balanced_accuracy_score(y_true, y_pred)
    logLoss = log_loss(y_true, proba)
    result = {
        "accuracy": acurracy,
        "balanced_accuracy": bal_acurr,
        "log_loss": logLoss,
    }
    if n_classes == 1:
        return result
    elif n_classes == 2:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        fOne = f1_score(y_true, y_pred)
        aur = roc_auc_score(y_true, proba[:,1])
        result["precision"] = prec
        result["recall"] = rec
        result["f1"] = fOne
        result["auroc"] = aur
        return result
    elif n_classes > 2:
        prec = average_precision_score(y_true, proba, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        fOne = f1_score(y_true, y_pred, average='macro')
        aur = roc_auc_score(y_true,proba, average="macro", multi_class='ovr')
        result["precision"] = prec
        result["recall"] = rec
        result["f1"] = fOne
        result["auroc"] = aur
        return result
    else:
        raise ValueError("No classes detected")


def check_metrics():
    """Compare compute_metrics with hand-worked values for four cases."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    proba = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
    expected = {"accuracy": 0.75, "balanced_accuracy": 0.75, "precision": 1.0,
                "recall": 0.5, "f1": 0.667, "auroc": 1.0, "log_loss": 0.400}
    found = compute_metrics(y_true, y_pred, proba)
    for name, value in expected.items():
        print(f"{name:18s} expected {value:.3f}  yours {found.get(name, float('nan')):.3f}")


def cross_val_predictions(model, X, y, folds):
    """Return out-of-fold predictions and probabilities for every case.

    For each fold, clone the model, fit it on the training rows and predict the
    held-out rows, so no case is predicted by a model that saw it. Store the
    probabilities in label order, using class_column to find each column.
    """
    n_samples = len(y)
    labels = np.unique(y)
    n_classes = len(labels)
    oof_preds = np.empty(n_samples, dtype=y.dtype)
    oof_probs = np.zeros((n_samples, n_classes), dtype=float)
    for train, predict in folds:
        mdlClone = clone(model)
        X_train = X[train]
        y_train = y[train]
        X_test = X[predict]
        mdlClone.fit(X_train, y_train)
        pred_proba = mdlClone.predict_proba(X_test)
        pred = mdlClone.predict(X_test)
        oof_preds[predict] = pred
        for target_col, label in enumerate(labels):
            if label in mdlClone.classes_:
                src_col = class_column(mdlClone, label)
                oof_probs[predict, target_col] = pred_proba[:, src_col]
    return oof_preds, oof_probs

def plot_multiclass_roc(y_true, models_oof_probs, labels):
    y_true_arr = np.asarray(y_true)
    y_true_bin = label_binarize(y_true_arr, classes=labels)
    plt.figure(figsize=(8,6))
    if y_true_bin.shape[1] == 1:
        pos_label = labels[1]
        for name, oof_probs in models_oof_probs.items():
            pos_col = 1 if oof_probs.shape[1] > 1 else 0
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), oof_probs[:, pos_col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.title(f"OOF ROC Curves (Positive Class {pos_label})")
    else:
        line_style = ['-','--','-.',':']
        for name, oof_probs in models_oof_probs.items():
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], oof_probs[:, i])
                roc_auc = auc(fpr, tpr)
                style = line_style[i % len(line_style)]
                plt.plot(
                    fpr, tpr,
                    linestyle=style,
                    lw = 2,
                    label=f"{name} - {label} (AUC = {roc_auc:.3f})"
                )
        plt.title("OFF ROC Curves across classifiers and classes")

    plt.plot([0,1],[0,1], color='navy', linestyle='--', label="Random Chance")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# For 1.3. Keys must match make_classifiers. If a classifier is in a Pipeline,
# prefix each parameter with the step name, for example "randomforestclassifier__max_depth".
PARAM_GRIDS = {
    "RandF": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]},
    "RotF": {"n_estimators": [5, 30, 50, 100, 200], "max_group": [2,3,5], "remove_proportion": [0.5, 0.75, 1.0]},
    # TODO: Add a similarly small grid for a second classifier.
}


def tune(model, param_grid, X_train, y_train, random_state=RANDOM_STATE):
    """Return a fitted GridSearchCV that selects parameters by accuracy.

    Use 5-fold stratified cross-validation on the training data only. With the
    default refit=True, the best configuration is refitted on all training data.
    """
    cv = StratifiedKFold(5,shuffle=True, random_state=random_state)
    gridSearch = GridSearchCV(
        model,
        param_grid,
        scoring='accuracy',
        cv=cv,
        refit=True,
    )

    return gridSearch.fit(X_train, y_train)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    problems = {"breast_cancer": load_breast_cancer_data(), "iris": load_iris_data()}
    classifiers = make_classifiers()
    for name, (X, y, class_names) in problems.items():
        counts = ", ".join(f"{c} {n}" for c, n in zip(class_names, np.bincount(y)))
        train, test = single_split(y)
        print(f"{name}: X {X.shape}, classes: {counts}")
        print(f"  split seed {RANDOM_STATE}: {len(train)} train, {len(test)} test")
        for name_class, ifier in classifiers.items():
            print(f"Training {name_class} on {name}")
            ifier.fit(X[train], y[train])
            y_pred = ifier.predict(X[test])
            y_proba = ifier.predict_proba(X[test])

            metrics = compute_metrics(y[test], y_pred, y_proba)
            print(metrics)
            conf_matrix = labelled_confusion_matrix(y[test], y_pred, class_names)
            print(conf_matrix)
            path = RESULTS_DIR / f"{name}_{name_class}_predictions.csv"
            save_predictions(path, test, y[test], y_pred, y_proba, class_names)

    print(f"Classifiers: {', '.join(classifiers.keys())}")

    check_metrics()
    print(f"Save your results in {RESULTS_DIR}")
    stratFold = make_folds(RANDOM_STATE)
    X, y, class_names = problems["breast_cancer"]
    labels = np.unique(y)

    mdl_oof_probs = {}
    for name, model in classifiers.items():
        _, oof_probs = cross_val_predictions(model, X, y, stratFold.split(X,y))
        mdl_oof_probs[name] = oof_probs
    #plot_multiclass_roc(y, mdl_oof_probs, labels)

    train, test = single_split(y)
    for name, model in classifiers.items():
        start_time = time.perf_counter()
        gs = tune(model, PARAM_GRIDS[name],X[train],y[train])
        end_time = time.perf_counter()
        completion_time = end_time - start_time
        print(f"Classifier: {name}, tuning time: {completion_time}, Best Params: {gs.best_params_}")

