import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
class_labels = np.unique(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def calculate_metrics(y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return acc, roc_auc, conf_matrix

def tune(model, param_grid, X_train, y_train, random_state=42):
    """Return a fitted GridSearchCV that selects parameters by accuracy.

    Use 5-fold stratified cross-validation on the training data only. With the
    default refit=True, the best configuration is refitted on all training data.
    """
    cv = StratifiedKFold(5,shuffle=True, random_state=random_state)
    gridSearch = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=cv,
        refit=True,
    )

    return gridSearch.fit(X_train, y_train)

def predict_output(model):
    y_pred = threshold_predict(model, 0.96)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc, roc_auc, conf_matrix = calculate_metrics(y_test, y_pred, y_proba)
    print("LOGISTIC REGRESSION")
    print("Accuracy: ", acc)
    print("ROC AUC: ", roc_auc)
    print("Confusion Matrix:")
    print(conf_matrix)

def threshold_predict(model, threshold):
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    return y_pred

PARAM_GRIDS = {
    "RandF": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]},
    "RotF": {"n_estimators": [5, 30, 50, 100, 200], "max_group": [2,3,5], "remove_proportion": [0.5, 0.75, 1.0]},
    "log_reg": {"log_reg__max_iter": [200,500,1000], "log_reg__tol": [1e-4,1e-3,1e-2]},
    "mlp": {"mlp__alpha": [1e-4,1e-2,1],"mlp__hidden_layer_sizes": [(16,),(64,),(64,32)]}
    # TODO: Add a similarly small grid for a second classifier.
}
log_reg = Pipeline([
    ('scalar', StandardScaler()),
    ('log_reg', LogisticRegression()),
])
log_reg = tune(
    log_reg,
    param_grid=PARAM_GRIDS['log_reg'],
    X_train=X_train,
    y_train=y_train,
    random_state=42
)
predict_output(log_reg)

mlp = Pipeline([
    ('scalar', StandardScaler()),
    ('mlp', MLPClassifier()),
])
mlp = tune(
    mlp,
    param_grid=PARAM_GRIDS['mlp'],
    X_train=X_train,
    y_train=y_train
)
predict_output(mlp)
