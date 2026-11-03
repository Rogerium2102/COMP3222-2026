from part3_setup import (CV, DATASETS, RANDOM_STATE, load_dataset,
                         make_pool, make_stacking)
from sklearn.base import clone

from pool import generate_oof_probas

estimator_pool = make_pool()
stacking = make_stacking()
datasets = [load_dataset(setName) for setName in DATASETS]
datasetsLength = len(datasets)
for X_train, y_train, X_test, y_test in datasets:
    oof_probas, score = generate_oof_probas(estimator_pool, )

