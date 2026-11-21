import time

import numpy as np
from pathlib import Path
from aeon.distances import pairwise_distance
from aeon.utils.numba.general import z_normalise_series_3d

from coursework3_setup import DATASETS, load_resample, N_RESAMPLES

def normalize(X_train, X_test):
    data = [X_train, X_test]
    for i in range(len(data)):
        data[i] = z_normalise_series_3d(data[i])
    return data[0], data[1]

def compute_and_time_matrices(
        X_train: np.ndarray,
          X_test: np.ndarray,
        dtw_dict: dict,
          n_jobs: int = -1
          ):
    t0 = time.perf_counter()
    D_train = pairwise_distance(X_train, n_jobs=n_jobs, method="dtw", window=0.1)
    train_mat_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    D_test = pairwise_distance(X_test, X_train, n_jobs=n_jobs, method="dtw", window=0.1)
    test_mat_time = time.perf_counter() - t0

    return D_train, D_test, train_mat_time, test_mat_time

def get_file(cache_dir:Path, dataset: str, resample_id: int, param_dict: dict, method: str):
    param_str = "_".join(f"{k}-{v}" for k, v, in param_dict.items()) or "default"
    out_dir = cache_dir / dataset / f"resample_{resample_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{method}_{param_str}.npz"

def load_or_calculate_cache(
        cache_dir:Path,
        dataset: str,
        resample_id: int,
        param_dict: dict,
        method: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
):
    file_path = get_file(cache_dir, dataset, resample_id, param_dict, method)
    if file_path.exists():
        with np.load(file_path) as data:
            return (
                data["D_train"],
                data["D_test"],
                float(data["train_time"]),
                float(data["test_time"]),
            )
    else:
        D_train, D_test, train_mat_time, test_mat_time = compute_and_time_matrices(X_train, X_test, param_dict)
        save_cache(cache_dir, dataset, resample_id, param_dict, method, D_train, D_test, train_mat_time, test_mat_time)
        return D_train, D_test, train_mat_time, test_mat_time



def save_cache(cache_dir: Path, dataset: str, resample_id: int, param_dict: dict, method: str, D_train, D_test, train_mat_time, test_mat_time):
    file_path = get_file(cache_dir, dataset, resample_id, param_dict, method)
    tmp_path = file_path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            D_train=D_train,
            D_test=D_test,
            train_time=train_mat_time,
            test_time=test_mat_time,
        )
        tmp_path.replace(file_path)
    except Exception as e:
        print(f"Error saving cache {e}")
        return False
    return True




for setname in DATASETS:
    print("Computing matrices for " + setname)
    for resample in range(N_RESAMPLES):
        X_train, y_train, X_test, y_test = load_resample(setname, resample)
        X_train, X_test = normalize(X_train, X_test)
        cache_dir = Path("../cache")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        D_train, D_test, train_mat_time, test_mat_time = load_or_calculate_cache(cache_dir, setname, resample, {}, "dtw",X_train, X_test)

