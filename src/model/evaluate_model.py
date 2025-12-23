from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt

def evaluate_baseline(y_train: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate baseline evaluation metrics (recall, precision, f1-score).

    Parameters
    ----------
        y_train (np.ndarray): True labels from the training set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        float: RMSE baseline score
    """

    # calculate baseline recall
    rmse_base = root_mean_squared_error(y_train, y_pred)

    return rmse_base

def evaluate_cv_train(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, Any, float, float]:
    """
    Evaluate both cross-validation performance and training performance.

    Parameters
    ----------
        estimator (Any): A fitted sklearn CV model (e.g., GridSearchCV, RandomizedSearchCV).
        x_train (np.ndarray): Feature labels from the training set.
        y_train (np.ndarray): True labels from the training set.

    Returns
    -------
        Tuple[Any, Any, float, float]:
            - CV best params
            - CV best model
            - RMSE CV
            - RMSE Train
    """

    # pick best param and best model
    best_param = estimator.best_params_
    best_model = estimator.best_estimator_

    # generate cv scores
    rmse_cv = -estimator.best_score_

    # predict models to train
    y_pred = best_model.predict(X_train)

    # generate train scores
    rmse_train = root_mean_squared_error(y_train, y_pred)

    return best_param, best_model, rmse_cv, rmse_train

def evaluate_test(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluate model performance on the test set.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        RMSE test (float): RMSE test score.
    """
    
    # generate metric scores
    rmse_test = root_mean_squared_error(y_test, y_pred)

    return rmse_test