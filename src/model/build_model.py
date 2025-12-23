from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.config import config
from sklearn.dummy import DummyRegressor
import numpy as np
import time
from sklearn.pipeline import Pipeline
from typing import Any, Dict


def build_baseline(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Build and train a baseline DummyRegressor using a stratified strategy.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    np.ndarray
        Predictions from the baseline model on training data.
    """
    
    # create baseline object 
    base_model = DummyRegressor(strategy='mean')

    # fit object to train data
    base_model.fit(X_train, y_train)

    # predict train data
    y_pred = base_model.predict(X_train)

    return y_pred

def build_cv_train(
        estimator: Any, 
        preprocessor: Any, 
        params: Dict[str, Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray) -> Any:
    """
    Perform cross-validated model training with preprocessing and SMOTE pipeline.
    Evaluates the best model on training data and returns predictions + best model.

    Parameters
    ----------
    estimator : Any
        Machine learning estimator to train.
    preprocessor : Any
        Preprocessing transformer.
    params : dict
        Hyperparameter search space for RandomizedSearchCV.
    X_train : np.ndarray
        Training input features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    tuple
        cv_model : Any  
            Group of CV models to be selected as a best model in the next of evaluation process.
    """

    # define start time process
    start_time = time.time()

    model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', estimator)])

    if isinstance(estimator, (RandomForestRegressor, CatBoostRegressor, XGBRegressor)):
        cv_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=config.N_ITER,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.N_JOBS
        )

    else:
        cv_model = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.N_JOBS
        )

    # fit cv and train
    cv_model.fit(X_train, y_train)
    
    # define end time process
    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)
    print(f'Model {estimator.__class__.__name__} has been created succesfully, time elapsed: {end_time} minutes.')

    return cv_model

def build_test(
        estimator: Any, 
        X_test: np.ndarray) -> np.ndarray:
    
    """
    Generate predictions from the final trained estimator on the test data.

    Parameters
    ----------
    estimator : Any
        Trained model used to generate predictions.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    np.ndarray
        Predicted labels on the test set.
    """

    # define start time
    start_time = time.time()

    # predict models on test data
    y_pred = estimator.predict(X_test)
    
    # calculate processing time
    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)

    print(f"Model {estimator.named_steps['model']} has been created succesfully, time elapsed: {end_time} minutes.")

    return y_pred