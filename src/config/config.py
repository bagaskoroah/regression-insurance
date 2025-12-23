import numpy as np

# data path
RAW_DATA = 'data/raw/insurance.csv'
CLEAN_DATA = 'data/processed/insurance_cleaned.csv'

# numerical and categorical features
NUM_COLS = ['age', 'bmi', 'children']
CAT_COLS = ['sex', 'smoker', 'region']

# splitting data
TARGET = 'charges'

# train test split
RANDOM_STATE = 123
TEST_SIZE = 0.2

# cv arguments
CV = 5
N_ITER = 100
VERBOSE = 0
N_JOBS = -1

# model parameters
XGBOOST_PARAMS = {
    'model__n_estimators': [100, 200, 500, 750, 1000],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 10],
    'model__reg_lambda': [0, 0.1, 0.25, 0.5, 1, 2],
    'model__reg_alpha': [0, 0.1, 0.5]
    }

CATBOOST_PARAMS = {
    'model__iterations': [200, 500, 800],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__depth': [4, 6, 8, 10]
    }

KNN_PARAMS = {
    'model__n_neighbors': list(range(3, 26, 2))
}

RF_PARAMS = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 5, 10, 20, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    "model__max_features": ['sqrt', 'log2']
}

LR_PARAMS = {
    'model__alpha': np.logspace(-3, 3, 20)
}

# model output directory path
MODEL_PATH = 'artifacts/model/best_model.pkl'