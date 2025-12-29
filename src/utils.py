import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple ML models.

    - Uses GridSearchCV for sklearn models
    - Uses direct training for CatBoost (not compatible with GridSearch)
    """

    try:
        report = {}

        for model_name, model in models.items():

            # -----------------------------
            # Case 1: CatBoost Model
            # -----------------------------
            if model.__class__.__name__ == "CatBoostRegressor":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                report[model_name] = score

            # -----------------------------
            # Case 2: Sklearn Models
            # -----------------------------
            else:
                params = param.get(model_name, {})

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)

                score = r2_score(y_test, y_pred)
                report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)