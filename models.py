import pandas as pd
import numpy as np
from sklearn.svm import SVR, LinearSVC
from sklearn.ensemble import GradientBoostingRegressor, BaggingClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from utils import Utils

import warnings
warnings.simplefilter("ignore")

class Models:
    def __init__(self):  # Asegúrate de que el constructor está correctamente definido
        self.reg = {
            'FORREST': RandomForestRegressor(),
            'LinearSVC': LinearSVC(),
            'GradientClass': GradientBoostingClassifier()
        }
        self.params = {
            'FORREST': {
                'n_estimators': range(6, 11),
                'criterion': ['squared_error', 'absolute_error'],
                'max_depth': range(4, 11)
            },
            'LinearSVC': {
                'max_iter': [1000],
            },
            'GradientClass': {
                'n_estimators': [125],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'squared_error']
            }
        }

    def grid_training(self, X, y, dataset_name):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            print(f"Entrenando modelo: {name}")
            grid_reg = GridSearchCV(reg, self.params[name], cv=3)
            grid_reg.fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            print(f"Modelo: {name}, Mejor puntuación: {score}")
            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        utils = Utils()
        utils.model_export(best_model, best_score, f"{dataset_name}_{name}")
        return best_model
