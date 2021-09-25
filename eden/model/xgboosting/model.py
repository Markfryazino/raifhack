import typing
import pickle
import pandas as pd
import numpy as np
import logging

from wandb.lightgbm import wandb_callback
from wandb.xgboost import wandb_callback

from lightgbm import LGBMRegressor

import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from model.raif_hack.data_transformers import SmoothedTargetEncoding

logger = logging.getLogger(__name__)

TARGET_NAME = "per_square_meter_price"

class EternalSunshineModel:
    def __init__(self, model_params, num_trees):
        self.column_names = [
            'lng', 'lat', 'osm_amenity_points_in_0.0075', 'osm_amenity_points_in_0.01',
            'osm_catering_points_in_0.01', 'osm_crossing_points_in_0.01',
            'osm_subway_closest_dist', 'total_square', 'realty_type', 
        ]

        self.model_params = model_params
        self.num_trees = num_trees
        self.model = None

    def fit_pt0(self, train0, val0=None):
        logger.info("Fitting PT 0")
        X_train0 = train0.drop(columns=[TARGET_NAME])[self.column_names]
        y_train0 = train0[TARGET_NAME]
        train_xgb = xgb.DMatrix(X_train0, y_train0, enable_categorical=True)
        evals = [(train_xgb, "train")]
        if val0 is not None:
            X_val0 = val0.drop(columns=[TARGET_NAME])[self.column_names]
            y_val0 = val0[TARGET_NAME]
            val_xgb = xgb.DMatrix(X_val0, y_val0, enable_categorical=True)
            evals.append((val_xgb, "val"))

        self.model = xgb.train(self.model_params, train_xgb, self.num_trees,
                               evals=evals, callbacks=[wandb_callback()])

    def fit(self, train0, train1, val0=None, val1=None):
        self.fit_pt0(train0, val0)

    def predict(self, test):
        logger.info("Predicting")
        test = test[self.column_names]
        test_xgb = xgb.DMatrix(test)
        predictions = self.model.predict(test_xgb) * 0.95
        return predictions

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
