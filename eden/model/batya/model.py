import typing
import pickle
import pandas as pd
import numpy as np
import logging

from wandb.lightgbm import wandb_callback
from wandb.xgboost import wandb_callback

from lightgbm import LGBMRegressor

import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from model.raif_hack.data_transformers import SmoothedTargetEncoding

from model.raif_hack.metrics import deviation_metric

from tqdm import trange


def deviation_xgb(predt, dtrain):
    y = dtrain.get_label()
    return 'raif', float(deviation_metric(y, predt))


logger = logging.getLogger(__name__)

TARGET_NAME = "per_square_meter_price"


class BatyaModel:
    def __init__(self):

        #self.model_params = model_params
        #self.num_trees = num_trees
        #self.model = None
        #self.use_wandb = use_wandb

        self.models = []

    def fit(self, train0, train1, val0=None, val1=None, p1=None):
        params_lgbm = {
            'num_leaves': 887,
            'n_estimators': 480,
            'max_depth': 11,
            'min_child_samples': 1073,
            'learning_rate': 0.05348257149091985,
            'min_data_in_leaf': 2,
            'feature_fraction': 0.9529134909800754
        }

        train = train1.drop(columns=[TARGET_NAME])
        target = train1[TARGET_NAME]

        losses = []

        kFold_random_state = [13, 666, 228, 777, 42]
        n_splits = 10
        iteration_mine = 0
        for ind_k, random_state in enumerate(kFold_random_state):
            kFold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for iteration, (train_index, valid_index) in enumerate(kFold.split(train, target)):

                X_train, X_valid = train.iloc[train_index].reset_index(drop=True), train.iloc[valid_index].reset_index(drop=True)
                y_train, y_valid = target[train_index].values, target[valid_index].values

                model = LGBMRegressor(**params_lgbm)
                model.fit(X_train, y_train)
                self.models.append(model)

                valid_pred = model.predict(X_valid)
                loss = deviation_metric(y_valid, valid_pred)

                logger.info(f"{iteration_mine}: Loss: {loss}")
                iteration_mine += 1

    def predict(self, test):
        number = 50
        logger.info("Predicting")

        predictions = self.models[0].predict(test) / number
        for i in trange(1, number):
            predictions += self.models[i].predict(test) / number

        return predictions

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
