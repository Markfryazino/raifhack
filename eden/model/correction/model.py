import typing
import pickle
import pandas as pd
import numpy as np
import logging

from wandb.lightgbm import wandb_callback

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from model.raif_hack.data_transformers import SmoothedTargetEncoding

logger = logging.getLogger(__name__)


class BetterCorrectionModel():
    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 model_params: typing.Dict[str, typing.Union[str,int,float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.nice_features = [
            'lng', 'lat', 'osm_amenity_points_in_0.0075', 'osm_amenity_points_in_0.01',
            'osm_catering_points_in_0.01', 'osm_crossing_points_in_0.01',
            'osm_subway_closest_dist', 'total_square', 'realty_type', 'predicted_target'
        ]

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
             self.ste_cat_features)])

        self.model = LGBMRegressor(**model_params)
        self.knn = KNeighborsRegressor()
        self.knn_scaler = StandardScaler()

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)])

        self._is_fitted = False

    def fit_knn(self, X_manual: pd.DataFrame, y_manual: pd.Series):

        logger.info("fit knn")

        X_manual_cp = X_manual.copy()

        predictions = self.pipeline.predict(X_manual)
        X_manual_cp["predicted_target"] = predictions
        y_knn = y_manual / predictions

        X_manual_cp = self.knn_scaler.fit_transform(X_manual_cp[self.nice_features])

        self.knn.fit(X_manual_cp, y_knn)

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series,
            X_manual: pd.DataFrame, y_manual: pd.Series):

        if "predicted_target" in X_manual.columns:
            X_manual.drop(columns=["predicted_target"], inplace=True)

        logger.info('Fit lightgbm')
        self.pipeline.fit(X_offer, y_offer, model__feature_name=[f'{i}' for i in range(70)],model__categorical_feature=['67','68','69'])
        
        
        
        self.fit_knn(X_manual, y_manual)
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:

        if "predicted_target" in X.columns:
            X.drop(columns=["predicted_target"], inplace=True)

        if self.__is_fitted:
            X_cp = X.copy()
            predictions = self.pipeline.predict(X)
            X_cp["predicted_target"] = predictions
            X_cp = self.knn_scaler.transform(X_cp[self.nice_features])
            corrected_price = predictions * self.knn.predict(X_cp)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model