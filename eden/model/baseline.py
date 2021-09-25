import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from model.raif_hack.model import BenchmarkModel
from model.raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from model.raif_hack.utils import PriceTypeEnum
from model.raif_hack.metrics import metrics_stat
from model.raif_hack.features import prepare_categorical

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def baseline_train(t0, t1, mp, v0=None, v1=None, p1=None, run=None, **kwargs):
    try:
        logger.info('START train.py')
        logger.info('Load train df')
        train_df = pd.concat([t0, t1])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(mp)

        predictions_offer = model.predict(X_offer)
        ometrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {ometrics}')

        predictions_manual = model.predict(X_manual)
        mmetrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {mmetrics}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')

    return mp, {
        "offer_metrics": ometrics,
        "manual_metrics": mmetrics
    }


def baseline_predict(p1, mp, o, run=None, **kwargs):
    try:
        logger.info('START predict.py')
        logger.info('Load test df')
        test_df = p1.copy()
        logger.info(f'Input shape: {test_df.shape}')
        test_df = prepare_categorical(test_df)

        logger.info('Load model')
        model = BenchmarkModel.load(mp)
        logger.info('Predict')
        test_df['per_square_meter_price'] = model.predict(test_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES])
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(o, index=False)
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)

    logger.info('END predict.py')
    return o, {}
