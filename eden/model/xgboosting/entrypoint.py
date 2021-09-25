import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from model.xgboosting.model import EternalSunshineModel
from model.raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from model.raif_hack.utils import PriceTypeEnum
from model.raif_hack.metrics import metrics_stat
from model.raif_hack.features import prepare_categorical

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def eternal_sunshine_train(t0, t1, mp, v0=None, v1=None, p1=None, run=None, **kwargs):
    _t0 = t0.copy()
    _t1 = t1.copy()
    _v0 = None if v0 is None else v0.copy()
    _v1 = None if v1 is None else v1.copy()
    _p1 = None if p1 is None else p1.copy()

    model = EternalSunshineModel(model_params=kwargs["xgboost_params"], num_trees=kwargs["num_trees"],
                                 use_wandb=kwargs["use_wandb"])
    logger.info('Fit model')
    model.fit(_t0, _t1, _v0, _v1)
    logger.info('Save model')
    model.save(mp)

    logger.info('END train.py')

    return mp, {}


def eternal_sunshine_predict(p1, mp, o=None, run=None, **kwargs):
    _p1 = p1.copy()
    logger.info('START predict.py')
    logger.info('Load test df')

    logger.info('Load model')
    model = EternalSunshineModel.load(mp)
    logger.info('Predict')
    _p1['per_square_meter_price'] = model.predict(_p1)
    logger.info('Save results')
    if o is not None:
        _p1[['id','per_square_meter_price']].to_csv(o, index=False)

    logger.info('END predict.py')
    return o, _p1, {}
