import pandas as pd
import numpy as np
import wandb
import logging
import os
import sys
import datetime

from sklearn.model_selection import train_test_split

from model.raif_hack.metrics import metrics_stat
from model.raif_hack.settings import LOGGING_CONFIG

from model.correction.better_corr import better_corr_baseline_predict
from model.correction.better_corr import better_corr_baseline_train

from model.baseline import baseline_predict, baseline_train
from model.xgboosting.entrypoint import eternal_sunshine_predict, eternal_sunshine_train


# LOGGING_FILENAME = "pipeline" + datetime.datetime.now().strftime(format="D%d-%H:%M:%S") + ".log"
# LOGGING_FILENAME = "logs/pipeline.log"
# LOGGING_CONFIG["handlers"]["file_handler"]["filename"] = LOGGING_FILENAME
# logging.config.dictConfig(LOGGING_CONFIG)
# logger = logging.getLogger(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

TARGET_NAME = "per_square_meter_price"


def read_data(split_data_path="data/split_data"):
    f0 = pd.read_csv(os.path.join(split_data_path, "t0.csv"))
    f1 = pd.read_csv(os.path.join(split_data_path, "t1.csv"))

    p1 = pd.read_csv(os.path.join(split_data_path, "p1.csv"))
    
    return f0, f1, p1


def create_run(config, **kwargs):
    wandb.login()
    run = wandb.init(
        entity="fencepainters",
        project="raifhack",
        job_type="pipeline",
        config=config,
        save_code=True,
        **kwargs
    )

    artifact = run.use_artifact("SplitDataset:latest")
    artifact.download(root="data/split_data")

    return run


def train_test_split(run, f0, f1):
    f0["date"] = f0["date"].apply(pd.to_datetime)
    f1["date"] = f1["date"].apply(pd.to_datetime)

    split_date = "2020-07-01 00:00:00"
    t0 = f0.loc[f0["date"] < datetime.datetime.strptime(split_date, "%Y-%m-%d %H:%M:%S")].sample(frac=1., random_state=42)
    v0 = f0.loc[f0["date"] >= datetime.datetime.strptime(split_date, "%Y-%m-%d %H:%M:%S")].sample(frac=1., random_state=42)
    t1 = f1.loc[f1["date"] < datetime.datetime.strptime(split_date, "%Y-%m-%d %H:%M:%S")].sample(frac=1., random_state=42)
    v1 = f1.loc[f1["date"] >= datetime.datetime.strptime(split_date, "%Y-%m-%d %H:%M:%S")].sample(frac=1., random_state=42)

    wandb.log({
        "split_date": split_date,
        "t0_shape": t0.shape,
        "t1_shape": t1.shape,
        "v0_shape": v0.shape,
        "v1_shape": v1.shape,
        "0_val_frac": v0.shape[0] / f0.shape[0],
        "1_val_frac": v1.shape[0] / f1.shape[0]
    })

    return t0, v0, t1, v1



def pipeline(config, run_kwargs):
    logger.info("START PIPELINE")

    run = create_run(config, **run_kwargs)

    f0, f1, p1 = read_data(config["data_path"])
    logger.info("START TRAIN TEST SPLIT")
    t0, v0, t1, v1 = train_test_split(run, f0, f1)

    logger.info("START TRAINING")
    mp, train_logs = config["train_method"](t0, t1, config["model_path"], v0=v0, v1=v1, **config["train_kwargs"])
    wandb.log({"train_logs": train_logs})

    logger.info("START PREDICTION")
    v1_o, v1_pred, v1_logs = config["predict_method"](v1, mp)
    p1_o, p1_pred, p1_logs = config["predict_method"](p1, mp, config["submission_path"])

    v1_metrics = metrics_stat(v1[TARGET_NAME].values, v1_pred[TARGET_NAME].values)
    wandb.log({
        "v1_predict_logs": v1_logs,
        "p1_predict_logs": p1_logs,
        "v1_metrics":v1_metrics
    })

    logger.info("FINISH")
    wandb.save(config["model_path"])
    wandb.save(config["submission_path"])
    wandb.finish()


CONFIG = {
    "model_path": "saved_models/dummy.pkl",
    "submission_path": "submissions/pipeline_submit.csv",
    "data_path": "data/split_data",
    "train_method": eternal_sunshine_train,
    "predict_method": eternal_sunshine_predict,
    "train_kwargs": {
        "xgboost_params": {
            'objective': 'reg:squarederror'
        },
        "num_trees": 7,
        "use_wandb": True
    }
}

if __name__ == "__main__":
    pipeline(CONFIG, {})
