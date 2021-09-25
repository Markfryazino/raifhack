import pandas as pd
import numpy as np
import wandb
import logging
import os

from sklearn.model_selection import train_test_split

from model.raif_hack.metrics import metrics_stat
from model.raif_hack.settings import LOGGING_CONFIG


def read_data(split_data_path="data/split_data"):
    f0 = pd.read(os.path.join(split_data_path + "t0.csv"))
    f1 = pd.read(os.path.join(split_data_path + "t1.csv"))

    p1 = pd.read(os.path.join(split_data_path + "p1.csv"))
    
    return f0, f1, p1


def create_run(**kwargs):
    wandb.login()
    run = wandb.init(
        entity="fencepainters",
        project="raifhack",
        job_type="pipeline",
        **kwargs
    )

    artifact = run.use_artifact("SplitDataset:latest")
    artifact.download(root="data/split_data")

    return run


def train_test_split(f0, f1):
    t0, v0 = train_test_split(f0, test_size=0.2, shuffle=True, random_state=42)
    t1, v1 = train_test_split(f1, test_size=0.25, shuffle=True, random_state=42)

    return t0, v0, t1, v1


