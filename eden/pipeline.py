import pandas as pd
import numpy as np
import wandb
import logging
import os

from model.raif_hack.metrics import metrics_stat


def read_data(split_data_path="data/split_data"):
    f0 = pd.read(os.path.join(split_data_path + "t0.csv"))
    f1 = pd.read(os.path.join(split_data_path + "t1.csv"))

    p1 = pd.read(os.path.join(split_data_path + "p1.csv"))
    
    return f0, f1, p1
