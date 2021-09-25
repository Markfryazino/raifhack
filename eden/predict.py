import argparse
import wandb

import pandas as pd

from model.baseline import baseline_train, baseline_predict
from model.correction.better_corr import better_corr_baseline_predict, better_corr_baseline_train
from model.xgboosting.entrypoint import eternal_sunshine_train, eternal_sunshine_predict
from model.batya.entrypoint import batya_train, batya_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_1", "-p1", type=str, dest="p1", required=True)
    parser.add_argument("--output", "-o", type=str, dest="o", required=True)
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True)

    return parser.parse_args()


def predict(p1, mp, o, run=None, **kwargs):
    pass


def main():
    args = vars(parse_args())
    p1 = pd.read_csv(args["p1"])
    batya_predict(p1, args["mp"], args["o"])

if __name__ == "__main__":
    main()
