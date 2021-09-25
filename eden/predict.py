import argparse
import wandb

import pandas as pd


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
    predict(p, args["mp"], args["o"])
