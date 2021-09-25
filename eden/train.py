import argparse
import wandb

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt_0", "-t0", type=str, dest="t0", required=True)
    parser.add_argument("--train_pt_1", "-t1", type=str, dest="t1", required=False)
    parser.add_argument("--val_pt_0", "-v0", type=str, dest="v0", required=True)
    parser.add_argument("--val_pt_1", "-v1", type=str, dest="v1", required=False)
    parser.add_argument("--predict_1", "-p1", type=str, dest="p1", required=False)

    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True)

    return parser.parse_args()


def train(t0, t1, mp, v0=None, v1=None, p1=None, run=None, **kwargs):
    pass


def main():
    args = vars(parse_args())
    t0 = pd.read_csv(t0)
    t1 = pd.read_csv(t1)

    other = {}
    for cur_name in ["v0", "v1", "p1"]:
        if cur_name in args:
            other[cur_name] = pd.read_csv(args[cur_name])
    
    train(t0, t1, args["mp"], **other)