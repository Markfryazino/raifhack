import argparse
import wandb

import pandas as pd

from model.baseline import baseline_train, baseline_predict
from model.correction.better_corr import better_corr_baseline_train, better_corr_baseline_predict
from model.xgboosting.entrypoint import eternal_sunshine_train, eternal_sunshine_predict
from model.batya.entrypoint import batya_train, batya_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt_0", "-t0", type=str, dest="t0", required=True)
    parser.add_argument("--train_pt_1", "-t1", type=str, dest="t1", required=True)
    parser.add_argument("--val_pt_0", "-v0", type=str, dest="v0", required=False)
    parser.add_argument("--val_pt_1", "-v1", type=str, dest="v1", required=False)
    parser.add_argument("--predict_1", "-p1", type=str, dest="p1", required=False)

    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True)

    return parser.parse_args()


def train(t0, t1, mp, v0=None, v1=None, p1=None, run=None, **kwargs):
    pass


TRAIN_KWARGS = {
    "xgboost_params": {
        'objective': 'reg:squarederror'
    },
    "num_trees": 7,
    "use_wandb": False
}



def main():
    args = vars(parse_args())
    t0 = pd.read_csv(args["t0"])
    t1 = pd.read_csv(args["t1"])

    other = {}
    for cur_name in ["v0", "v1", "p1"]:
        if args[cur_name] is not None:
            other[cur_name] = pd.read_csv(args[cur_name])
    
    batya_train(t0, t1, args["mp"], **other)#, **TRAIN_KWARGS)

if __name__ == "__main__":
    main()