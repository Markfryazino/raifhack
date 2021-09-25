import argparse
import pandas as pd
import os
import wandb
import logging

from preprocessing import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, dest="i", required=True)
    parser.add_argument("--output_folder", "-o", type=str, dest="o", required=True)

    return parser.parse_args()

def preprocess(t0, t1, p1):
    logging.info("common preprocessing p1")
    p1 = common.preprocessing(p1)
    logging.info("common preprocessing t1")
    t1 = common.preprocessing(t1)
    logging.info("common preprocessing t0")
    t0 = common.preprocessing(t0)
    return t0, t1, p1

def main():
    args = vars(parse_args())
    t0 = pd.read_csv(os.path.join(args["i"] + "/" + "t0.csv"))
    t1 = pd.read_csv(os.path.join(args["i"] + "/" + "t1.csv"))
    p1 = pd.read_csv(os.path.join(args["i"] + "/" + "p1.csv"))
    t0, t1, p1 = preprocess(t0, t1, p1)
    if not os.path.exists(args["o"]):
        os.mkdir(args["o"])
    logging.info("saving results")
    t0.to_csv(os.path.join(args["o"] + "/" + "t0.csv"), index=False)
    t1.to_csv(os.path.join(args["o"] + "/" + "t1.csv"), index=False)
    p1.to_csv(os.path.join(args["o"] + "/" + "p1.csv"), index=False)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %I:%M:%S"
    )
    main()
    

