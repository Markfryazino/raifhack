# Raifhack DS

This repository contains a project by the "Why are you painting that fence again?" team we created during the Raifhack DS hackathon. 
The hackathon's goal was to build a model for predicting the value of commercial real estate. We placed 28th out of nearly 400 teams.

To run the model, you need to do the following:

1. Go to [Eden](./Eden) directory.
```
cd eden
```

2. Set up environment using poetry

```
poetry install
```

3. Load data from W&B

```
poetry run python3 ../read_data.py
```

4. Run preprocessing

```
poetry run python3 preprocess.py -i data/split_data -o data/some_preprocessed_data
```

5. Train the model

```
poetry run python3 train.py -t0 <t0 path> -t1 <t1 path> -mp saved_models/baseline_changed.pkl
```

`<t0 path>` -- path to csv file with `price_type = 0`.

`<t1 path>` -- path to csv file with `price_type = 1`.

6. Predict
```
poetry run python3 predict.py -p1 <test path> -mp saved_models/baseline_changed.pkl -o submissions/baseline_changed.csv
```