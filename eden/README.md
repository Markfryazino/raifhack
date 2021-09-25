# Запуск сабмита 1

`<t0 path>` -- путь к csv с трейном, где у всех строк `price_type = 0`.

`<t1 path>` -- путь к csv с трейном, где у всех строк `price_type = 1`.

```
poetry install
poetry run python3 train.py -t0 <t0 path> -t1 <t1 path> -mp saved_models/baseline_changed.pkl
poetry run python3 predict.py -p1 <test path> -mp saved_models/baseline_changed.pkl -o submissions/baseline_changed.csv
```