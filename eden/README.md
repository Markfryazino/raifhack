# Алгоритм работы со скриптами

1. Настраиваем окружение

```
poetry install
```

2. Загружаем данные 

```
poetry run python3 ../read_data.py
```

3. Препроцессим данные

```
poetry run python3 preprocess.py -i data/split_data -o data/some_preprocessed_data
```

4. Тренируем модель

```
poetry run python3 train.py -t0 <t0 path> -t1 <t1 path> -mp saved_models/baseline_changed.pkl
```

`<t0 path>` -- путь к csv с трейном, где у всех строк `price_type = 0`.

`<t1 path>` -- путь к csv с трейном, где у всех строк `price_type = 1`.

5. Предиктим модель

```
poetry run python3 predict.py -p1 <test path> -mp saved_models/baseline_changed.pkl -o submissions/baseline_changed.csv
```