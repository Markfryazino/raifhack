import pandas as pd

floor_from_str = {
 'подвал': -1,
 'цоколь, 1': 1,
 '1,2,антресоль': 2,
 'цоколь': 0,
 'тех.этаж (6)': 6,
 'Подвал': -1,
 'Цоколь': 0,
 'фактически на уровне 1 этажа': 1,
 '1,2,3': 2,
 '1, подвал': 0,
 '1,2,3,4': 2,
 '1,2': 1,
 '1,2,3,4,5': 3,
 '5, мансарда': 5,
 '1-й, подвал': 0,
 '1, подвал, антресоль': 0,
 'мезонин': 2,
 'подвал, 1-3': 2,
 '1 (Цокольный этаж)': 0,
 '3, Мансарда (4 эт)': 3,
 'подвал,1': 0,
 '1, антресоль': 0,
 '1-3': 2,
 'мансарда (4эт)': 4,
 '1, 2.': 1,
 'подвал , 1 ': 0,
 '1, 2': 1,
 '1.2': 1,
 'подвал, 1,2,3': 2,
 '1 + подвал (без отделки)': 0,
 'мансарда': 1,
 '2,3': 2,
 '4, 5': 4,
 '1-й, 2-й': 1,
 '1 этаж, подвал': 1,
 '1, цоколь': 1,
 'подвал, 1-7, техэтаж': 3,
 '3 (антресоль)': 3,
 '1, 2, 3': 2,
 'Цоколь, 1,2(мансарда)': 1,
 'подвал, 3. 4 этаж': 3,
 'подвал, 1-4 этаж': 2,
 'подва, 1.2 этаж': 1,
 '2, 3': 2,
 '7,8': 7,
 '1 этаж': 1,
 '1-й': 1,
 '3 этаж': 3,
 '4 этаж': 4,
 '5 этаж': 5,
 'подвал,1,2,3,4,5': 3,
 'подвал, цоколь, 1 этаж': 1,
 '3, мансарда': 3,
 'подвал, 1': 0
}

def apply_floor(x):
    if x in floor_from_str:
        x = floor_from_str[x]
    try:
        x = float(x)
    except:
        x = 228
    return x

def fill_median_by_category(df, fill_column, category_column):
    df[fill_column] = df.groupby(category_column).transform(lambda x: x.fillna(x.median()))
    return df

def fill_mean_by_cluster(df, fill_column, cluster_column):
    df[fill_column] = df.groupby(cluster_column).transform(lambda x: x.fillna(x.mean()))
    return df

def floor_update(df):
    df["floor"] = df.floor.apply(apply_floor)
    df = df[~(df.floor > 100)]
    df = fill_mean_by_cluster(df, "floor", "city")
    df["floor"] = df.floor.fillna(df.floor.mean())
    return df

def preprocessing(df):
    df = floor_update(df)

    df = df[df["osm_city_nearest_population"].notna()]

    df['reform_house_population_1000'].fillna(df['reform_house_population_500'],inplace=True)
    df['reform_house_population_500'].fillna(df['reform_house_population_1000'],inplace=True)
    df["reform_house_population_1000"].fillna(0,inplace=True)
    df["reform_house_population_500"].fillna(1,inplace=True)

    df['reform_mean_floor_count_1000'].fillna(df['reform_mean_floor_count_500'],inplace=True)
    df['reform_mean_floor_count_500'].fillna(df['reform_mean_floor_count_1000'],inplace=True)
    df["reform_mean_floor_count_1000"].fillna(1,inplace=True)
    df["reform_mean_floor_count_500"].fillna(1,inplace=True)

    df['reform_mean_year_building_1000'].fillna(df['reform_mean_year_building_500'],inplace=True)
    df['reform_mean_year_building_500'].fillna(df['reform_mean_year_building_1000'],inplace=True)
    df = fill_median_by_category(df, "reform_mean_year_building_1000", "region")
    df = fill_median_by_category(df, "reform_mean_year_building_500", "region")

    df["street"] = df["street"].fillna('missing')

    return df
