import pandas as pd
from src.utils import is_outlier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
import joblib
import os

DATA_PATH = "./data/preprocessed"
COLS_W_OUTLIERS = [
    "price",
    "minimum_nights",
    "reviews_per_month",
    "calculated_host_listings_count",
]
SHOW_PERCENTILES = [0.5, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995]
POLY_ORDER = 2
FEATURES = [
    "neighbourhood_group",
    "latitude",
    "longitude",
    "room_type",
    "price",
    "minimum_nights",
    "calculated_host_listings_count",
    "availability_365",
    "nyc_tm_distance_km",
    "nyc_cp_distance_km",
    "nyc_fd_distance_km",
    "jfk_airport_distance_km",
    "lga_airport_distance_km",
    "ewr_airport_distance_km",
    "avg_airport_distance_km",
    "total_min_cost",
    "len_name",
    "nb_adj_in_name",
    "nb_adv_in_name",
    "nb_nouns_in_name",
    "nb_propn_in_name",
    "rate_adj_in_name",
    "rate_adv_in_name",
    "rate_nouns_in_name",
    "rate_propn_in_name",
]
CAT_FEATURES = [
    "neighbourhood_group",
    "room_type",
]
PROCESSOR_FILENAME = '../../models/preprocessor.joblib'


print(f"[DATA COLLECTION] Reading train and test from path: {DATA_PATH}")
df_train = pd.read_csv(os.path.join(DATA_PATH, "train_set.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test_set.csv"))
print("[DATA COLLECTION] Datasets loaded correctly, starting Feature Engineering...")
print(df_train[COLS_W_OUTLIERS].describe(percentiles=SHOW_PERCENTILES))

price_outlier_cond = is_outlier(df_train["price"], 99)
minimum_nights_outlier_cond = is_outlier(df_train["minimum_nights"], 99)
reviews_per_month_outlier_cond = is_outlier(df_train["reviews_per_month"], 99.5)
listing_count_outlier_cond = is_outlier(df_train["calculated_host_listings_count"], 99)
df_train = df_train.loc[
    price_outlier_cond
    & minimum_nights_outlier_cond
    & reviews_per_month_outlier_cond
    & listing_count_outlier_cond
].copy()

cont_features = [f for f in CAT_FEATURES if f not in CAT_FEATURES]
general_preprocessor = make_column_transformer(
    (PolynomialFeatures(degree=POLY_ORDER), cont_features),
    (StandardScaler(), cont_features),
    (
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
        CAT_FEATURES,
    ),
)
general_preprocessor.set_output(transform="pandas")

joblib.dump(general_preprocessor, PROCESSOR_FILENAME)
print("[DATA SPLIT] Successfully saved train and test datasets...")
