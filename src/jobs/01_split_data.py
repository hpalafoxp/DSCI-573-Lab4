import pandas as pd
import numpy as np
from src.utils import custom_train_test_split
import os

DATA_PATH = "./data/AB_NYC_2019.csv"
SAVE_DATA_PATH = "./data/split_data"
TARGET = "reviews_per_month"
RANDOM_STATE = 42
TEST_SIZE = 0.3
NOT_USE_COLS = [
    "id",
    "host_id",
    "host_name",
    "last_review",
]
np.random.seed(RANDOM_STATE)


print(f"[DATA COLLECTION] Reading csv file from path: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("[DATA COLLECTION] Drop missing values in target...")
df = df.dropna(subset=[TARGET])
print(
    f"[DATA COLLECTION] Dataset contains {df.shape[0]} rows and {df.shape[1]} columns..."
)
print(
    f"[DATA SPLIT] Split dataset into train ({1 - TEST_SIZE}) and test ({TEST_SIZE}) by user randomization..."
)
df_train, df_test = custom_train_test_split(df, "host_id", test_size=TEST_SIZE)
print("PCT of rows in train:", 100.0 * len(df_train) / len(df))
print("PCT of rows in test:", 100.0 * len(df_test) / len(df))

use_cols = [c for c in df if c not in NOT_USE_COLS]
df_train = df_train[use_cols].copy()
df_test = df_test[use_cols].copy()
print(f"[DATA SPLIT] Saving train and test in: {SAVE_DATA_PATH}")
train_path = os.path.join(SAVE_DATA_PATH, "train_set.csv")
df_train.to_csv(train_path, index=False)
test_path = os.path.join(SAVE_DATA_PATH, "test_set.csv")
df_test.to_csv(test_path, index=False)
print("[DATA SPLIT] Successfully saved train and test datasets...")
