import pandas as pd
from src.utils import compute_feature_engineering
import os

DATA_PATH = "./data/split_data"
SAVE_DATA_PATH = "./data/preprocessed"


print(f"[DATA COLLECTION] Reading train and test from path: {DATA_PATH}")
df_train = pd.read_csv(os.path.join(DATA_PATH, "train_set.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test_set.csv"))
print("[DATA COLLECTION] Datasets loaded correctly, starting Feature Engineering...")
df_train = compute_feature_engineering(df_train)
print("\n\nFinished TRAIN feature engineering...\n")
df_test = compute_feature_engineering(df_test)
print("\n\nFinished TEST feature engineering...")
print(f"[DATA SPLIT] Saving train and test in: {SAVE_DATA_PATH}")
train_path = os.path.join(SAVE_DATA_PATH, "fe_train_set.csv")
df_train.to_csv(train_path, index=False)
test_path = os.path.join(SAVE_DATA_PATH, "fe_test_set.csv")
df_test.to_csv(test_path, index=False)
print("[DATA SPLIT] Successfully saved train and test datasets...")