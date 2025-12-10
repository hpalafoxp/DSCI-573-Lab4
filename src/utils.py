import pandas as pd
import numpy as np

def custom_train_test_split(dataframe: pd.DataFrame, cat_col: str, test_size: float = 0.3, random_state: int = 573):
    np.random.seed(random_state)

    df: pd.DataFrame = dataframe.copy()

    cat_ids: np.ndarray = df[cat_col].unique()

    random_shuffling: dict = {cat:np.random.rand() for cat in cat_ids}

    df["ordered_cat"] = df[cat_col].map(random_shuffling)

    df = df.sort_values("ordered_cat")

    cum_count = 0

    for cat, count in df.groupby(cat_col, sort=False)[cat_col].agg("count").items():
        cum_count += count
        if cum_count / len(df) >= 1 - test_size:
            cutoff = cat
            break
            
    df = df.drop("ordered_cat", axis=1).set_index(cat_col)

    train_df = df.loc[:cutoff].iloc[:-1].reset_index()
    test_df = df.loc[cutoff:].reset_index()

    return train_df, test_df
