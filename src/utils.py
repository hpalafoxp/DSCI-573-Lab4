import pandas as pd
import numpy as np
import spacy
import en_core_web_md


def custom_train_test_split(
    dataframe: pd.DataFrame,
    cat_col: str,
    test_size: float = 0.3,
    random_state: int = 573,
):
    np.random.seed(random_state)
    df = dataframe.copy()
    cat_ids = df[cat_col].unique()
    random_shuffling = {cat: np.random.rand() for cat in cat_ids}
    df["ordered_cat"] = df[cat_col].map(random_shuffling)
    df = df.sort_values("ordered_cat")

    cum_count = 0
    cat_counts = df.groupby(cat_col, sort=False)[cat_col].agg("count").items()
    for cat, count in cat_counts:
        cum_count += count
        if cum_count / len(df) >= 1 - test_size:
            cutoff = cat
            break

    df = df.drop("ordered_cat", axis=1).set_index(cat_col)
    train_df = df.loc[:cutoff].iloc[:-1].reset_index()
    test_df = df.loc[cutoff:].reset_index()

    return train_df, test_df


def inhouse_haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance using the Haversine formula.
    Works with scalars or NumPy arrays.
    Distance is returned in kilometers.
    """

    R = 6371.0  # Earth radius in km

    # Convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_feature_engineering(data):
    # Times square
    nycts_latitude = 40.7589
    nycts_longitude = -73.9851
    # Finantial district
    nyc_fd_latitude = 40.7077
    nyc_fd_longitude = -74.0083
    # Central Park
    nyccp_latitude = 40.7826
    nyccp_longitude = -73.9656
    # JFK Airport
    jfk_latitude = 40.6446
    jfk_longitude = -73.7797
    # EWR Airport
    ewr_latitude = 40.6885
    ewr_longitude = -74.1769
    # LGA Airport
    lga_latitude = 40.7766
    lga_longitude = -73.8742
    # create NLP object
    nlp = spacy.load("en_core_web_md", disable=["ner", "parser", "lemmatizer"])
    # Compute features
    data["name"] = data["name"].astype(str)
    print("COMPUTING FEATURE: nyc_tm_distance_km")
    data["nyc_tm_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], nycts_latitude, nycts_longitude
    )
    print("COMPUTING FEATURE: nyc_cp_distance_km")
    data["nyc_cp_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], nyccp_latitude, nyccp_longitude
    )
    print("COMPUTING FEATURE: nyc_fd_distance_km")
    data["nyc_fd_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], nyc_fd_latitude, nyc_fd_longitude
    )
    print("COMPUTING FEATURE: jfk_airport_distance_km")
    data["jfk_airport_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], jfk_latitude, jfk_longitude
    )
    print("COMPUTING FEATURE: lga_airport_distance_km")
    data["lga_airport_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], lga_latitude, lga_longitude
    )
    print("COMPUTING FEATURE: ewr_airport_distance_km")
    data["ewr_airport_distance_km"] = inhouse_haversine(
        data["latitude"], data["longitude"], ewr_latitude, ewr_longitude
    )
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["avg_airport_distance_km"] = data[
        [
            "jfk_airport_distance_km",
            "lga_airport_distance_km",
            "ewr_airport_distance_km",
        ]
    ].mean(axis=1)
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["total_min_cost"] = data["price"] * data["minimum_nights"]
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["len_name"] = data["name"].apply(lambda x: len(x))
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["nb_adj_in_name"] = data["name"].apply(
        lambda x: sum([1 for t in nlp(x) if t.pos_ == "ADJ"])
    )
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["nb_adv_in_name"] = data["name"].apply(
        lambda x: sum([1 for t in nlp(x) if t.pos_ == "ADV"])
    )
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["nb_nouns_in_name"] = data["name"].apply(
        lambda x: sum([1 for t in nlp(x) if t.pos_ == "NOUN"])
    )
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["nb_propn_in_name"] = data["name"].apply(
        lambda x: sum([1 for t in nlp(x) if t.pos_ == "PROPN"])
    )
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["rate_adj_in_name"] = data["nb_adj_in_name"] / data["len_name"]
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["rate_adv_in_name"] = data["nb_adv_in_name"] / data["len_name"]
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["rate_nouns_in_name"] = data["nb_nouns_in_name"] / data["len_name"]
    print("COMPUTING FEATURE: avg_airport_distance_km")
    data["rate_propn_in_name"] = data["nb_propn_in_name"] / data["len_name"]
    return data


def is_outlier(x, p):
    return x < np.percentile(x, p)


def get_important_features_important_than(shap_values, features, random_feats):
    # Compute mean absolute SHAP values for each feature
    shap_importances = np.mean(np.abs(shap_values), axis=0)

    # Create a list of tuples (feature name, importance)
    feature_importance = list(zip(features, shap_importances))

    # Sort the list by importance
    sorted_feature_importance = sorted(
        feature_importance, key=lambda x: x[1], reverse=True
    )

    # Print sorted feature importance
    feature_importance_dict = {}
    i = 1
    for feature, importance in sorted_feature_importance:
        feature_importance_dict[f"{feature}"] = i
        i += 1
    rand_min_importance = np.max(
        [
            feature_importance_dict[random_feats[0]],
            feature_importance_dict[random_feats[1]],
        ]
    )
    important_enough_features = [
        v[0]
        for v in list(feature_importance_dict.items())
        if v[1] < rand_min_importance
        and v[0] != random_feats[0]
        and v[0] != random_feats[1]
    ]
    return sorted_feature_importance, important_enough_features
