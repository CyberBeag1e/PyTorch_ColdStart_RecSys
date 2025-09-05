import argparse
import json
import os
import pandas as pd
import yaml

from typing import Tuple, Literal, Dict, List
from src.config.paths_config import P

def pick_cutoff(df: pd.DataFrame, q: float, column_name: str = "ts") -> pd.Series[pd.Timestamp]:
    """
    Simulate the "new items" by selecting a cutoff within the timestamp. Treat items after the selected cutoff as new items.

    Args
        df: Data frame of interactions between users and items.
        q: The quantile of cutoff, i.e., the percentage of previous items (compared to new items) in the whole dataset.
        column_name: Column name for timestamp. Default "ts"
    """

    assert column_name in df.columns, f"{column_name} not in data frame. Please ensure you passed in the right column for timestamp."
    assert q >= 0.0 and q <= 1.0, f"q must be in the range [0, 1] (both inclusive)."
    ts = df[column_name].sort_values().values
    
    index = int(len(ts) * q) if q < 1.0 else len(ts) - 1
    return pd.to_datetime(ts[index])

def build_user_item_mappings(users_train: pd.Series, 
                             items_all: pd.Series) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    
    """
    Map userId to contiguous integer indices for efficient indexing. We only used users that appear in our training set since they are the users
    who witnessed the new items.

    Args
        user_train: Pandas Series of user index in the training set.
        items_all: Pandas Series of all item indices
    
    Returns
        `user2idx`: Original user indices to contiguous integer indices (from 0).
        `item2idx`: Original item indices to contiguous integer indices (from 0).
        `unique_users`: List of original indices of unique users.
        `unique_items`: List of original indices of unique items.
    """

    unique_users = pd.Index(users_train.unique(), name = "userId").astype(str)
    user2idx = {u: i for i, u in enumerate(unique_users.to_list())}

    unique_items = pd.Index(items_all.unique(), name = "movieId").astype(str)
    item2idx = {i: j for j, i in enumerate(unique_items.to_list())}

    return user2idx, item2idx, list(unique_users), list(unique_items)

def thresholding_users(train_df: pd.DataFrame, 
                       threshold: int) -> Tuple[pd.DataFrame, set]:
    
    """
    Drop users with insufficient interaction history from the training set.

    Args
        train_df: Training data
        threshold: Threshold of dropping.
    
    Returns
        train_df: Training data after dropping
        keep_users: Remaining users
    """
    cnt = train_df.groupby("userId").size()
    keep_users = cnt[cnt >= threshold].index

    return train_df[train_df["userId"].isin(keep_users)].copy(), set(keep_users.astype(str))

def main(cutoff_mode: Literal["quantile", "date"] = "quantile", 
         cutoff_value: str = "0.8", 
         val_share: float = 0.1, 
         interactions_threshold: int = 5, 
         print_info: bool = False):

    """
    Args
        cutoff_mode: By quantile or date, default quantile.
        cutoff_value: If cutoff by quantile, set this parameter to a float `q` within [0, 1], where `q` is the proportion of data
            that will be treated as pre-cutoff data. If cutoff by date, set this parameter to a stringified datetime, where the 
            interactions before this date will be treated as pre-cutoff data.
        val_share: How much post-cutoff data are treated as validation data.
        interactions_threshold: Users with number of records lower than this value will be dropped.
        print_info: Whether to print a summary of the split, default `False`.
    """
    
    os.makedirs(P.PROCESSED, exist_ok = True)

    items = pd.read_parquet(P.ITEMS)
    interactions = pd.read_parquet(P.INTERACTIONS)

    interactions["userId"] = interactions["userId"].astype(int)
    interactions["movieId"] = interactions["movieId"].astype(int)
    interactions["ts"] = pd.to_datetime(interactions["ts"])

    interactions = interactions[interactions["movieId"].isin(items["movieId"].unique())]
    interactions = interactions.sort_values("ts").reset_index(drop = True)

    ## Can choose between using a quantile or date to define new items
    if cutoff_mode == "quantile":
        q = float(cutoff_value)
        cutoff = pick_cutoff(interactions, q)
    else:
        cutoff = pd.to_datetime(cutoff_value)
    
    train = interactions[interactions["ts"] <= cutoff].copy()
    post = interactions[interactions["ts"] > cutoff].copy()

    if len(train) < 1000:
        raise ValueError(f"[Train-Val-Test Split] Not enough training data: {len(train)}. Try a higher quantile or earlier date.")

    train, keep_users = thresholding_users(train, interactions_threshold)
    post = post[post["userId"].astype(str).isin(keep_users)].copy()

    if len(post) == 0:
        raise ValueError(f"[Train-Val-Test Split] No post-cutoff interactions after thresholding. Adjust `interactions_threshold` or cutoff.")
    
    ## Select the first small proportion of post-cutoff data as validation set
    ## The rest of post-cutoff data is test set
    val_cutoff_idx = int(len(post) * val_share)
    val = post.iloc[:val_cutoff_idx].copy()
    test = post.iloc[val_cutoff_idx:].copy()

    item_first = interactions.groupby("movieId")["ts"].min()
    item_last = interactions.groupby("movieId")["ts"].max()

    items_meta = pd.DataFrame({
        "movieId": item_first.index.values,
        "first_ts": item_first.values,
        "last_ts": item_last.loc[item_first.index].values
    })

    prev_seen = train["movieId"].unique()
    items_meta["prev_seen"] = items_meta["movieId"].isin(prev_seen)
    items_meta["is_new"] = ~items_meta["prev_seen"]

    train.to_parquet(P.TRAIN, index = False)
    val.to_parquet(P.VAL, index = False)
    test.to_parquet(P.TEST, index = False)
    items_meta.to_parquet(P.ITEMS_META, index = False)

    user2idx, item2idx, idx2user, idx2item = build_user_item_mappings(train["movieId"], train["userId"], items["movieId"])

    with open(P.MAPPINGS, "w", encoding = "utf-8") as f:
        json.dump({
            "user2idx": user2idx,
            "item2idx": item2idx,
            "idx2user": idx2user,
            "idx2item": idx2item
        }, f)

    split_info = {
        "cutoff_mode"           : cutoff_mode,
        "cutoff_val"            : cutoff_value,
        "cutoff_ts"             : cutoff.isoformat(),
        "val_share"             : val_share,
        "interactions_threshold": interactions_threshold, 
        "n_train"               : int(len(train)), 
        "n_val"                 : int(len(val)), 
        "n_test"                : int(len(test)), 
        "n_user_train"          : int(train["userId"].nunique()), 
        "n_items_total"         : int(items["movieId"].nunique()),
        "n_new_items"           : int(items_meta["is_new"].sum()),
        "new_item_rate"         : float((items_meta["is_new"].mean()))
    }

    with open(os.path.join(P.PROCESSED, "split_info.yaml"), "w") as f:
        yaml.safe_dump(split_info, f)
    
    if print_info:
        print(f"[Train-Val-Test Split] Temporal Split Summary")
        for k, v in split_info.items():
            print(f"\t{k}: {v}")
    
    print(f"[Train-Val-Test Split] Saved splits & metadata to {P.PROCESSED}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff_mode", choices = ["quantile", "date"], default = "quantile")
    parser.add_argument("--cutoff_value", default = "0.8")
    parser.add_argument("--val_share", type = float, default = 0.1)
    parser.add_argument("--interactions_threshold", type = int, default = 5)
    parser.add_argument("--print_info", action = "store_true")

    args = parser.parse_args()

    main(**vars(args))
