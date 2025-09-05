import argparse
import numpy as np
import pandas as pd

from collections import Counter
from typing import List, Tuple

from src.config.paths_config import P
from src.eval.metrics import recall_at_k, nDCG_at_k

def build_global_popularity(train_df: pd.DataFrame) -> np.ndarray:
    """
    Build candidate list based on popularity
    """
    cnt = Counter(train_df["movieId"].astype(int).to_list())
    items = [it for it, _ in cnt.most_common()]

    return np.array(items, dtype = np.int64)

def get_user_gt(df: pd.DataFrame) -> Tuple[List[int], List[List[int]]]:
    """
    Get ground truths of users.
    """
    gt = df.groupby("userId")["movieId"].apply(lambda s: s.astype(int).to_list())
    users = gt.index.astype(int).to_list()
    lists = gt.to_list()

    return users, lists

def main(k: int):
    
    train = pd.read_parquet(P.TRAIN)
    test = pd.read_parquet(P.TEST)
    meta = pd.read_parquet(P.ITEMS_META)

    global_popularity = build_global_popularity(train)

    history = train.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int).to_list())).to_dict()

    users, gt_all = get_user_gt(test)
    # prev_seen = set(meta.loc[meta["prev_seen"], "movieId"].astype(int).to_list())
    items_new = set(meta.loc[~meta["prev_seen"], "movieId"].astype(int).to_list())

    rec_lists = []
    rec_lists_new = []
    rec_lists_seen = []

    for u in users:
        hist = history.get(u, set())
        candidates = [i for i in global_popularity if i not in hist]
        rec = candidates[:k]
        rec_lists.append(rec)

        ## Existing items with trained embeddings
        rec_lists_seen.append([i for i in rec if i not in items_new])

        ## New items without trained embeddings (random initialized embeddings)
        rec_lists_new.append([i for i in rec if i in items_new])
    
    ## Ground truths
    gt_seen = [[i for i in lst if i not in items_new] for lst in gt_all]
    gt_new = [[i for i in lst if i in items_new] for lst in gt_all]

    print(f"[Baseline Model] Popularity Baseline")
    print(f"    All  : Recall@{k} = {recall_at_k(gt_all, rec_lists, k):.4f}   nDCG@{k} = {nDCG_at_k(gt_all, rec_lists, k):.4f}")
    print(f"    Seen : Recall@{k} = {recall_at_k(gt_seen, rec_lists_seen, k):.4f}   nDCG@{k} = {nDCG_at_k(gt_seen, rec_lists_seen, k):.4f}")
    print(f"    New  : Recall@{k} = {recall_at_k(gt_new, rec_lists_new, k):.4f}   nDCG@{k} = {nDCG_at_k(gt_new, rec_lists_new, k):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type = int, default = 50)
    
    args = parser.parse_args()
    main(args.k)