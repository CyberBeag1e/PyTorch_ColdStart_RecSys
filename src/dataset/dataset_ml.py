import json
import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset
from src.config.paths_config import P

class PairDataset(Dataset):
    """
    Positive samples for a user: items that existed before the cutoff, with observed interactions before the cutoff.
    Negative samples for a user: items that existed before the cutoff, without observed interactions before the cutoff. (First interaction after the cutoff)
    """

    def __init__(self):
        self.train_df = pd.read_parquet(P.TRAIN)

        with open(P.MAPPINGS, "r") as f:
            mappings = json.load(f)
        
        self.user2idx = {int(k): v for k, v in mappings["user2idx"].items()}
        self.item2idx = {int(k): v for k, v in mappings["item2idx"].items()}

        ## Build positive samples
        self.user_pos = {}
        for r in self.train_df.itertuples(index = False):
            u, i = int(r.userId), int(r.movieId)
            if u not in self.user2idx or i not in self.item2idx:
                continue

            self.user_pos.setdefault(u, set()).add(i)
        
        self.samples = [(u, i) for u, pos in self.user_pos.items() for i in pos]
        
        ## Negatives are items that existed pre-cutoff
        if os.path.exists(P.ITEMS_META):
            meta = pd.read_parquet(P.ITEMS_META)
            prev_seen = set(meta.loc[meta["prev_seen"], "movieId"].astype(int).to_list())
        else:
            prev_seen = set(self.item2idx.keys())

        self.neg_pool = np.array(list(prev_seen), dtype = np.int64)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        For each positive pair, find a negative sample. Return (u, i_pos, i_neg)
        """
        u, i_pos = self.samples[index]

        while True:
            i_neg = int(self.neg_pool[np.random.randint(0, len(self.neg_pool))])
            if i_neg not in self.user_pos[u]:
                break
        
        return (self.user2idx[u], self.item2idx[i_pos], self.item2idx[i_neg])