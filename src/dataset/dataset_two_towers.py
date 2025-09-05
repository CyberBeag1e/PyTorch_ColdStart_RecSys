import json
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from src.config.paths_config import P

class TwoTowerDataset(Dataset):
    """
    Select the last L interacted items (indices) for each user at each position, left-padded with -1.
    """
    def __init__(self, max_hist_len: int = 20):
        self.train = pd.read_parquet(P.TRAIN).sort_values("ts")
        with open(P.MAPPINGS, "r") as f:
            mappings = json.load(f)
        
        self.idx2item = [int(x) for x in mappings["idx2item"]]
        self.item2idx = {int(mID): i for i, mID in enumerate(self.idx2item)}

        self.train["item_idx"] = self.train["movieId"].astype(int).map(self.item2idx)
        self.user_hist = {}

        for u, group in self.train.groupby("userId"):
            seq = group["item_idx"].to_list()
            self.user_hist[int(u)] = seq
        
        self.samples = []
        self.max_hist_len = max_hist_len

        for u, seq in self.user_hist.items():
            for t in range(1, len(seq)):
                hist = seq[max(0, t - max_hist_len): t]
                if hist:
                    self.samples.append((u, hist, seq[t]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        u, hist, pos = self.samples[index]
        L = self.max_hist_len
        pad = [-1] * (L - len(hist)) + hist[-L:]

        return np.array(pad, dtype = np.int64), np.int64(pos)