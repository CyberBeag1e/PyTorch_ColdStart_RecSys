import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Literal

from src.config.paths_config import P
from src.dataset.dataset_ml import PairDataset
from src.eval.utils import BPR_loss
from src.eval.metrics import recall_at_k, nDCG_at_k
from src.models.mf import MF_BPR

def evaluate(model: MF_BPR, 
             dataset: Literal["val", "test"], 
             *,
             k: int = 50, 
             device: Literal["cuda", "cpu"] = "cuda") -> Dict[str, Tuple[float, float]]:
    """
    Evaluation of Matrix factorization model with BPR loss.
    """
    
    with open(P.MAPPINGS, "r") as f:
        mappings = json.load(f)
    
    user2idx = {int(k): v for k, v in mappings["user2idx"].items()}
    idx2item = [int(x) for x in mappings["idx2item"]]

    split_path = P.VAL if dataset == "val" else P.TEST
    df = pd.read_parquet(split_path)
    meta = pd.read_parquet(P.ITEMS_META)
    items_new = set(meta.loc[~meta["prev_seen"], "movieId"].astype(int).to_list())

    gt = df.groupby("userId")["movieId"].apply(lambda s: s.astype(int).to_list())
    users = [u for u in gt.index if u in user2idx]
    gt_all = [gt[u] for u in users]

    model.eval()
    rec_lists = []
    rec_lists_seen, rec_lists_new = [], []
    
    with torch.no_grad():
        for u in users:
            ui = torch.tensor([user2idx[u]], device = device)
            scores = model.predict_user(ui).squeeze(0).detach().cpu().numpy()

            ## Rank by scores (top-k)
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

            rec = [int(idx2item[j]) for j in topk_idx]
            rec_lists.append(rec)
            rec_lists_seen.append([i for i in rec if i not in items_new])
            rec_lists_new.append([i for i in rec if i in items_new])
    
    gt_seen = [[i for i in lst if i not in items_new] for lst in gt_all]
    gt_new = [[i for i in lst if i in items_new] for lst in gt_all]

    return {
        "All": (recall_at_k(gt_all, rec_lists, k), nDCG_at_k(gt_all, rec_lists, k)), 
        "Seen": (recall_at_k(gt_seen, rec_lists_seen, k), nDCG_at_k(gt_seen, rec_lists_seen, k)),
        "New": (recall_at_k(gt_new, rec_lists_new, k), nDCG_at_k(gt_new, rec_lists_new, k))
    }

def main(n_epochs: int = 10, 
         batch_size: int = 2048, 
         dim: int = 64, 
         lr: float = 5e-3,
         l2: float = 1e-6, 
         device: Literal["cuda", "cpu"] = "cuda", 
         k: int = 50) -> None:
    
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[Baseline Model] Device error: CUDA is not available")
            return

    dataset = PairDataset()
    with open(P.MAPPINGS, "r") as f:
        mappings = json.load(f)
    
    n_users = len(mappings["idx2user"])
    n_items = len(mappings["idx2item"])
    model = MF_BPR(n_users, n_items, dim = dim, l2 = l2).to(device)
    optimizer = Adam(model.parameters(), lr = lr)

    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []

        for u, i, j in tqdm(data_loader, desc = f"Epoch {epoch}"):
            u, i, j = u.to(device), i.to(device), j.to(device)
            x, pu, qi, qj = model(u, i, j)
            loss = BPR_loss(x, (pu, qi, qj), model.l2)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        print(f"[Epoch {epoch}] Training loss = {np.mean(losses):.4f}")
    
    for split_name in ["val", "test"]:
        res = evaluate(model, 
                       split_name, 
                       k = k, 
                       device = device)

        (r_all, n_all), (r_seen, n_seen), (r_new, n_new) = res["All"], res["Seen"], res["New"]
        print(f"[Baseline Model] MF Baseline with BPR Loss: {split_name.capitalize()}")
        print(f"    All  : Recall@{k} = {r_all:.4f}   nDCG@{k} = {n_all:.4f}")
        print(f"    Seen : Recall@{k} = {r_seen:.4f}   nDCG@{k} = {n_seen:.4f}")
        print(f"    New  : Recall@{k} = {r_new:.4f}   nDCG@{k} = {n_new:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 2048)
    parser.add_argument("--dim", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 5e-3)
    parser.add_argument("--l2", type = float, default = 1e-6)
    parser.add_argument("--device", default = "cuda")
    parser.add_argument("--k", type = int, default = 50)
    

    args = parser.parse_args()
    main(args.epochs, 
         args.batch_size,
         args.dim, 
         args.lr, 
         args.l2, 
         args.device, 
         args.k)