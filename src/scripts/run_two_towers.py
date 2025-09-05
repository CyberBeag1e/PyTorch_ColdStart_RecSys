import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Tuple, Literal, Optional

from src.config.paths_config import P
from src.dataset.dataset_two_towers import TwoTowerDataset
from src.eval.metrics import recall_at_k, nDCG_at_k
from src.eval.utils import InfoNCE, collate_hist
from src.models.two_tower import ItemTower, UserTower


def load_item_features(use_text: bool = True, 
                       use_img: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load item features as input to the item tower.

    Args
        use_text: Whether to use textual embeddings. Default `True`.
        use_img: Whether to use image embeddings. Default `True`.

    Returns
        items: Data frame of item features
        X: Concatenated textual embeddings and image embeddings.
    """
    
    items = pd.read_parquet(P.ITEMS).reset_index(drop = True)
    features = []

    if use_text and os.path.exists(P.TEXT_EMB):
        text = np.load(P.TEXT_EMB).astype(np.float32)
        features.append(text)
    
    if use_img and os.path.exists(P.IMG_EMB):
        img = np.load(P.IMG_EMB).astype(np.float32)
        features.append(img)
    
    X = np.concatenate(features, axis = 1)
    return items, X

@torch.no_grad()
def build_item_embeddings(item_tower: ItemTower, 
                          item_features: np.ndarray, 
                          *,
                          item_indices: Optional[torch.Tensor] = None,
                          id_mask: Optional[torch.Tensor] = None,
                          device: Literal["cuda", "cpu"] = "cuda", 
                          batch_size: int = 1024) -> torch.Tensor:
    """
    Build batched item embeddings from the item tower.

    Args
        item_tower: Item Tower instance.
        item_features: Concatenated text embeddings and image embeddings.
        item_indices: Optional. Item indices for mapping item id embeddings.
        id_mask: Optional. 1 for SEEN items and 0 for NEW items.
        device: Make sure the tensors for computation are on the same device. Default CUDA.
        batch_size: Default 1024.
    """
    embs = []
    for i in range(0, item_features.shape[0], batch_size):
        xb = torch.from_numpy(item_features[i: i + batch_size]).to(device)
        id_xb = item_indices[i: i + batch_size] if item_indices is not None else None
        mask_xb = id_mask[i: i + batch_size] if id_mask is not None else None
        zb = item_tower(xb, item_idx = id_xb, id_mask = mask_xb)
        embs.append(zb.detach())
    
    return torch.cat(embs, dim = 0)

def evaluate(user_tower: UserTower, 
             item_tower: ItemTower, 
             item_features: np.ndarray, 
             dataset: Literal["val", "test"], 
             k: int = 50, 
             device: Literal["cuda", "cpu"] = "cuda"):
    
    eval_path = P.VAL if dataset == "val" else P.TEST
    df = pd.read_parquet(eval_path)
    with open(P.MAPPINGS, "r") as f:
        mappings = json.load(f)
    
    idx2item = [int(x) for x in mappings["idx2item"]]
    item2idx = {int(mID): i for i, mID in enumerate(idx2item)}
    meta = pd.read_parquet(P.ITEMS_META)

    items_new = set(meta.loc[~meta["prev_seen"], "movieId"].astype(int).to_list())
    id_mask_np = np.array([0. if mID in items_new else 1.0 for mID in idx2item], dtype = np.float32)
    id_mask = torch.from_numpy(id_mask_np).to(device)
    item_indices = torch.arange(len(idx2item), device = device, dtype = torch.long)

    item_embs = build_item_embeddings(item_tower, item_features, item_indices = item_indices, id_mask = id_mask, device = device)
    
    train = pd.read_parquet(P.TRAIN).sort_values("ts")
    train["item_idx"] = train["movieId"].astype(int).map(item2idx)

    max_hist_len = 20
    hist_by_user = train.groupby("userId")["item_idx"].apply(lambda s: s.dropna().astype(int).to_list()).to_dict()

    gt = df.groupby("userId")["movieId"].apply(lambda s: s.astype(int).to_list())
    users = [u for u in gt.index if u in hist_by_user]
    gt_all = [gt[u] for u in users]
    gt_seen = [[i for i in lst if i not in items_new] for lst in gt_all]
    gt_new = [[i for i in lst if i in items_new] for lst in gt_all]

    rec_lists = []
    rec_lists_seen, rec_lists_new = [], []

    with torch.no_grad():
        for u in users:
            hist = hist_by_user[u][-max_hist_len:]
            if not hist:
                rec_lists.append([])
                rec_lists_seen.append([])
                rec_lists_new.append([])
                continue
            
            hist_idx = torch.tensor([[-1] * (max_hist_len - len(hist)) + hist], device = device)
            mask = (hist_idx >= 0)
            hist_emb = item_embs[hist_idx.clamp(min = 0).squeeze(dim = 0)].unsqueeze(dim = 0)
            user_emb = user_tower(hist_emb, mask = (mask)).squeeze(dim = 0)

            scores = torch.mv(item_embs, user_emb)
            topk = torch.topk(scores, k = k).indices.tolist()
            rec = [idx2item[j] for j in topk]
            rec_lists.append(rec)
            rec_lists_seen.append([i for i in rec if i not in items_new])
            rec_lists_new.append([i for i in rec if i in items_new])
    
    res = {
        "All": (recall_at_k(gt_all, rec_lists, k), nDCG_at_k(gt_all, rec_lists, k)), 
        "Seen": (recall_at_k(gt_seen, rec_lists_seen, k), nDCG_at_k(gt_seen, rec_lists_seen, k)),
        "New": (recall_at_k(gt_new, rec_lists_new, k), nDCG_at_k(gt_new, rec_lists_new, k))
    }

    return res

def main(max_hist_len: int = 20, 
         out_dim: int = 128, 
         use_items_id: bool = True,
         items_id_dim: int = 32,
         batch_size: int = 1024, 
         n_epochs: int = 10, 
         lr: float = 1e-3, 
         decay: float = 1e-5, 
         temperature: float = 1.0,
         use_text: bool = True, 
         use_img: bool = True, 
         device: Literal["cuda", "cpu"] = "cuda", 
         k: int = 50):
    
    ## Force the model to run on GPU
    if not torch.cuda.is_available():
        print("[Two Tower Model] Device error: CUDA is not available")
        return
    
    _, item_features = load_item_features(use_text, use_img)
    in_dim = item_features.shape[1]

    dataset = TwoTowerDataset(max_hist_len)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_hist, drop_last = True)

    n_items = item_features.shape[0]
    item_tower = ItemTower(in_dim, out_dim, use_id = use_items_id, n_items = n_items, id_dim = items_id_dim).to(device)
    user_tower = UserTower(dim = out_dim, agg = "attn").to(device)

    optimizer = AdamW(list(item_tower.parameters()) + list(user_tower.parameters()), 
                      lr = lr, weight_decay = decay)
    
    scaler = GradScaler("cuda", enabled = (device == "cuda"))

    best_val = -1.0
    os.makedirs("experiments", exist_ok = True)
    model_path = "experiments/two_tower.pt"
    for epoch in range(1, n_epochs + 1):
        item_tower.train()
        user_tower.train()
        losses = []

        tqdm_bar = tqdm(dataloader, desc = f"Epoch {epoch}")
        for hist_idx, pos_idx in tqdm_bar:
            xb = torch.from_numpy(item_features[pos_idx.numpy()]).to(device)
            pos_idx = pos_idx.to(device)

            with autocast("cuda", enabled = (device == "cuda")):
                i_pos = item_tower(xb, item_idx = pos_idx)
                h_idx = torch.clamp(hist_idx, min = 0)
                # h_idx[h_idx < 0] = 0

                hist_features = torch.from_numpy(item_features[h_idx.numpy()]).to(device)
                flat_features = hist_features.view(-1, in_dim)
                flat_idx = h_idx.view(-1).to(device)
                pad_mask = (hist_idx >= 0).to(device)

                flat_emb = item_tower(flat_features, item_idx = flat_idx, id_mask = (flat_idx > 0))
                hist_emb = flat_emb.view(hist_features.size(dim = 0), hist_features.size(dim = 1), -1)

                user_emb = user_tower(hist_emb, mask = pad_mask)
                logits, labels = InfoNCE(user_emb, i_pos, i_pos, temperature = temperature)
                loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(list(item_tower.parameters()) + list(user_tower.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            tqdm_bar.set_postfix(loss = np.mean(losses))
        
        item_tower.eval()
        user_tower.eval()
        val_res = evaluate(user_tower, item_tower, item_features, dataset = "val", k = k, device = device)
        (r_all, n_all), (r_seen, n_seen), (r_new, n_new) = val_res["All"], val_res["Seen"], val_res["New"]
        print(f"[Two Tower Model] Validation : Recall@{k}   All = {r_all:.4f}; Seen = {r_seen:.4f}; New = {r_new:.4f}")
        
        if r_new > best_val:
            best_val = r_new
            
            torch.save({
                "item_tower": item_tower.state_dict(),
                "user_tower": user_tower.state_dict(),
                "in_dim": in_dim,
                "out_dim": out_dim
            }, model_path)
        
            print(f"[Two Tower Model] Saved checkpoint (New Recall@{k} = {r_new:.4f})")
    
    check_point = torch.load(model_path, map_location = device)
    item_tower.load_state_dict(check_point["item_tower"])
    user_tower.load_state_dict(check_point["user_tower"])
    test_res = evaluate(user_tower, item_tower, item_features, dataset = "test", k = k, device = device)
    (r_all, n_all), (r_seen, n_seen), (r_new, n_new) = test_res["All"], test_res["Seen"], test_res["New"]
    print(f"[Two Tower Model] Test : Recall@{k}   All = {r_all:.4f}; Seen = {r_seen:.4f}; New = {r_new:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hist_len", type = int, default = 20)
    parser.add_argument("--out_dim", type = int, default = 128)
    parser.add_argument("--use_items_id", action = "store_true", default = True)
    parser.add_argument("--items_id_dim", type = int, default = 32)
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--n_epochs", type = int, default = 10)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--decay", type = float, default = 1e-5)
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--k", type = int, default = 50)
    parser.add_argument("--use_text", action = "store_true", default = True)
    parser.add_argument("--use_img", action = "store_true", default = True)
    
    args = parser.parse_args()
    main(**vars(args))