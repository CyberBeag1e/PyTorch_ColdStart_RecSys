import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

def BPR_loss(x: torch.Tensor, reg_terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], l2: float):
    """
    Compute BPR Loss with L2 normalization.

    Args
        x: (u * i_pos - u * i_neg)
        reg_terms: tuple of (user_embeeding, positive_item_embedding, negative_item_embedding)
        l2: numerical scalar at L2 regularizer.
    """
    pu, qi, qj = reg_terms
    loss = - F.logsigmoid(x).mean()
    penalty = (pu.pow(2).sum(dim = 1) + qi.pow(2).sum(dim = 1) + qj.pow(2).sum(dim = 1)).mean() * l2

    return loss + penalty

def InfoNCE(u: torch.Tensor, 
            i_pos: torch.Tensor, 
            i_all: torch.Tensor, 
            temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    InfoNCE loss with in-batch negatives. For each user u, treat all positive items of the other users as negatives.
    """
    
    pos = (u * i_pos).sum(dim = -1, keepdim = True)
    neg = torch.matmul(u, i_all.T)

    logits = neg / temperature
    idx = torch.arange(u.size(dim = 0), device = u.device)
    logits[idx, idx] = (pos.squeeze(dim = -1) / temperature).to(logits.dtype)
    labels = torch.arange(u.size(dim = 0), device = u.device)

    return logits, labels

def collate_hist(batch: List[Tuple[np.ndarray, int]]):
    h_arr = np.stack([b[0] for b in batch], axis = 0)
    p_arr = np.stack([b[1] for b in batch], dtype = np.int64)

    h = torch.from_numpy(h_arr).long()
    p = torch.from_numpy(p_arr).long()

    return h, p