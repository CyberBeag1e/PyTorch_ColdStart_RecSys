import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional

class ItemTower(nn.Module):
    """
    Item Tower of Two Tower model. Use text + image embeddings, and optionally item_id embedding as inputs
    """
    def __init__(self, in_dim: int, out_dim: int = 128, *, use_id: bool = False, n_items: Optional[int] = None, id_dim: int = 32):
        super().__init__()
        self.use_id = use_id
        if use_id:
            ## Optionally include item_id embeddings as input
            assert n_items is not None, "[Item Tower] `n_items` required in initialization when `use_id = True`."
            self.id_emb = nn.Embedding(n_items, id_dim)
            nn.init.normal_(self.id_emb.weight, std = 0.02)
            in_total = in_dim + id_dim
        
        else:
            in_total = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_total, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    
    def forward(self, x: torch.Tensor, *, item_idx: Optional[torch.Tensor] = None, id_mask: Optional[torch.Tensor] = None):
        if self.use_id:
            assert item_idx is not None, "[Item Tower] `item_idx` required in forward pass when `use_id = True`."
            id_vec = self.id_emb(item_idx)

            if id_mask is not None:
                ## Mask new items
                id_vec = id_vec * id_mask.unsqueeze(dim = -1).to(id_vec.dtype)
            
            x = torch.cat([x, id_vec], dim = -1)

        z = self.net(x)
        return F.normalize(z, dim = -1)


class UserTower(nn.Module):
    """
    User Tower of Two Tower Model.
    """
    def __init__(self, dim: int = 128, agg: Literal["mean", "attn"] = "attn"):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 128), 
            nn.ReLU(),
            nn.Linear(128, dim)
        )

        self.agg = agg
        if agg == "attn":
            self.W = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, 1, bias = False)
    
    def forward(self, hist: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.agg == "attn":
            h = torch.tanh(self.W(hist))
            scores = self.v(h).squeeze(dim = -1)

            if mask is not None:
                ## Mask padding positions
                scores = scores.masked_fill(~mask, -float('inf'))
            
            attn = torch.softmax(scores, dim = 1).unsqueeze(-1)
            pooled = (hist * attn).sum(dim = 1)

        else:
            
            if mask is not None:
                demon = mask.sum(dim = 1, keepdim = True).clamp_min(1.0).to(hist.dtype)
                pooled = (hist * mask.unsqueeze(dim = -1)).sum(dim = 1) / demon
            else:
                pooled = hist.mean(dim = 1)
        
        u = self.ffn(pooled)
        return F.normalize(u, dim = -1)