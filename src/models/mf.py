import torch
import torch.nn as nn

class MF_BPR(nn.Module):
    """
    Matrix Factorization model with BPR Loss
    """
    def __init__(self, n_users: int, n_items: int, dim: int = 64, l2: float = 1e-6):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std = 0.01)
        nn.init.normal_(self.item.weight, std = 0.01)

        self.l2 = l2
    
    def forward(self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        pu = self.user(u)       ## user embedding
        qi = self.item(i)       ## positive item embedding
        qj = self.item(j)       ## negative item embedding

        x = (pu * (qi - qj)).sum(dim = 1)

        return x, pu, qi, qj

    def predict_user(self, u: torch.Tensor):
        pu = self.user(u)
        all_items = self.item.weight.t()
        scores = torch.matmul(pu, all_items)

        return scores