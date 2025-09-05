import numpy as np
from typing import List

def recall_at_k(gt_items: List[List[int]], rec_items: List[List[int]], k: int = 50) -> float:
    """
    Recall@k

    ### Args
        gt_items (`List[List[int]]`) : ground truth items of users, i.e., the items that each user has actually interacted with.
        rec_items (`List[List[int]]`) : recommended items of users
        k (`int`): k in Recall@k
    """
    hits = 0
    total = 0

    for gt, rec in zip(gt_items, rec_items):
        if not gt:
            continue

        total += len(gt)
        hits += sum(1 for g in gt if g in set(rec[:k]))
    
    return hits / max(total, 1)


def nDCG_at_k(gt_items: List[List[int]], rec_items: List[List[int]], k: int = 50) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG) @k

    ### Args
        gt_items (`List[List[int]]`) : ground truth items of users, i.e., the items that each user has actually interacted with.
        rec_items (`List[List[int]]`) : recommended items of users
        k (`int`): k in nDCG@k
    """

    def DCG(scores: List[int | float]) -> float:
        return sum(r / np.log2(i + 2) for i, r in enumerate(scores))
    
    nDCGs = []
    for gt, rec in zip(gt_items, rec_items):
        if not gt:
            continue

        gt_set = set(gt)

        ## Only use a binary relevance
        scores = [1 if i in gt_set else 0 for i in rec[:k]]
        ideal = sorted(scores, reverse = True)
        iDCG = DCG(ideal) or 1.0

        nDCGs.append(DCG(scores) / iDCG)
    
    return float(np.mean(nDCGs)) if nDCGs else 0.0