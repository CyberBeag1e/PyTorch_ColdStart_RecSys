import argparse
import numpy as np
import os
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from typing import Iterable, Literal

from src.config.paths_config import P

def generate_chunks(iterable: Iterable, n: int):
    """
    Generate chunked data
    """
    for i in range(0, len(iterable), n):
        yield iterable[i: i + n]

def main(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
         batch_size: int = 128, 
         device: Literal["cuda", "cpu"] = "cuda"):
    
    """
    Pre-compute text (textual overviews, titles) embeddings using SentenceBERT.
    """
    
    os.makedirs(P.INTERIM, exist_ok = True)
    df = pd.read_parquet(P.ITEMS).reset_index(drop = True)

    text = df["overview"].fillna("").astype(str)
    is_empty = text.str.len() == 0

    ## Use title if overview is empty
    text.loc[is_empty] = df.loc[is_empty, "title_tmdb"].fillna(df.loc[is_empty, "title_ml"]).astype(str)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device = device)
    model.eval()

    embs = np.zeros((len(df), model.get_sentence_embedding_dimension()), dtype = np.float32)

    with torch.no_grad():
        for ids in generate_chunks(list(range(len(df))), batch_size):
            batch = list(text.iloc[ids].values)
            e = model.encode(batch, 
                             batch_size = len(batch), 
                             convert_to_numpy = True, 
                             show_progress_bar = False,
                             normalize_embeddings = True)

            embs[ids] = e
    
    np.save(P.TEXT_EMB, embs)

    norms = np.linalg.norm(embs, axis = 1)
    df["pos_norm"] = norms > 0.0
    df["text_norm"] = norms.astype(np.float32)
    df.to_parquet(P.ITEMS, index = False)

    print(f"[Embedding Computation] Saved text embeddings at {P.TEXT_EMB} with shape {embs.shape}")
    print(f"[Embedding Computation] Updated items.parquet ({P.ITEMS}). Mean norm = {norms.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type = int, default = 128)
    args = parser.parse_args()
    main(**vars(args))