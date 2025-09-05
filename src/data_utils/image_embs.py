import argparse
import numpy as np
import open_clip
import os
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from typing import Optional, Literal
from src.config.paths_config import P

def load_image(path: str) -> Optional[Image.Image]:
    """Load posters via `PIL`."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return

def main(clip_model: str = "ViT-B-32", 
         pretrained: str = "openai", 
         batch_size: int = 64, 
         device: Literal["cuda", "cpu"] = "cuda"):
    
    """
    Pre-compute image (poster) embeddings using CLIP models. By default use pretrained model from openai.

    In this implementation, the computations are forced to completed on CUDA to avoid out of memory error on CPU.
    """
    
    os.makedirs(P.INTERIM, exist_ok = True)
    df = pd.read_parquet(P.ITEMS).reset_index(drop = True)

    ## Force the computation on CUDA
    if not torch.cuda.is_available():
        print("[Embedding Computation] CUDA is not available")
        return
    
    device = device or "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained = pretrained, device = device)
    model.eval()

    dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
    embs = np.zeros((len(df), dim), dtype = np.float32)
    flag = np.zeros(len(df), dtype = bool)

    paths = df["poster_path_local"].fillna("").astype(str).to_list()
    n_paths = len(paths)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled = (device == "cuda")):
        batch_imgs, batch_ids = [], []

        for i, p in enumerate(paths):
            if not p or not os.path.exists(p):
                continue
            
            img = load_image(p)
            if img is None:
                continue

            batch_imgs.append(preprocess(img))
            batch_ids.append(i)

            if len(batch_imgs) == batch_size or (batch_imgs and i == n_paths - 1):
                pixel = torch.stack(batch_imgs).to(device)
                feat = model.encode_image(pixel)
                feat = F.normalize(feat, dim = -1)
                embs[np.array(batch_ids)] = feat.float().cpu().numpy()
                flag[np.array(batch_ids)] = True

                batch_imgs, batch_ids = [], []
    
    np.save(P.IMG_EMB, embs)

    norms = np.linalg.norm(embs, axis = 1)
    df["with_img"] = flag
    df["img_norm"] = norms.astype(np.float32)
    df.to_parquet(P.ITEMS, index = False)

    coverage = flag.mean() * 100
    print(f"[Embedding Computation] Saved image embeddings at {P.IMG_EMB} with shape {embs.shape}. Poster coverage: {coverage:.1f}%")
    print(f"[Embedding Computation] Updated items.parquet ({P.ITEMS}). Mean norm = {norms[flag].mean():.3f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", default = "ViT-B-32")
    parser.add_argument("--pretrained", default = "openai")
    parser.add_argument("--batch_size", type = int, default = 64)
    args = parser.parse_args()
    main(**vars(args))