from dataclasses import dataclass
import os

DATA_DIR = "data"
DATASET = "ml-100k"
@dataclass(frozen = True)
class Paths:
    """
    Store the file paths and directories.
    """
    RAW         : str = os.path.join(DATA_DIR, "raw")
    INTERIM     : str = os.path.join(DATA_DIR, "interim", DATASET)
    PROCESSED   : str = os.path.join(DATA_DIR, "processed", DATASET)

    ITEMS       : str = os.path.join(DATA_DIR, "processed", "items.parquet")
    INTERACTIONS: str = os.path.join(DATA_DIR, "processed", "interactions.parquet")
    TEXT_EMB    : str = os.path.join(INTERIM, "text_emb.npy")
    IMG_EMB     : str = os.path.join(INTERIM, "img_emb.npy")

    TRAIN       : str = os.path.join(PROCESSED, "train.parquet")
    VAL         : str = os.path.join(PROCESSED, "val.parquet")
    TEST        : str = os.path.join(PROCESSED, "test.parquet")
    ITEMS_META  : str = os.path.join(PROCESSED, "items_meta.parquet")
    MAPPINGS    : str = os.path.join(PROCESSED, "mappings.json")

P = Paths()
