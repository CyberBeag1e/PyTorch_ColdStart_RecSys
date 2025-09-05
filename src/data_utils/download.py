import argparse
import os
import requests
import zipfile

from typing import Literal

from src.config.paths_config import P

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "full": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}

def stream_download(url: str, output_path: str) -> None:
    """
    Download files from `url` to `output_path` as .zip file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    req = requests.get(url, stream = True, timeout = 60)
    req.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in req.iter_content(chunk_size = 1 << 20):
            if chunk:
                f.write(chunk)


def unzip(file_path: str, dest_dir: str) -> None:
    """
    Unzip .zip file to `dest_dir`.
    """
    os.makedirs(dest_dir, exist_ok = True)
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall(dest_dir)


def main(version: Literal["small", "full"]) -> None:
    """
    Download the specified version of MovieLens data. 
    """
    file_name = os.path.join(P.RAW, "movielens", f"ml-{version}.zip")
    dest_dir = os.path.join(P.RAW, "movielens")

    url = URLS[version]
    print(f"[File Download] Downloading {version} version from {url}")
    try:
        stream_download(url, file_name)
    except Exception as e:
        print(f"[File Download] download error: {e}")

    print("[File Download] Unzipping ...")
    try:
        unzip(file_name, dest_dir)
    except Exception as e:
        print(f"[File Download] unzip error: {e}")

    print(f"[File Downloading] Finished. Files in {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices = ["small", "full"], default = "small")
    # parser.add_argument("--data_dir", default = "data")
    
    args = parser.parse_args()
    main(args.version, args.data_dir)