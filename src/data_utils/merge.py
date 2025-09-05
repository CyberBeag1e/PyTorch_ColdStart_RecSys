import json
import os
import pandas as pd
import requests
import time

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from typing import Optional

from src.config.paths_config import P, DATA_DIR

BASE_URL = "https://image.tmdb.org/t/p/w500"
THRESHOLD = 4.0

@retry(stop = stop_after_attempt(5),
       wait = wait_exponential_jitter(2, max = 10))
def tmdb_get(url: str, params: dict) -> str:
    """
    Fetch data from TMDB.
    """
    req = requests.get(url, params = params, timeout = 30)
    if req.status_code == 429:
        time.sleep(2)
    
    req.raise_for_status()
    return req.json()

def fetch_movie(tmdb_id: int, api_key: str, json_dir: str) -> str:
    """
    Fetch movie info from TMDB server. Return JSON formatted data.
    """
    json_path = os.path.join(json_dir, f"{tmdb_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding = "utf-8") as f:
            return json.load(f)
    
    data = tmdb_get(f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                    params = {
                        "api_key": api_key,
                        "language": "en-US"
                    })
    
    os.makedirs(json_dir, exist_ok = True)
    with open(json_path, "w", encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii = False)
    
    return data

def download_poster(poster_path: str, posters_dir: str) -> Optional[str]:
    """
    Download posters from TMDB server. Return the local file path.
    """
    if not poster_path:
        return
    
    poster_url = BASE_URL + poster_path
    os.makedirs(posters_dir, exist_ok = True)

    name = poster_path.strip("/").replace("/", "_")
    output = os.path.join(posters_dir, name)

    if os.path.exists(output):
        return output

    req = requests.get(poster_url, timeout = 60)
    if req.status_code == 200:
        with open(output, "wb") as f:
            f.write(req.content)
        
        return output
    
    return

def main() -> None:
    """
    Gather data (movie info + posters) from TMDB server as features fed to the recommenders.
    """
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    assert api_key, "TMDB_API_KEY not found."

    raw_dir = os.path.join(P.RAW, "movielens", "ml-latest-small")
    ratings = os.path.join(raw_dir, "ratings.csv")
    movies = os.path.join(raw_dir, "movies.csv")
    links = os.path.join(raw_dir, "links.csv")

    df_ratings = pd.read_csv(ratings)
    df_movies = pd.read_csv(movies)
    df_links = pd.read_csv(links)

    df_links.dropna(subset = ["tmdbId"])
    df_links = df_links[df_links["tmdbId"] > 0]
    df_full = pd.merge(left = df_movies, 
                       right = df_links[["movieId", "tmdbId"]], 
                       on = "movieId", 
                       how = "inner")
    
    df_full["tmdbId"] = df_full["tmdbId"].astype(int)

    json_dir = os.path.join(P.RAW, "tmdb", "json")
    posters_dir = os.path.join(P.RAW, "tmdb", "posters")

    rows = []
    for tmdb_id, group in df_full.groupby("tmdbId"):
        try:
            j = fetch_movie(tmdb_id, api_key, json_dir)
            title_tmdb  = j.get("title")
            overview    = (j.get("overview") or "").strip()
            date        = j.get("release_date")
            genres      = [g.get("name") for g in (j.get("genres") or [])]
            poster_id   = j.get("poster_path")
            poster_path = download_poster(poster_id, posters_dir) if poster_id else None

            row = group.iloc[0]
            rows.append({
                "movieId"           : int(row["movieId"]), 
                "tmdbId"            : int(tmdb_id), 
                "title_ml"          : row["title"],
                "genres_ml"         : row["genres"],
                "title_tmdb"        : title_tmdb,
                "overview"          : overview,
                "genres_tmdb"       : genres,
                "release_date"      : date, 
                "poster_id_tmdb"    : poster_id,
                "poster_path_local" : poster_path
            })

        except Exception as e:
            print(f"[Preprocess] tmdbId {tmdb_id} failed: {e}")
    
    items = pd.DataFrame(rows)
    
    interim_dir = os.path.join(DATA_DIR, "interim")
    os.makedirs(interim_dir, exist_ok = True)
    items.to_parquet(os.path.join(interim_dir, "items.parquet"), index = False)

    df_ratings["ts"] = pd.to_datetime(df_ratings["timestamp"], unit = "s")
    df_pos = df_ratings[df_ratings["rating"] >= THRESHOLD].copy()
    df_pos = df_pos[df_pos["movieId"].isin(items["movieId"].unique())]
    df_pos = df_pos[["userId", "movieId", "ts"]].sort_values("ts")

    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok = True)
    df_pos.to_parquet(os.path.join(processed_dir, "interactions.parquet"), index = False)

    items["with_text"] = items["overview"].str.len().fillna(0) > 0
    items_model = items[items["with_text"]].copy()
    items_model.to_parquet(os.path.join(processed_dir, "items.parquet"), index = False)

    prop = len(items_model) / len(items) if len(items) else 0
    print("-" * 20)
    print(f"[Preprocess] Merged MovieLens data with TMDB features.")
    print(f"[Preprocess] Items enriched: {len(items)}, {len(items_model)} with overview ({prop:.1%})")
    print(f"[Preprocess] Saved:")
    print(f"             {DATA_DIR}/interim/items.parquet (enriched raw)")
    print(f"             {DATA_DIR}/processed/items.parquet (model-ready)")
    print(f"             {DATA_DIR}/processed/interactions.parquet (implicit positives)")
    print(f"             {DATA_DIR}/interim/items.parquet (enriched raw)")


if __name__ == "__main__":
    main()