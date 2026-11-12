"""Loading the MovieLens 100k dataset (downloaded automatically if missing)."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRES = [
    "Unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def ensure_dataset(data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Return paths to ratings/movies files, downloading MovieLens 100k if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"
    if ratings_path.exists() and movies_path.exists():
        return ratings_path, movies_path

    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        ratings_path.write_bytes(zf.read("ml-100k/u.data"))
        movies_path.write_bytes(zf.read("ml-100k/u.item"))
    return ratings_path, movies_path


def load_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def load_movies(path: Path) -> pd.DataFrame:
    cols = ["movie_id", "raw_title", "release_date", "video_release", "imdb_url", *GENRES]
    movies = pd.read_csv(path, sep="|", header=None, names=cols, encoding="latin-1")
    movies["year"] = movies["raw_title"].str.extract(_YEAR_RE, expand=False).astype("Int64")
    movies["title"] = movies["raw_title"].apply(clean_title)
    movies["genres"] = movies[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1 and g != "Unknown"], axis=1
    )
    return movies[["movie_id", "title", "year", "genres", *GENRES]]


def clean_title(raw: str) -> str:
    """'Usual Suspects, The (1995)' -> 'The Usual Suspects'."""
    title = _YEAR_RE.sub("", raw).strip()
    # MovieLens titles may carry an alternate name: 'Seven (Se7en)'
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title
    m = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Das|Der|Die)$", title)
    if m:
        title = f"{m.group(2)} {m.group(1)}"
    return title


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path, movies_path = ensure_dataset(data_dir)
    return load_ratings(ratings_path), load_movies(movies_path)
