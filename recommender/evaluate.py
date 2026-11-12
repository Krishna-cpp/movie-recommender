"""Offline evaluation: leave-one-out hit rate @ K.

For every user, one movie they rated >= 4 is hidden. The model is trained on the
remaining ratings and we check whether the hidden movie shows up in that user's
top-K recommendations.

    python -m recommender.evaluate --k 10
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load_dataset
from .model import Recommender


def leave_one_out(ratings, seed: int = 42):
    rng = np.random.default_rng(seed)
    liked = ratings[ratings["rating"] >= 4]
    held = liked.groupby("user_id").sample(n=1, random_state=rng.integers(1 << 31))
    train = ratings.drop(held.index)
    return train, held


def evaluate(k: int = 10, seed: int = 42, min_ratings: int = 5) -> dict[str, float]:
    ratings, movies = load_dataset()
    train, held = leave_one_out(ratings, seed)
    model = Recommender(train, movies)

    hits = {
        "popularity": 0,
        "top_rated": 0,
        "rating_prediction": 0,
        "hybrid (co-watch only)": 0,
        "hybrid (default)": 0,
    }
    users = 0
    pop = model.stats["num_ratings"].to_numpy(dtype=float)
    chart = model.top_rated(n=len(model.movie_ids), min_ratings=min_ratings)
    for row in held.itertuples():
        uid, target = row.user_id, row.movie_id
        if uid not in model._uidx:
            continue
        users += 1
        u = model._uidx[uid]
        seen = model.movie_ids[model.mask[:, u]]
        ok = model._candidate_mask(min_ratings, None, None, exclude=seen)

        hits["popularity"] += target in model._top(pop, ok, k, "s")["movie_id"].values
        tr = chart[~chart["movie_id"].isin(seen)].head(k)
        hits["top_rated"] += target in tr["movie_id"].values
        pred = model.predict_ratings(uid)
        hits["rating_prediction"] += target in model._top(pred, ok, k, "s")["movie_id"].values
        recs = model.recommend_for_user(
            uid, n=k, mainstream=1.0, genre_weight=0.0, min_ratings=min_ratings
        )
        hits["hybrid (co-watch only)"] += target in recs["movie_id"].values
        recs = model.recommend_for_user(uid, n=k, min_ratings=min_ratings)
        hits["hybrid (default)"] += target in recs["movie_id"].values

    return {name: h / users for name, h in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = evaluate(k=args.k, seed=args.seed)
    print(f"Leave-one-out HitRate@{args.k}")
    for name, value in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {value:.3f}")


if __name__ == "__main__":
    main()
