"""Hybrid movie recommender: item-based collaborative filtering + genre similarity."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import GENRES

DEFAULT_MAINSTREAM = 0.45
DEFAULT_GENRE_WEIGHT = 0.1


class Recommender:
    """Hybrid item-item recommender blending three similarity signals:

    * **co-watch** - cosine similarity on implicit feedback (who rated what).
      Strong at predicting what people will watch next; leans mainstream.
    * **taste** - adjusted cosine on ratings (centred on each user's mean) with
      significance shrinkage, so movies sharing few raters aren't over-scored.
      Captures *how much* people liked things; surfaces more niche picks.
    * **genre** - cosine similarity between genre vectors (content-based).

    ``similarity(mainstream, genre_weight)`` mixes them. The defaults were
    tuned with ``python -m recommender.evaluate``.
    """

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, shrinkage: float = 25.0):
        self._sim_cache: dict[tuple[float, float], np.ndarray] = {}
        self.movies = movies.set_index("movie_id").sort_index()
        self.movie_ids = self.movies.index.to_numpy()
        self._idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        ratings = ratings[ratings["movie_id"].isin(self._idx)]
        self.user_ids = np.sort(ratings["user_id"].unique())
        self._uidx = {uid: i for i, uid in enumerate(self.user_ids)}

        n_items, n_users = len(self.movie_ids), len(self.user_ids)
        rows = ratings["movie_id"].map(self._idx).to_numpy()
        cols = ratings["user_id"].map(self._uidx).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=float)

        R = np.zeros((n_items, n_users))
        R[rows, cols] = vals
        mask = R > 0
        self.R = R
        self.mask = mask

        # Per-movie stats
        counts = mask.sum(axis=1)
        sums = R.sum(axis=1)
        self.stats = pd.DataFrame(
            {
                "num_ratings": counts,
                "avg_rating": np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0),
            },
            index=self.movie_ids,
        )

        # Adjusted cosine similarity
        user_counts = mask.sum(axis=0)
        self.user_means = np.divide(
            R.sum(axis=0), user_counts, out=np.zeros(n_users), where=user_counts > 0
        )
        Rc = np.where(mask, R - self.user_means, 0.0)
        norms = np.linalg.norm(Rc, axis=1)
        norms[norms == 0] = 1.0
        cf = (Rc @ Rc.T) / np.outer(norms, norms)
        co_counts = mask.astype(float) @ mask.T.astype(float)
        denom = co_counts + shrinkage
        cf *= np.divide(co_counts, denom, out=np.zeros_like(denom), where=denom > 0)
        np.fill_diagonal(cf, 0.0)
        self.taste_sim = cf

        # Implicit-feedback (co-watch) cosine similarity
        wnorm = np.sqrt(counts.astype(float))
        wnorm[wnorm == 0] = 1.0
        cooc = co_counts / np.outer(wnorm, wnorm)
        np.fill_diagonal(cooc, 0.0)
        self.cowatch_sim = cooc

        # Genre similarity
        G = self.movies[GENRES].to_numpy(dtype=float)
        gnorm = np.linalg.norm(G, axis=1)
        gnorm[gnorm == 0] = 1.0
        Gn = G / gnorm[:, None]
        content = Gn @ Gn.T
        np.fill_diagonal(content, 0.0)
        self.genre_sim = content

    # ------------------------------------------------------------------ helpers
    def similarity(
        self, mainstream: float = DEFAULT_MAINSTREAM, genre_weight: float = DEFAULT_GENRE_WEIGHT
    ) -> np.ndarray:
        """Blended item-item similarity.

        mainstream: 0 = only rating-based taste similarity, 1 = only co-watch.
        genre_weight: share of the final score coming from genre overlap.
        """
        key = (round(mainstream, 3), round(genre_weight, 3))
        if key not in self._sim_cache:
            cf = mainstream * self.cowatch_sim + (1 - mainstream) * self.taste_sim
            self._sim_cache.clear()  # keep memory bounded: one blend at a time
            self._sim_cache[key] = (1 - genre_weight) * cf + genre_weight * self.genre_sim
        return self._sim_cache[key]

    def index_of(self, movie_id: int) -> int:
        return self._idx[movie_id]

    def _candidate_mask(
        self,
        min_ratings: int,
        genres: Sequence[str] | None,
        years: tuple[int, int] | None,
        exclude: Iterable[int] = (),
    ) -> np.ndarray:
        ok = self.stats["num_ratings"].to_numpy() >= min_ratings
        if genres:
            ok &= self.movies[list(genres)].to_numpy().any(axis=1)
        if years:
            y = self.movies["year"].astype("float").to_numpy()
            ok &= (y >= years[0]) & (y <= years[1])
        for mid in exclude:
            if mid in self._idx:
                ok[self._idx[mid]] = False
        return ok

    def _frame(self, idx: np.ndarray, scores: np.ndarray, score_name: str) -> pd.DataFrame:
        mids = self.movie_ids[idx]
        out = self.movies.loc[mids, ["title", "year", "genres"]].copy()
        out[score_name] = scores
        out = out.join(self.stats)
        out.index.name = "movie_id"
        return out.reset_index()

    def _top(self, scores: np.ndarray, ok: np.ndarray, n: int, name: str) -> pd.DataFrame:
        scores = np.where(ok, scores, -np.inf)
        k = min(n, int(ok.sum()))
        if k == 0:
            return self._frame(np.array([], dtype=int), np.array([]), name)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return self._frame(top, scores[top], name)

    # ------------------------------------------------------------- public API
    def similar_movies(
        self,
        movie_id: int,
        n: int = 10,
        mainstream: float = DEFAULT_MAINSTREAM,
        genre_weight: float = DEFAULT_GENRE_WEIGHT,
        min_ratings: int = 20,
        genres: Sequence[str] | None = None,
        years: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Movies most similar to ``movie_id``."""
        return self.recommend_from_liked(
            [movie_id], n, mainstream, genre_weight, min_ratings, genres, years
        )

    def recommend_from_liked(
        self,
        liked: Sequence[int],
        n: int = 10,
        mainstream: float = DEFAULT_MAINSTREAM,
        genre_weight: float = DEFAULT_GENRE_WEIGHT,
        min_ratings: int = 20,
        genres: Sequence[str] | None = None,
        years: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Recommendations for someone who liked every movie in ``liked``."""
        liked = [m for m in liked if m in self._idx]
        if not liked:
            raise ValueError("None of the given movie ids are known.")
        idx = [self._idx[m] for m in liked]
        S = self.similarity(mainstream, genre_weight)
        scores = S[:, idx].mean(axis=1)
        ok = self._candidate_mask(min_ratings, genres, years, exclude=liked)
        return self._top(scores, ok, n, "match")

    def predict_ratings(self, user_id: int, neighbours: int = 30) -> np.ndarray:
        """Predicted star rating (1-5) of every movie for ``user_id``.

        Classic item-kNN on the rating-based similarity: the user's mean plus a
        similarity-weighted average of their deviations on the k nearest rated movies.
        """
        u = self._uidx[user_id]
        rated = np.flatnonzero(self.mask[:, u])
        dev = self.R[rated, u] - self.user_means[u]

        S = np.clip(self.taste_sim[:, rated], 0, None)
        if S.shape[1] > neighbours:
            cutoff = np.partition(S, -neighbours, axis=1)[:, -neighbours][:, None]
            S = np.where(S >= cutoff, S, 0.0)
        # +1 damping so predictions backed by weak evidence regress to the mean
        pred = self.user_means[u] + (S @ dev) / (S.sum(axis=1) + 1.0)
        return np.clip(pred, 1, 5)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        mainstream: float = DEFAULT_MAINSTREAM,
        genre_weight: float = DEFAULT_GENRE_WEIGHT,
        min_ratings: int = 20,
        genres: Sequence[str] | None = None,
        years: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Top-N unseen movies for an existing user, with a predicted star rating."""
        if user_id not in self._uidx:
            raise ValueError(f"Unknown user {user_id}")
        u = self._uidx[user_id]
        watched = self.mask[:, u]
        scores = self.similarity(mainstream, genre_weight)[:, watched].sum(axis=1)
        scores = scores / max(watched.sum(), 1)
        ok = self._candidate_mask(min_ratings, genres, years, exclude=self.movie_ids[watched])
        out = self._top(scores, ok, n, "match")
        pred = self.predict_ratings(user_id)
        out["predicted_rating"] = pred[[self._idx[m] for m in out["movie_id"]]]
        return out

    def user_history(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """The user's highest-rated movies."""
        u = self._uidx[user_id]
        rated = np.flatnonzero(self.mask[:, u])
        order = rated[np.argsort(-self.R[rated, u], kind="stable")][:n]
        return self._frame(order, self.R[order, u], "your_rating")

    def top_rated(
        self,
        n: int = 10,
        min_ratings: int = 50,
        genres: Sequence[str] | None = None,
        years: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Chart ranked by Bayesian average (IMDb-style weighted rating)."""
        v = self.stats["num_ratings"].to_numpy(dtype=float)
        r = self.stats["avg_rating"].to_numpy()
        c = (r * v).sum() / v.sum()
        m = float(min_ratings)
        score = (v / (v + m)) * r + (m / (v + m)) * c
        ok = self._candidate_mask(min_ratings, genres, years)
        return self._top(score, ok, n, "weighted_rating")

    def most_popular(
        self, n: int = 10, genres: Sequence[str] | None = None, years: tuple[int, int] | None = None
    ) -> pd.DataFrame:
        score = self.stats["num_ratings"].to_numpy(dtype=float)
        ok = self._candidate_mask(1, genres, years)
        return self._top(score, ok, n, "popularity")

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        q = query.strip().lower()
        hits = self.movies[self.movies["title"].str.lower().str.contains(q, regex=False)]
        return hits.join(self.stats).sort_values("num_ratings", ascending=False).head(limit)
