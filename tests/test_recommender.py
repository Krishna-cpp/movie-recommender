import numpy as np
import pandas as pd
import pytest

from recommender.data import GENRES, clean_title
from recommender.model import Recommender
from recommender.posters import placeholder


def make_movies():
    rows = [
        (1, "Space Battle", 1990, ["Action", "Sci-Fi"]),
        (2, "Space Battle II", 1992, ["Action", "Sci-Fi"]),
        (3, "Love in Paris", 1995, ["Romance", "Comedy"]),
        (4, "Love in Rome", 1996, ["Romance", "Comedy"]),
        (5, "Haunted House", 1980, ["Horror"]),
    ]
    df = pd.DataFrame(rows, columns=["movie_id", "title", "year", "genres"])
    for g in GENRES:
        df[g] = df["genres"].apply(lambda gs, g=g: int(g in gs))
    df["year"] = df["year"].astype("Int64")
    return df


def make_ratings():
    # users 1-3 love sci-fi, users 4-6 love romance; everyone dislikes horror
    data = []
    for u in (1, 2, 3):
        data += [(u, 1, 5), (u, 2, 5), (u, 3, 1), (u, 5, 2)]
    for u in (4, 5, 6):
        data += [(u, 3, 5), (u, 4, 5), (u, 1, 1), (u, 5, 2)]
    data += [(7, 1, 5)]  # user 7 has only seen movie 1
    df = pd.DataFrame(data, columns=["user_id", "movie_id", "rating"])
    df["timestamp"] = 0
    return df


@pytest.fixture
def model():
    return Recommender(make_ratings(), make_movies(), shrinkage=0.0)


def test_similarity_matrices_are_symmetric_with_zero_diagonal(model):
    for S in (model.taste_sim, model.cowatch_sim, model.genre_sim):
        assert np.allclose(S, S.T)
        assert np.allclose(np.diag(S), 0)


def test_similar_movies_finds_the_sequel(model):
    recs = model.similar_movies(1, n=2, min_ratings=1)
    assert recs.iloc[0]["movie_id"] == 2
    assert 1 not in recs["movie_id"].values


def test_pure_genre_similarity(model):
    recs = model.similar_movies(3, n=1, mainstream=0.5, genre_weight=1.0, min_ratings=1)
    assert recs.iloc[0]["movie_id"] == 4


def test_recommend_from_liked_excludes_inputs(model):
    recs = model.recommend_from_liked([3], n=10, min_ratings=1)
    assert 3 not in recs["movie_id"].values
    assert recs.iloc[0]["movie_id"] == 4


def test_recommend_for_user_skips_seen_and_predicts_stars(model):
    recs = model.recommend_for_user(7, n=10, min_ratings=1)
    assert 1 not in recs["movie_id"].values
    assert recs.iloc[0]["movie_id"] == 2
    assert recs["predicted_rating"].between(1, 5).all()


def test_unknown_user_raises(model):
    with pytest.raises(ValueError):
        model.recommend_for_user(999)


def test_filters(model):
    recs = model.similar_movies(1, n=10, min_ratings=1, genres=["Romance"])
    assert set(recs["movie_id"]) == {3, 4}
    recs = model.similar_movies(1, n=10, min_ratings=1, years=(1985, 1993))
    assert set(recs["movie_id"]) == {2}
    assert model.similar_movies(1, n=10, min_ratings=100).empty


def test_top_rated_uses_bayesian_average(model):
    chart = model.top_rated(n=5, min_ratings=1)
    assert chart["weighted_rating"].is_monotonic_decreasing
    assert chart.iloc[-1]["movie_id"] == 5  # horror is rated 2 by everyone


def test_user_history_sorted(model):
    hist = model.user_history(1)
    assert hist["your_rating"].is_monotonic_decreasing


def test_search(model):
    assert set(model.search("love")["title"]) == {"Love in Paris", "Love in Rome"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Usual Suspects, The (1995)", "The Usual Suspects"),
        ("Toy Story (1995)", "Toy Story"),
        ("Seven (Se7en) (1995)", "Seven"),
        ("Pagemaster, The (1994)", "The Pagemaster"),
    ],
)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_placeholder_is_data_uri():
    assert placeholder("A <weird> & long movie title here").startswith("data:image/svg+xml;base64,")
