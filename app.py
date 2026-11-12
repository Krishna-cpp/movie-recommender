"""Streamlit front-end for the hybrid movie recommender."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from recommender import GENRES, Recommender, load_dataset
from recommender.model import DEFAULT_GENRE_WEIGHT, DEFAULT_MAINSTREAM
from recommender.posters import get_api_key, posters_for

st.set_page_config(page_title="CineMatch · Movie Recommender", page_icon="🎬", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0;}
.hero p {color: #a9a9b3; margin-top: .25rem;}
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr));
       gap: 1.1rem; margin: .5rem 0 1.5rem;}
.card {background: #1b1b22; border-radius: 10px; overflow: hidden;
       box-shadow: 0 2px 10px rgba(0,0,0,.35); transition: transform .15s ease;
       position: relative;}
.card:hover {transform: translateY(-4px) scale(1.02);}
.card img {width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block;
           background: #2b2b3a;}
.card .body {padding: .55rem .65rem .7rem;}
.card .title {font-weight: 600; font-size: .92rem; line-height: 1.2; color: #f2f2f5;}
.card .meta {font-size: .78rem; color: #9a9aa6; margin-top: .2rem;}
.card .chips {margin-top: .35rem; display: flex; flex-wrap: wrap; gap: .25rem;}
.card .chip {font-size: .66rem; background: #2c2c38; color: #cfcfe0;
             padding: .05rem .4rem; border-radius: 999px;}
.badge {position: absolute; top: .45rem; left: .45rem; background: #e50914;
        color: white; font-weight: 700; font-size: .75rem; padding: .1rem .45rem;
        border-radius: 6px;}
.rank {position: absolute; top: .45rem; right: .45rem; background: rgba(0,0,0,.7);
       color: white; font-weight: 700; font-size: .75rem; padding: .1rem .45rem;
       border-radius: 6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- data/model
@st.cache_resource(show_spinner="Loading MovieLens and training the model…")
def get_model() -> Recommender:
    ratings, movies = load_dataset()
    return Recommender(ratings, movies)


model = get_model()


def label(movie_id: int) -> str:
    row = model.movies.loc[movie_id]
    return f"{row.title} ({row.year})" if pd.notna(row.year) else row.title


# Selectable movies, most-rated first so popular titles are easy to find
movie_options = model.stats.sort_values("num_ratings", ascending=False).index.tolist()


def render_cards(df: pd.DataFrame, badge: str | None = None, badge_fmt=None) -> None:
    if df.empty:
        st.info("No movies match these filters — try loosening them in the sidebar.")
        return
    years = [int(y) if pd.notna(y) else None for y in df["year"]]
    posters = posters_for(zip(df["title"], years, strict=True))
    cards = []
    for i, (row, poster) in enumerate(zip(df.itertuples(), posters, strict=True), start=1):
        year = f"{row.year} · " if pd.notna(row.year) else ""
        chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in row.genres[:3])
        badge_html = ""
        if badge:
            value = getattr(row, badge)
            badge_html = f'<span class="badge">{badge_fmt(value) if badge_fmt else value}</span>'
        cards.append(
            f'<div class="card">{badge_html}<span class="rank">#{i}</span>'
            f'<img src="{html.escape(poster)}" alt="" loading="lazy">'
            f'<div class="body"><div class="title">{html.escape(row.title)}</div>'
            f'<div class="meta">{year}★ {row.avg_rating:.1f} · {row.num_ratings:,} ratings</div>'
            f'<div class="chips">{chips}</div></div></div>'
        )
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def pct(x: float) -> str:
    return f"{x:.0%} match"


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Tuning")
    n_recs = st.slider("Number of recommendations", 4, 24, 12, step=2)
    mainstream = st.slider(
        "Niche ⟷ Mainstream",
        0.0,
        1.0,
        DEFAULT_MAINSTREAM,
        0.05,
        help="Left leans on rating similarity (hidden gems people loved). "
        "Right leans on co-watching patterns (crowd favourites).",
    )
    genre_weight = st.slider(
        "Genre influence",
        0.0,
        1.0,
        DEFAULT_GENRE_WEIGHT,
        0.05,
        help="How much shared genres count versus audience behaviour.",
    )
    min_ratings = st.slider(
        "Minimum number of ratings",
        1,
        200,
        20,
        help="Hide obscure movies with too few ratings to trust.",
    )

    st.header("🔎 Filters")
    genre_filter = st.multiselect("Genres", [g for g in GENRES if g != "Unknown"])
    years = model.movies["year"].dropna().astype(int)
    year_range = st.slider(
        "Release year", int(years.min()), int(years.max()), (int(years.min()), int(years.max()))
    )

    st.divider()
    if get_api_key():
        st.caption("🖼️ Posters: OMDb API connected")
    else:
        st.caption(
            "🖼️ Posters: set `OMDB_API_KEY` to show real posters "
            "([get a free key](https://www.omdbapi.com/apikey.aspx))."
        )
    st.caption(
        "Data: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) "
        "· GroupLens Research"
    )

filters = dict(
    genres=genre_filter or None,
    years=year_range,
)
blend = dict(mainstream=mainstream, genre_weight=genre_weight)


# ----------------------------------------------------------------------- page
st.markdown(
    '<div class="hero"><h1>🎬 CineMatch</h1>'
    f"<p>A hybrid recommender trained on {len(model.movie_ids):,} movies and "
    f"{int(model.mask.sum()):,} ratings from {len(model.user_ids):,} users.</p></div>",
    unsafe_allow_html=True,
)

tab_similar, tab_taste, tab_user, tab_charts = st.tabs(
    ["🎯 More like this", "✨ Build your taste", "👤 Recommend for a user", "🏆 Charts"]
)

with tab_similar:
    movie = st.selectbox(
        "Pick a movie you enjoyed",
        movie_options,
        format_func=label,
        index=movie_options.index(1) if 1 in movie_options else 0,
    )
    st.subheader(f"Because you liked {model.movies.loc[movie, 'title']}")
    recs = model.similar_movies(movie, n_recs, min_ratings=min_ratings, **blend, **filters)
    render_cards(recs, "match", pct)

with tab_taste:
    liked = st.multiselect(
        "Add a few movies you love — the more, the better",
        movie_options,
        format_func=label,
        default=[50, 98, 127],
        max_selections=15,
    )
    if liked:
        recs = model.recommend_from_liked(
            liked, n_recs, min_ratings=min_ratings, **blend, **filters
        )
        st.subheader("Picked for you")
        render_cards(recs, "match", pct)
    else:
        st.info("Select at least one movie to get recommendations.")

with tab_user:
    c1, c2 = st.columns([1, 3])
    user_id = c1.number_input(
        "MovieLens user id",
        min_value=int(model.user_ids.min()),
        max_value=int(model.user_ids.max()),
        value=196,
        step=1,
    )
    n_rated = int(model.mask[:, model._uidx[user_id]].sum())
    c2.markdown(f"<br>User **{user_id}** has rated **{n_rated}** movies.", unsafe_allow_html=True)
    with st.expander("Their favourite movies", expanded=False):
        render_cards(model.user_history(user_id, 8), "your_rating", lambda r: f"★ {r:.0f}")
    recs = model.recommend_for_user(user_id, n_recs, min_ratings=min_ratings, **blend, **filters)
    st.subheader(f"Top picks for user {user_id}")
    st.caption("Badge shows the predicted star rating from item-kNN rating prediction.")
    render_cards(recs, "predicted_rating", lambda r: f"★ {r:.1f}")

with tab_charts:
    kind = st.radio("Chart", ["Top rated", "Most popular"], horizontal=True)
    if kind == "Top rated":
        st.caption(
            "Ranked by Bayesian average, so a 5★ movie with 3 ratings "
            "doesn't outrank a classic with hundreds."
        )
        chart = model.top_rated(n_recs, min_ratings=max(min_ratings, 1), **filters)
        render_cards(chart, "weighted_rating", lambda r: f"★ {r:.2f}")
    else:
        chart = model.most_popular(n_recs, **filters)
        render_cards(chart)
