# 🎬 CineMatch: Hybrid Movie Recommender

[![CI](https://github.com/Krishna-cpp/movie-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Krishna-cpp/movie-recommender/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)

A Netflix-style movie recommender built with Streamlit on the
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset
(100,000 ratings, 943 users, 1,682 movies). It blends collaborative filtering
with content-based genre similarity, and shows real posters from the OMDb API.

## Features

| Tab | What it does |
| --- | --- |
| 🎯 **More like this** | Pick a movie and get the most similar titles |
| ✨ **Build your taste** | Pick several favourites and get recommendations for the combination |
| 👤 **Recommend for a user** | Personalised top-N for any MovieLens user, with predicted star ratings |
| 🏆 **Charts** | Top rated (Bayesian average) and most popular |

**Sidebar controls:** number of results, a *Niche ⟷ Mainstream* slider, genre
influence, minimum-ratings threshold, genre filter and release-year range.

## How it works

The model (`recommender/model.py`) is an item-item recommender that blends
three similarity signals:

1. **Co-watch similarity**: cosine similarity on implicit feedback (who
   rated what). It predicts well what people actually watch next.
2. **Taste similarity**: *adjusted* cosine on ratings (each rating centred on
   that user's mean) with significance shrinkage `n / (n + 25)`, so two movies
   sharing three raters can't look "identical". It brings out niche picks.
3. **Genre similarity**: cosine between genre vectors. This is content-based
   and helps with sparse items.

`score = (1 − g)·[m·cowatch + (1 − m)·taste] + g·genre`, with defaults
m = 0.45 and g = 0.10. A user's recommendations come from summing similarity to
everything they've watched. Predicted star ratings use classic item-kNN (k = 30).

All similarity matrices are precomputed with NumPy (1,682 × 1,682), so every
recommendation is a single vectorised lookup.

## Evaluation

Leave-one-out **HitRate@10**: for each of the 943 users, one movie they rated
≥ 4★ is hidden. The model is trained on the rest, and we check whether the
hidden movie appears in their top 10.

```bash
python -m recommender.evaluate
```

| Method | HitRate@10 |
| --- | --- |
| **Hybrid (default blend)** | **0.308** |
| Hybrid, co-watch only | 0.283 |
| Most popular (baseline) | 0.177 |
| Ranking by predicted rating | 0.139 |
| Top rated (baseline) | 0.055 |

The default blend beats the popularity baseline by **~74%**.

## Getting started

```bash
git clone https://github.com/Krishna-cpp/movie-recommender.git
cd movie-recommender
python -m venv .venv && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The MovieLens data downloads automatically into `data/` on first run.

### Posters (optional)

Get a free key at <https://www.omdbapi.com/apikey.aspx>, then either:

- set an environment variable: `OMDB_API_KEY=your-key`, or
- copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add it there.

Without a key, the app shows generated title cards instead of posters.

### Deploy to Streamlit Community Cloud

Push the repo, create an app at [share.streamlit.io](https://share.streamlit.io)
pointing to `app.py`, and add `OMDB_API_KEY` under **Settings → Secrets**.

## Development

```bash
pip install -r requirements-dev.txt
pytest          # unit tests on a small synthetic dataset (no network)
ruff check .    # lint
```

CI runs lint and tests on Python 3.10 and 3.12 for every push.

## Project structure

```
app.py                  Streamlit UI
recommender/
  data.py               dataset download, parsing, title cleanup
  model.py              hybrid recommender
  posters.py            OMDb poster lookup (parallel, cached) and placeholders
  evaluate.py           offline leave-one-out evaluation
tests/                  pytest suite
```

## Acknowledgements

Data: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
History and Context.* ACM TiiS 5, 4. The dataset isn't redistributed in this repo,
in line with its licence. Posters come from [OMDb](https://www.omdbapi.com/).
