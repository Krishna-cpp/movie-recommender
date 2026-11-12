"""Poster lookup via the OMDb API, with a generated placeholder fallback."""

from __future__ import annotations

import base64
import html
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests

OMDB_URL = "https://www.omdbapi.com/"
_session = requests.Session()


def get_api_key() -> str | None:
    """OMDb key from the OMDB_API_KEY env var or Streamlit secrets."""
    key = os.environ.get("OMDB_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("OMDB_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=4096)
def fetch_poster(title: str, year: int | None, api_key: str | None) -> str | None:
    if not api_key:
        return None
    attempts = [{"t": title, "y": year}, {"t": title}] if year else [{"t": title}]
    for params in attempts:
        try:
            res = _session.get(
                OMDB_URL, params={**params, "apikey": api_key, "type": "movie"}, timeout=5
            ).json()
        except (requests.RequestException, ValueError):
            return None
        poster = res.get("Poster")
        if res.get("Response") == "True" and poster and poster != "N/A":
            return poster
    return None


def placeholder(title: str) -> str:
    """An inline SVG poster with the movie title, as a data URI."""
    words, lines, line = title.split(), [], ""
    for w in words:
        if len(line) + len(w) > 14 and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    lines = lines[:5]
    start = 225 - (len(lines) - 1) * 16
    text = "".join(
        f'<text x="150" y="{start + i * 32}" text-anchor="middle">{html.escape(t)}</text>'
        for i, t in enumerate(lines)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#2b2b3a"/><stop offset="1" stop-color="#141418"/>'
        "</linearGradient></defs>"
        '<rect width="300" height="450" fill="url(#g)"/>'
        '<text x="150" y="120" text-anchor="middle" font-size="56">🎬</text>'
        f'<g fill="#e8e8ee" font-family="Helvetica,Arial,sans-serif" font-size="24" '
        f'font-weight="600">{text}</g></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def posters_for(movies: Iterable[tuple[str, int | None]]) -> list[str]:
    """Poster URLs for (title, year) pairs, fetched in parallel."""
    movies = list(movies)
    key = get_api_key()
    with ThreadPoolExecutor(max_workers=8) as pool:
        urls = list(pool.map(lambda m: fetch_poster(m[0], m[1], key), movies))
    return [u or placeholder(t) for u, (t, _) in zip(urls, movies, strict=True)]
