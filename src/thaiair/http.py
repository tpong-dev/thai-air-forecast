"""Small HTTP helper: polite GET with retries, backoff and an optional on-disk JSON cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from thaiair import config

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = config.USER_AGENT
    return _session


def get(url: str, params: dict[str, Any] | None = None, retries: int = 4, pause: float = 0.0) -> requests.Response:
    """GET with exponential backoff on network errors, 429 and 5xx. Raises on final failure."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session().get(url, params=params, timeout=config.HTTP_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} for {resp.url}", response=resp)
            resp.raise_for_status()
            if pause:
                time.sleep(pause)
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    assert last is not None
    raise last


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    cache: Path | None = None,
    pause: float = 0.0,
) -> Any:
    """GET a JSON document. If `cache` exists it is read instead; otherwise the response is saved there."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = get(url, params=params, pause=pause).json()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def download(url: str, dest: Path, pause: float = 0.0) -> Path:
    """Download a binary file once (skips if `dest` already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = get(url, pause=pause)
    dest.write_bytes(resp.content)
    return dest
