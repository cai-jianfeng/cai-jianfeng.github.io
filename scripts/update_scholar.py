#!/usr/bin/env python3
"""Refresh the Google Scholar citation count used by the Jekyll site."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROFILE_ID = os.environ.get("SCHOLAR_PROFILE_ID", "1gqJAgcAAAAJ")
DATA_PATH = Path(os.environ.get("SCHOLAR_DATA_PATH", "_data/scholar.json"))
PROFILE_URL = f"https://scholar.google.com/citations?user={PROFILE_ID}&hl=en"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


def request_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_from_serpapi(api_key: str) -> int:
    query = urlencode(
        {
            "engine": "google_scholar_author",
            "author_id": PROFILE_ID,
            "hl": "en",
            "api_key": api_key,
        }
    )
    payload = request_json(f"https://serpapi.com/search.json?{query}")
    for metric in payload.get("cited_by", {}).get("table", []):
        citations = metric.get("citations", {}).get("all")
        if citations is not None:
            return validate_count(citations)
    raise ValueError("SerpApi response did not contain a total citation count")


def fetch_from_public_profile() -> int:
    request = Request(PROFILE_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    matches = re.findall(r'<td class="gsc_rsb_std">([0-9,]+)</td>', html)
    if not matches:
        raise ValueError("Google Scholar profile did not contain citation metrics")
    return validate_count(matches[0].replace(",", ""))


def validate_count(value: object) -> int:
    count = int(value)
    if count < 0:
        raise ValueError("Citation count cannot be negative")
    return count


def load_fallback() -> dict:
    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        validate_count(payload["citations"])
        return payload
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def write_data(citations: int, source: str) -> None:
    payload = {
        "profile_id": PROFILE_ID,
        "citations": citations,
        "updated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": source,
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATA_PATH.with_suffix(f"{DATA_PATH.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(DATA_PATH)


def main() -> int:
    fallback = load_fallback()
    fetchers: list[tuple[str, Callable[[], int]]] = []
    api_key = os.environ.get("SERPAPI_KEY", "").strip()
    if api_key:
        fetchers.append(("serpapi", lambda: fetch_from_serpapi(api_key)))
    fetchers.append(("google-scholar-profile", fetch_from_public_profile))

    for source, fetcher in fetchers:
        try:
            citations = fetcher()
            write_data(citations, source)
            print(f"Updated Google Scholar citations: {citations} ({source})")
            return 0
        except Exception:
            continue

    if fallback:
        print("::warning::Scholar refresh failed; using the checked-in citation count.")
        return 0

    print("::error::Scholar refresh failed and no valid fallback data is available.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
