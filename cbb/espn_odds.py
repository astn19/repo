"""Download historical odds from ESPN's core API and write a lines CSV.

ESPN keeps per-sportsbook lines for past games, with opening and closing numbers
(opening numbers available from the 2023-24 season onward).

Usage:
    python -m cbb.espn_odds --seasons 2022-2026 -o data/lines/espn_2022_2026.csv
Raw JSON is cached in data/raw/espn_odds/<season>/<game_id>.json.gz, so reruns only fetch new games.
"""
import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from . import config as C

URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
       "/events/{0}/competitions/{0}/odds")
SKIP_PROVIDERS = {"59", "1001"}  # live in-game odds, accuscore projections
NON_BOOKS = ("consensus", "teamrankings", "numberfire", "accuscore")
PREFERRED = ["40", "58", "47", "45", "52", "57", "41", "36"]  # DraftKings, ESPN BET, MGM, Caesars...
CACHE = C.RAW / "espn_odds"


def fetch(game_id, season, session, retries=3):
    path = CACHE / str(season) / f"{game_id}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    for attempt in range(retries):
        try:
            r = session.get(URL.format(game_id), timeout=30)
            if r.status_code == 404:
                d = {"items": []}
                break
            r.raise_for_status()
            d = r.json()
            break
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(d).encode()))
    return d


def _val(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return np.nan
        d = d.get(k)
    try:
        return float(d)
    except (TypeError, ValueError):
        return np.nan


def _american(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return np.nan
        d = d.get(k)
    try:
        return float(str(d).replace("+", "")) if d not in (None, "", "EVEN", "even") else (100.0 if d else np.nan)
    except ValueError:
        return np.nan


def parse(game_id, d):
    """One row per sportsbook with home spread/total at open and close."""
    rows = []
    for it in (d or {}).get("items", []):
        pid = str(it.get("provider", {}).get("id"))
        name = it.get("provider", {}).get("name", "")
        if pid in SKIP_PROVIDERS or name.lower().startswith(NON_BOOKS):
            continue
        h = it.get("homeTeamOdds", {})
        row = {
            "game_id": game_id, "provider_id": pid, "provider": name,
            # NB: the "value" field is sometimes decimal odds; "american" holds the actual line.
            "spread": _american(h, "close", "pointSpread", "american"),
            "spread_open": _american(h, "open", "pointSpread", "american"),
            "total": _american(it, "close", "total", "american"),
            "total_open": _american(it, "open", "total", "american"),
            "home_price": _american(h, "close", "spread", "american"),
            "away_price": _american(it.get("awayTeamOdds", {}), "close", "spread", "american"),
            "over_price": _american(it, "close", "over", "american"),
            "under_price": _american(it, "close", "under", "american"),
        }
        # Older seasons: no open/close blocks -> fall back to the posted (final) line.
        if np.isnan(row["total"]):
            row["total"] = _val(it, "overUnder")
        if np.isnan(row["spread"]) and not np.isnan(_val(it, "spread")):
            # ESPN's top-level `spread` is from the home team's perspective
            row["spread"] = _val(it, "spread")
        # Sanity filter: some entries carry prices (-110) in the line fields.
        for c in ("spread", "spread_open"):
            if not abs(row[c]) <= 50:
                row[c] = np.nan
        for c in ("total", "total_open"):
            if not 95 <= row[c] <= 210:
                row[c] = np.nan
        rows.append(row)
    return rows


def consolidate(book_rows):
    """Pick one line per game: preferred sportsbook if present, else median across books."""
    b = pd.DataFrame(book_rows)
    if b.empty:
        return b
    b["rank"] = b.provider_id.map({p: i for i, p in enumerate(PREFERRED)}).fillna(99)
    best = b.sort_values("rank").drop_duplicates("game_id").set_index("game_id")
    med = b.groupby("game_id")[["spread", "spread_open", "total", "total_open"]].median()
    out = best[["provider", "spread", "spread_open", "total", "total_open",
                "home_price", "away_price", "over_price", "under_price"]].copy()
    # Lines = median across books (robust to one book's bad entry); prices from the preferred book.
    for c in ("spread", "total", "spread_open", "total_open"):
        out[c] = med[c]
    out["n_books"] = b.groupby("game_id").size()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2022-2026")
    ap.add_argument("-o", "--out", default=str(C.LINES / "espn_2022_2026.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    a, b = map(int, args.seasons.split("-"))
    g = pd.read_parquet(C.PROCESSED / "games.parquet")
    g = g[g.season.between(a, b) & g.home_d1 & g.away_d1]
    sess = requests.Session()
    rows = []
    for season, gs in g.groupby("season"):
        ids = gs.game_id.tolist()
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda gid: fetch(gid, season, sess), ids))
        n_fail = sum(r is None for r in res)
        for gid, d in zip(ids, res):
            rows += parse(gid, d)
        print(f"  {season}: {len(ids):,} games, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
    lines = consolidate(rows)
    meta = g[["game_id", "game_date", "home_name", "away_name"]]
    lines = meta.merge(lines, on="game_id")
    lines.insert(1, "date", lines.pop("game_date").dt.strftime("%Y-%m-%d"))
    lines = lines.rename(columns={"home_name": "home", "away_name": "away"})
    lines.to_csv(args.out, index=False)
    print(f"{len(lines):,} games with lines -> {args.out}")
    for c in ["spread", "spread_open", "total", "total_open"]:
        print(f"  {c}: {lines[c].notna().mean():.1%}")


if __name__ == "__main__":
    main()
