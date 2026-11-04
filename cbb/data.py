"""Download ESPN box scores/schedules and build a one-row-per-game table.

Usage: python -m cbb.data [--refresh-current]
"""
import argparse
import sys

import numpy as np
import pandas as pd
import requests

from . import config as C


def _download(url, dest, force=False):
    if dest.exists() and not force:
        return
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_all(refresh_current=False):
    C.RAW.mkdir(parents=True, exist_ok=True)
    for season in range(C.FIRST_SEASON, C.LAST_SEASON + 1):
        force = refresh_current and season == C.LAST_SEASON
        for kind, url in (("team_box", C.TEAM_BOX_URL), ("schedule", C.SCHEDULE_URL)):
            dest = C.RAW / f"{kind}_{season}.parquet"
            try:
                _download(url.format(season=season), dest, force)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {kind} {season}: {e}", file=sys.stderr)
        print(f"  season {season} ok")


BOX_COLS = {
    "team_score": "pts", "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb",
    "total_turnovers": "tov", "assists": "ast", "steals": "stl", "blocks": "blk", "fouls": "pf",
}


def build_games():
    """Return one row per completed game, home/away oriented (neutral flagged)."""
    rows = []
    for season in range(C.FIRST_SEASON, C.LAST_SEASON + 1):
        tb_path = C.RAW / f"team_box_{season}.parquet"
        sc_path = C.RAW / f"schedule_{season}.parquet"
        if not tb_path.exists():
            continue
        tb = pd.read_parquet(tb_path)
        tb = tb[tb.team_home_away.isin(["home", "away"])]
        keep = ["game_id", "season", "season_type", "game_date", "team_id",
                "team_display_name", "team_home_away"] + list(BOX_COLS)
        tb = tb[keep].rename(columns=BOX_COLS)
        h = tb[tb.team_home_away == "home"].drop(columns="team_home_away")
        a = tb[tb.team_home_away == "away"].drop(columns=["team_home_away", "season", "season_type", "game_date"])
        g = h.merge(a, on="game_id", suffixes=("_h", "_a"))
        g = g.rename(columns={"team_id_h": "home_id", "team_id_a": "away_id",
                              "team_display_name_h": "home_name", "team_display_name_a": "away_name"})
        if sc_path.exists():
            sc = pd.read_parquet(sc_path, columns=["game_id", "neutral_site", "conference_competition",
                                                   "venue_address_state", "status_type_completed",
                                                   "status_period"])
            g = g.merge(sc, on="game_id", how="left")
        else:
            g["neutral_site"] = False
            g["conference_competition"] = False
            g["status_period"] = 2
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    g["game_date"] = pd.to_datetime(g.game_date)
    for c in ["home_id", "away_id", "game_id"]:
        g[c] = g[c].astype("int64")
    g["neutral_site"] = g.neutral_site.fillna(False).astype(bool)
    g["conference_competition"] = g.conference_competition.fillna(False).astype(bool)
    g = g[(g.pts_h > 0) & (g.pts_a > 0) & g.fga_h.notna() & g.fga_a.notna()]
    g = g.drop_duplicates("game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # Possessions (average of both teams' estimates).
    def poss(s):
        return g[f"fga_{s}"] - g[f"orb_{s}"] + g[f"tov_{s}"] + 0.475 * g[f"fta_{s}"]
    g["poss"] = (poss("h") + poss("a")) / 2
    g["n_ot"] = (pd.to_numeric(g.status_period, errors="coerce").fillna(2) - 2).clip(0, 6)
    # Pace per 40 minutes (overtime removed) — used for tempo ratings.
    g["poss40"] = g.poss * 40 / (40 + 5 * g.n_ot)
    # Overtime games inflate totals/possessions; normalise efficiency only (poss used as-is for OE).
    g["margin"] = g.pts_h - g.pts_a
    g["total"] = g.pts_h + g.pts_a
    g = g[(g.poss > 40) & (g.poss < 110)]

    # Flag D1 teams per season.
    cnt = pd.concat([g[["season", "home_id"]].rename(columns={"home_id": "tid"}),
                     g[["season", "away_id"]].rename(columns={"away_id": "tid"})]).value_counts()
    d1 = set(cnt[cnt >= C.MIN_GAMES_D1].index)
    # In-progress season: nobody has 10 games yet, so also accept last season's D1 teams.
    d1 |= {(s + 1, t) for s, t in d1 if s + 1 == C.LAST_SEASON}
    g["home_d1"] = [(s, t) in d1 for s, t in zip(g.season, g.home_id)]
    g["away_d1"] = [(s, t) in d1 for s, t in zip(g.season, g.away_id)]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-current", action="store_true", help="re-download the in-progress season")
    args = ap.parse_args()
    print("Downloading...")
    download_all(args.refresh_current)
    g = build_games()
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    g.to_parquet(C.PROCESSED / "games.parquet")
    print(f"Built {len(g):,} games, seasons {g.season.min()}-{g.season.max()}")
    print(g.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
