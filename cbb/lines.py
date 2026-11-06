"""Load a betting-lines CSV from any source and match it to ESPN games.

Expected CSV columns (header names are case-insensitive):
    date        game date (YYYY-MM-DD)
    home        home team name (any common spelling: "UConn", "Connecticut", "St. John's"...)
    away        away team name
    spread      HOME team spread, e.g. -5.5 means home favored by 5.5   (optional)
    total       over/under                                                  (optional)
Optional:
    spread_open, total_open        opening lines (enables CLV analysis; spread/total = closing)
    home_price, away_price         American odds for the spread (default -110)
    over_price, under_price        American odds for the total (default -110)
    neutral                        1/0 if your source marks neutral-site games (helps matching)

If a source lists games as team1/team2 for neutral sites, put either one in `home`;
the matcher tries the swapped orientation and flips the spread when needed.
"""
import difflib
import re

import numpy as np
import pandas as pd

from . import config as C

ALIAS_FILE = C.LINES / "team_aliases.csv"  # optional overrides: alias,team_id

_SUBS = [(r"\bst\b\.?", "state"), (r"\bsaint\b", "st"), (r"&", "and"), (r"\buniv(ersity)?\b", ""),
         (r"\bthe\b", ""), (r"[^a-z0-9 ]", ""), (r"\s+", " ")]


def norm(name):
    s = str(name).lower().strip()
    s = s.replace("st.", "st")
    s = re.sub(r"\bst\b(?=\s+\w)", "saint", s) if s.startswith("st ") else s  # leading "St" = Saint
    for pat, rep in _SUBS:
        s = re.sub(pat, rep, s)
    return s.strip()


COMMON = {  # frequent book spellings that differ from ESPN locations
    "connecticut": "UConn", "uconn": "UConn", "miami fl": "Miami", "miami florida": "Miami",
    "miami oh": "Miami (OH)", "miami ohio": "Miami (OH)", "ole miss": "Ole Miss", "mississippi": "Ole Miss",
    "nc state": "NC State", "north carolina state": "NC State", "unc": "North Carolina",
    "usc": "USC", "southern california": "USC", "lsu": "LSU", "ucf": "UCF", "smu": "SMU", "tcu": "TCU",
    "byu": "BYU", "vcu": "VCU", "unlv": "UNLV", "utep": "UTEP", "utsa": "UTSA", "uab": "UAB",
    "pitt": "Pittsburgh", "umass": "Massachusetts", "fiu": "Florida International", "fau": "Florida Atlantic",
    "etsu": "East Tennessee State", "siu edwardsville": "SIU Edwardsville", "liu": "Long Island University",
    "loyola chicago": "Loyola Chicago", "loyola il": "Loyola Chicago", "saint marys": "Saint Mary's",
    "st marys": "Saint Mary's", "st marys ca": "Saint Mary's", "saint johns": "St. John's", "st johns": "St. John's",
}


class TeamMatcher:
    def __init__(self, team_box_paths):
        cols = ["team_id", "team_location", "team_display_name", "team_short_display_name", "team_abbreviation"]
        tb = pd.concat([pd.read_parquet(p, columns=cols) for p in team_box_paths]).drop_duplicates()
        self.alias = {}
        for r in tb.itertuples(index=False):
            for n in (r.team_location, r.team_display_name, r.team_short_display_name):
                if isinstance(n, str):
                    self.alias.setdefault(norm(n), int(r.team_id))
        loc = {norm(r.team_location): int(r.team_id) for r in tb.itertuples(index=False) if isinstance(r.team_location, str)}
        for k, v in COMMON.items():
            if norm(v) in loc:
                self.alias[norm(k)] = loc[norm(v)]
        if ALIAS_FILE.exists():
            for r in pd.read_csv(ALIAS_FILE).itertuples(index=False):
                self.alias[norm(r.alias)] = int(r.team_id)
        self.keys = list(self.alias)

    def __call__(self, name):
        n = norm(name)
        if n in self.alias:
            return self.alias[n]
        m = difflib.get_close_matches(n, self.keys, n=1, cutoff=0.88)
        return self.alias[m[0]] if m else None


def load_lines(path, games, matcher):
    """Return lines merged onto `games` (needs game_id, game_date, home_id, away_id)."""
    L = pd.read_csv(path)
    L.columns = [c.lower().strip() for c in L.columns]
    keep_cols = ["spread", "total", "spread_open", "total_open", "home_price", "away_price",
                 "over_price", "under_price"]
    if "game_id" in L:  # ESPN-sourced lines: exact match, no name matching needed
        m = L.merge(games[["game_id"]], on="game_id")
        stats = {"rows": len(L), "matched": len(m), "unmatched_names": pd.Series(dtype=int)}
        return m[["game_id"] + [c for c in keep_cols if c in m]], stats
    L["date"] = pd.to_datetime(L["date"])
    L["home_tid"] = L["home"].map(matcher)
    L["away_tid"] = L["away"].map(matcher)
    unmatched = pd.concat([L.loc[L.home_tid.isna(), "home"], L.loc[L.away_tid.isna(), "away"]]).value_counts()
    L = L.dropna(subset=["home_tid", "away_tid"]).astype({"home_tid": int, "away_tid": int})

    keys = games[["game_id", "game_date", "home_id", "away_id"]]
    L["_rid"] = np.arange(len(L))
    out = []
    # Priority order: exact date first, then +/-1 day (time-zone slop); listed orientation before swapped.
    # Each line row and each game is used at most once -- matters for back-to-back series
    # (same two teams on consecutive days).
    for prio, shift in enumerate((0, -1, 1)):
        k = keys.assign(date=keys.game_date + pd.Timedelta(days=shift))
        direct = L.merge(k, left_on=["date", "home_tid", "away_tid"], right_on=["date", "home_id", "away_id"])
        swapped = L.merge(k, left_on=["date", "home_tid", "away_tid"], right_on=["date", "away_id", "home_id"])
        out += [direct.assign(flip=False, _prio=2 * prio), swapped.assign(flip=True, _prio=2 * prio + 1)]
    m = (pd.concat(out).sort_values("_prio", kind="stable")
         .drop_duplicates("_rid").drop_duplicates("game_id"))
    # Flip home/away-oriented fields if the source had the teams reversed.
    for col in ("spread", "spread_open"):
        if col in m:
            m.loc[m.flip, col] = -m.loc[m.flip, col]
    if {"home_price", "away_price"} <= set(m):
        hp = m.home_price.copy()
        m.loc[m.flip, "home_price"] = m.loc[m.flip, "away_price"]
        m.loc[m.flip, "away_price"] = hp[m.flip]
    keep = ["game_id"] + [c for c in ["spread", "total", "spread_open", "total_open", "home_price", "away_price",
                                      "over_price", "under_price"] if c in m]
    stats = {"rows": len(pd.read_csv(path)), "matched": len(m), "unmatched_names": unmatched}
    return m[keep], stats
