"""Convert SportsbookReviewsOnline NCAAB odds spreadsheets into the lines CSV format.

SBR layout: two rows per game (V then H, or N/N for neutral). In the Open/Close columns
the favorite's row holds the spread (as a positive number) and the other row holds the
total. 'pk' = pick'em, 'NL' = no line.

Usage: python -m cbb.sbr path/to/ncaa-basketball-2019-20.xlsx [...] -o data/lines/sbr.csv
"""
import argparse
import re

import numpy as np
import pandas as pd

TOTAL_MIN = 90  # numbers above this are totals, below are spreads


def _num(x):
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("pk", "pick", "p"):
            return 0.0
        s = s.replace("½", ".5")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return float(x) if pd.notna(x) else np.nan


def split_name(s):
    """'BostonCollege' -> 'Boston College', 'AppalachianSt' -> 'Appalachian St', 'NCState' -> 'NC State'."""
    s = str(s).strip()
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s.replace(".", "").strip()


def _lines(v, h):
    """Given (V value, H value) for one column -> (home_spread, total)."""
    spread, total = np.nan, np.nan
    for val, is_home in ((v, False), (h, True)):
        if np.isnan(val):
            continue
        if val >= TOTAL_MIN:
            total = val
        else:  # this team is the favorite by |val| (some dates write favorites as negative)
            val = abs(val)
            spread = -val if is_home else val
    return spread, total


def convert(path):
    d = pd.read_excel(path)
    m = re.search(r"(\d{4})-(\d{2})", str(path))
    y1 = int(m.group(1))
    rows = []
    vals = d.to_dict("records")
    i = 0
    while i + 1 < len(vals):
        v, h = vals[i], vals[i + 1]
        if not (str(v["VH"]).upper() in ("V", "N") and str(h["VH"]).upper() in ("H", "N")):
            i += 1
            continue
        mmdd = int(h["Date"])
        month, day = mmdd // 100, mmdd % 100
        year = y1 if month >= 7 else y1 + 1
        sc, tc = _lines(_num(v["Close"]), _num(h["Close"]))
        so, to = _lines(_num(v["Open"]), _num(h["Open"]))
        rows.append({
            "date": f"{year}-{month:02d}-{day:02d}",
            "home": split_name(h["Team"]), "away": split_name(v["Team"]),
            "spread": sc if not np.isnan(sc) else so, "total": tc if not np.isnan(tc) else to,
            "spread_open": so, "total_open": to,
            "neutral": int(str(h["VH"]).upper() == "N"),
            "sbr_home_final": h["Final"], "sbr_away_final": v["Final"],
        })
        i += 2
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    df = pd.concat([convert(f) for f in args.files], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} games -> {args.out}")
    print(f"  with spread: {df.spread.notna().mean():.1%}   with total: {df.total.notna().mean():.1%}")


def learn_aliases(lines_csv, games_parquet, alias_csv):
    """Learn SBR-name -> ESPN team_id by matching games on date + final score.

    Each SBR name gets the team_id it lines up with most often; requires >= 5 agreeing
    games and >= 80% agreement. Existing aliases are kept.
    """
    L = pd.read_csv(lines_csv)
    L["date"] = pd.to_datetime(L.date)
    g = pd.read_parquet(games_parquet, columns=["game_date", "home_id", "away_id", "pts_h", "pts_a"])
    votes = []
    for shift in (0, -1, 1):
        k = g.assign(date=g.game_date + pd.Timedelta(days=shift))
        a = L.merge(k, left_on=["date", "sbr_home_final", "sbr_away_final"], right_on=["date", "pts_h", "pts_a"])
        votes += [a[["home", "home_id"]].set_axis(["alias", "team_id"], axis=1),
                  a[["away", "away_id"]].set_axis(["alias", "team_id"], axis=1)]
        b = L.merge(k, left_on=["date", "sbr_home_final", "sbr_away_final"], right_on=["date", "pts_a", "pts_h"])
        votes += [b[["home", "away_id"]].set_axis(["alias", "team_id"], axis=1),
                  b[["away", "home_id"]].set_axis(["alias", "team_id"], axis=1)]
    v = pd.concat(votes).value_counts().rename("n").reset_index()
    tot = v.groupby("alias").n.transform("sum")
    best = v.sort_values("n", ascending=False).drop_duplicates("alias")
    best = best[(best.n >= 5) & (best.n / tot[best.index] >= 0.8)]
    try:
        old = pd.read_csv(alias_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        old = pd.DataFrame(columns=["alias", "team_id"])
    new = pd.concat([old, best[["alias", "team_id"]]]).drop_duplicates("alias", keep="first")
    new.to_csv(alias_csv, index=False)
    return len(new) - len(old)


if __name__ == "__main__":
    main()
