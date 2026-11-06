"""Daily paper-trading run: refresh data, log today's picks, grade finished games.

Usage: python -m cbb.daily [--date YYYY-MM-DD] [--skip-refresh]

Ledger: data/paper/ledger.csv (one row per game, the pick is locked the first time it is logged).
Dashboard feed: data/paper/dashboard.json

Bet rule (from the backtest): OPENING total, calibrated model probability >= 0.56.
Grades against the opening total (what the backtest measured) and against the line
available when the pick was logged (what you could realistically bet).
"""
import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import config as C
from .calibrate import load as load_calibration, p_over
from .espn_odds import URL, consolidate, parse
from .segments import HIGH_MID, POWER

PAPER = C.ROOT / "data" / "paper"
LEDGER = PAPER / "ledger.csv"
FEED = PAPER / "dashboard.json"
PICKS_JSON = PAPER / "picks.json"  # rows for the web dashboard
PICK_COLS = ["game_id", "game_date", "away", "home", "tiers", "open_total", "line_at_pick", "model_total",
             "pick", "prob", "play", "status", "final_total", "result_open", "result_line", "units_open",
             "units_line", "spread_pick", "spread_at_pick", "model_spread_home", "logged_at"]
MIN_PROB = 0.56
STAKE = 1.0  # units per play


def refresh():
    from . import data, ratings
    data.download_all(refresh_current=True)
    g = data.build_games()
    g.to_parquet(C.PROCESSED / "games.parquet")
    ratings.main()


def fetch_lines(game_ids):
    sess = requests.Session()
    rows = []
    for gid in game_ids:
        try:
            rows += parse(int(gid), sess.get(URL.format(gid), timeout=30).json())
        except Exception as e:  # noqa: BLE001
            print(f"  odds fetch failed for {gid}: {e}")
    return consolidate(rows)


def tier_pair(game_ids):
    s = pd.read_parquet(C.RAW / f"schedule_{C.LAST_SEASON}.parquet",
                        columns=["game_id", "home_conference_id", "away_conference_id"])
    s["game_id"] = s.game_id.astype("int64")
    t = lambda c: np.where(c.isin(POWER), "power", np.where(c.isin(HIGH_MID), "high-mid", "low-major"))  # noqa: E731
    s["tiers"] = [" v ".join(sorted(x)) for x in zip(t(s.home_conference_id), t(s.away_conference_id))]
    return s.set_index("game_id").tiers.reindex(game_ids).values


def payout(odds):
    odds = float(odds) if pd.notna(odds) else -110.0
    return 100 / -odds if odds < 0 else odds / 100


def log_picks(day):
    from .predict import project_upcoming
    res = project_upcoming(1, day)
    if res is None:
        print(f"No games scheduled on {day}.")
        return pd.DataFrame()
    out, _, sd_m, _ = res
    lines = fetch_lines(out.game_id)
    if lines.empty:
        print("No lines posted yet.")
        return pd.DataFrame()
    df = out.merge(lines, on="game_id")
    df = df[df.total_open.notna() | df.total.notna()].copy()
    cal = load_calibration()
    ref = df.total_open.fillna(df.total)  # grade line: opener (fallback: current)
    po = p_over(cal, df.ml_total, ref)
    from scipy.stats import norm
    ph = norm.cdf((df.ml_margin + df.spread) / sd_m)
    now = datetime.now(ZoneInfo("America/New_York"))
    rec = pd.DataFrame({
        "logged_at": now.strftime("%Y-%m-%d %H:%M ET"),
        "game_id": df.game_id.astype("int64"),
        "game_date": pd.to_datetime(df.game_date).dt.strftime("%Y-%m-%d"),
        "away": df.away_name, "home": df.home_name,
        "tiers": tier_pair(df.game_id.astype("int64")),
        "model_total": df.ml_total.round(1),
        "open_total": df.total_open, "line_at_pick": df.total, "over_price": df.over_price,
        "under_price": df.under_price,
        "pick": np.where(po >= 0.5, "OVER", "UNDER"),
        "prob": np.maximum(po, 1 - po).round(3),
        "model_spread_home": (-df.ml_margin).round(1), "spread_at_pick": df.spread,
        "spread_pick": np.where(df.spread.isna(), "", np.where(ph >= 0.5, df.home_name, df.away_name)),
        "spread_prob": np.maximum(ph, 1 - ph).round(3),
        "books": df.n_books,
    })
    rec["play"] = rec.prob >= MIN_PROB
    rec["final_total"] = np.nan
    rec["result_open"] = ""
    rec["result_line"] = ""
    rec["units_open"] = np.nan
    rec["units_line"] = np.nan
    rec["status"] = "pending"
    return rec


def grade(ledger):
    g = pd.read_parquet(C.PROCESSED / "games.parquet", columns=["game_id", "total"])
    fin = g.set_index("game_id").total
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    for i, r in ledger[ledger.status == "pending"].iterrows():
        if r.game_id in fin.index:
            tot = fin[r.game_id]
            ledger.at[i, "final_total"] = tot
            for line_col, res_col, u_col in (("open_total", "result_open", "units_open"),
                                             ("line_at_pick", "result_line", "units_line")):
                line = r[line_col]
                if pd.isna(line):
                    continue
                if tot == line:
                    res, u = "P", 0.0
                else:
                    won = (tot > line) if r.pick == "OVER" else (tot < line)
                    price = r.over_price if r.pick == "OVER" else r.under_price
                    res, u = ("W", STAKE * payout(price)) if won else ("L", -STAKE)
                ledger.at[i, res_col] = res
                ledger.at[i, u_col] = round(u, 3) if r.play else np.nan
            ledger.at[i, "status"] = "final"
        elif pd.Timestamp(r.game_date) < today - pd.Timedelta(days=3):
            ledger.at[i, "status"] = "no result"
    return ledger


def summary(ledger):
    p = ledger[(ledger.play) & (ledger.status == "final")]
    def rec(col, ucol):
        w, l, pu = (p[col] == "W").sum(), (p[col] == "L").sum(), (p[col] == "P").sum()
        return {"W": int(w), "L": int(l), "P": int(pu), "win_pct": round(w / (w + l), 4) if w + l else None,
                "units": round(float(p[ucol].sum()), 2), "bets": int(w + l + pu)}
    daily = (p.groupby("game_date").agg(units=("units_line", "sum"), bets=("game_id", "size"))
             .sort_index().assign(cum_units=lambda x: x.units.cumsum()).round(2).reset_index())
    return {"vs_open": rec("result_open", "units_open"), "vs_line_at_pick": rec("result_line", "units_line"),
            "pending_plays": int(((ledger.play) & (ledger.status == "pending")).sum()),
            "daily": daily.to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="day to log picks for (default: today, US Eastern)")
    ap.add_argument("--skip-refresh", action="store_true")
    args = ap.parse_args()
    day = args.date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    PAPER.mkdir(parents=True, exist_ok=True)
    if not args.skip_refresh:
        print("Refreshing data and ratings...")
        refresh()
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    new = log_picks(day)
    if not new.empty:
        if not ledger.empty:
            new = new[~new.game_id.isin(ledger.game_id)]  # picks are locked once logged
        ledger = pd.concat([ledger, new], ignore_index=True)
        print(f"Logged {len(new)} new games for {day} ({int(new.play.sum())} plays).")
    if ledger.empty:
        return
    ledger["play"] = ledger.play.astype(bool)
    ledger = grade(ledger)
    ledger.to_csv(LEDGER, index=False)
    feed = {"updated": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
            "rule": f"Opening total, calibrated model probability >= {MIN_PROB}, 1 unit per play",
            "summary": summary(ledger),
            "games": json.loads(ledger.sort_values(["game_date", "prob"], ascending=[False, False])
                                .to_json(orient="records"))}
    FEED.write_text(json.dumps(feed, indent=1))
    rows = ledger[PICK_COLS].copy()
    rows["game_id"] = rows.game_id.astype(str)
    rows = rows.sort_values(["game_date", "prob"], ascending=[False, False])
    PICKS_JSON.write_text(rows.to_json(orient="records"))
    s = feed["summary"]
    print(f"Record vs open: {s['vs_open']}  |  vs line at pick: {s['vs_line_at_pick']}  |  pending plays: {s['pending_plays']}")
    today_plays = ledger[(ledger.game_date == day) & ledger.play]
    if len(today_plays):
        print(f"\nPlays for {day}:")
        print(today_plays[["away", "home", "open_total", "line_at_pick", "model_total", "pick", "prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
