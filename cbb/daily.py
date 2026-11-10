"""Daily paper-trading run: refresh data, log today's picks, grade finished games.

Usage: python -m cbb.daily [--date YYYY-MM-DD] [--skip-refresh]

Ledger: data/paper/ledger.csv (one row per game, the pick is locked the first time it is logged).
Dashboard feed: data/paper/dashboard.json

Bet rule (from the backtest): OPENING totals and spreads, calibrated confidence >= 0.56,\nat most 5 plays a day (most confident first).
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
from .calibrate import SPREAD_SCALE, load as load_calibration, load_spread, p_home_cover, p_over
from .espn_odds import URL, consolidate, parse
from .segments import HIGH_MID, POWER

PAPER = C.ROOT / "data" / "paper"
LEDGER = PAPER / "ledger.csv"
FEED = PAPER / "dashboard.json"
PICKS_JSON = PAPER / "picks.json"  # rows for the web dashboard
PICK_COLS = ["key", "game_id", "market", "game_date", "away", "home", "tiers", "pick", "pick_side", "open_total",
             "line_at_pick", "model_total", "prob", "raw_prob", "play", "status", "final", "final_total",
             "result_open", "result_line", "units_open", "units_line", "logged_at"]
MIN_PROB = 0.56   # totals prob; spreads use the rescaled prob (raw 0.53)
MAX_PLAYS = 5     # per day, most confident first, totals and spreads together
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


def short_names(game_ids):
    s = pd.read_parquet(C.RAW / f"schedule_{C.LAST_SEASON}.parquet",
                        columns=["game_id", "home_short_display_name", "away_short_display_name"])
    s = s.assign(game_id=s.game_id.astype("int64")).drop_duplicates("game_id").set_index("game_id").reindex(game_ids)
    return s.home_short_display_name.values, s.away_short_display_name.values


def candidates(day):
    """Every game with an opening line -> one total row and one spread row (not yet chosen as plays)."""
    from .predict import project_upcoming
    res = project_upcoming(1, day)
    if res is None:
        print(f"No games scheduled on {day}.")
        return pd.DataFrame()
    out, f, _, _ = res
    lines = fetch_lines(out.game_id)
    if lines.empty:
        print("No lines posted yet.")
        return pd.DataFrame()
    df = out.merge(f[["game_id", "min_gp"]], on="game_id").merge(lines, on="game_id")
    gid = df.game_id.astype("int64")
    hs, as_ = short_names(gid)
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    base = pd.DataFrame({"logged_at": now, "game_id": gid, "game_date": pd.to_datetime(df.game_date).dt.strftime("%Y-%m-%d"),
                         "away": df.away_name.values, "home": df.home_name.values, "tiers": tier_pair(gid),
                         "books": df.n_books.values})
    rows = []
    # Totals: calibrated P(over) vs the opening total.
    tc = load_calibration()
    t = df.total_open.notna()
    po = p_over(tc, df.ml_total, df.total_open.fillna(145))
    over = po >= 0.5
    rows.append(base[t].assign(
        market="total", pick=np.where(over, "OVER", "UNDER")[t], pick_side=np.where(over, "over", "under")[t],
        open_total=df.total_open[t].values, line_at_pick=df.total[t].values, model_total=df.ml_total[t].round(1).values,
        price=np.where(over, df.over_price, df.under_price)[t], raw_prob=np.maximum(po, 1 - po)[t].round(3)))
    # Spreads: calibrated P(home covers) vs the opening spread; lines shown from the picked team's side.
    sc = load_spread()
    sp = df.spread_open.notna()
    ph = p_home_cover(sc, df.ml_margin, df.spread_open.fillna(0), df.min_gp)
    home = ph >= 0.5
    sign = np.where(home, 1, -1)
    rows.append(base[sp].assign(
        market="spread", pick=np.where(home, hs, as_)[sp], pick_side=np.where(home, "home", "away")[sp],
        open_total=(sign * df.spread_open)[sp].values, line_at_pick=(sign * df.spread)[sp].values,
        model_total=(-sign * df.ml_margin)[sp].round(1).values,
        price=np.where(home, df.home_price, df.away_price)[sp], raw_prob=np.maximum(ph, 1 - ph)[sp].round(3)))
    rec = pd.concat(rows, ignore_index=True)
    rec["prob"] = np.where(rec.market == "spread", 0.5 + SPREAD_SCALE * (rec.raw_prob - 0.5), rec.raw_prob).round(3)
    rec["key"] = rec.game_id.astype(str) + "-" + rec.market
    for c, v in (("final_total", np.nan), ("final", ""), ("result_open", ""), ("result_line", ""),
                 ("units_open", np.nan), ("units_line", np.nan), ("status", "pending"), ("play", False)):
        rec[c] = v
    return rec


def choose_plays(rec, already):
    """Top MAX_PLAYS by confidence (>= MIN_PROB), across totals and spreads, minus plays already logged that day."""
    room = max(0, MAX_PLAYS - already)
    top = rec[rec.prob >= MIN_PROB].sort_values("prob", ascending=False).head(room).index
    rec.loc[top, "play"] = True
    return rec


SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
              "?dates={}&groups=50&limit=500")


def espn_finals(dates):
    """(home pts, away pts) straight from ESPN's scoreboard (the game-data release can lag a day)."""
    out = {}
    for d in sorted(set(dates)):
        try:
            evs = requests.get(SCOREBOARD.format(d.replace("-", "")), timeout=30).json().get("events", [])
        except Exception as e:  # noqa: BLE001
            print(f"  scoreboard fetch failed for {d}: {e}")
            continue
        for ev in evs:
            comp = ev["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            try:
                pts = {t["homeAway"]: int(t["score"]) for t in comp["competitors"]}
                out[int(ev["id"])] = (pts["home"], pts["away"])
            except (KeyError, ValueError):
                pass
    return out


def migrate(ledger):
    """Old ledgers had one totals row per game; give them the per-market columns."""
    if ledger.empty or "market" in ledger:
        return ledger
    ledger = ledger.copy()
    ledger["market"] = "total"
    ledger["key"] = ledger.game_id.astype("int64").astype(str) + "-total"
    ledger["pick_side"] = ledger.pick.str.lower()
    ledger["price"] = np.where(ledger.pick == "OVER", ledger.over_price, ledger.under_price)
    ledger["raw_prob"] = ledger.prob
    ledger["final"] = ledger.final_total.map(lambda v: "" if pd.isna(v) else str(int(v)))
    return ledger.drop(columns=[c for c in ["over_price", "under_price", "model_spread_home", "spread_at_pick",
                                            "spread_pick", "spread_prob"] if c in ledger])


def grade(ledger):
    g = pd.read_parquet(C.PROCESSED / "games.parquet", columns=["game_id", "pts_h", "pts_a"]).dropna()
    fin = {int(r.game_id): (int(r.pts_h), int(r.pts_a)) for r in g.itertuples()}
    pend = ledger[ledger.status == "pending"]
    if len(pend):
        days = set(pend.game_date)
        days |= {(pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in days}  # late tips
        fin.update(espn_finals(days))
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    for i, r in ledger[ledger.status == "pending"].iterrows():
        if int(r.game_id) not in fin:
            if pd.Timestamp(r.game_date) < today - pd.Timedelta(days=3):
                ledger.at[i, "status"] = "no result"
            continue
        h, a = fin[int(r.game_id)]
        if r.market == "total":
            ledger.at[i, "final_total"] = h + a
            ledger.at[i, "final"] = str(h + a)
            score = (h + a) if r.pick_side == "over" else -(h + a)       # beat the line if score + sign*line > 0
            lines = {"open_total": -r.open_total if r.pick_side == "over" else r.open_total,
                     "line_at_pick": -r.line_at_pick if r.pick_side == "over" else r.line_at_pick}
        else:
            mine, theirs = (h, a) if r.pick_side == "home" else (a, h)
            ledger.at[i, "final_total"] = mine - theirs
            ledger.at[i, "final"] = f"{mine}-{theirs}"
            score = mine - theirs
            lines = {"open_total": r.open_total, "line_at_pick": r.line_at_pick}
        for line_col, res_col, u_col in (("open_total", "result_open", "units_open"),
                                         ("line_at_pick", "result_line", "units_line")):
            if pd.isna(lines[line_col]):
                continue
            m = score + lines[line_col]
            res, u = ("P", 0.0) if m == 0 else (("W", STAKE * payout(r.price)) if m > 0 else ("L", -STAKE))
            ledger.at[i, res_col] = res
            ledger.at[i, u_col] = round(u, 3) if r.play else np.nan
        ledger.at[i, "status"] = "final"
    return ledger


def summary(ledger):
    p = ledger[(ledger.play) & (ledger.status == "final")]
    def rec(x, col, ucol):
        w, l, pu = (x[col] == "W").sum(), (x[col] == "L").sum(), (x[col] == "P").sum()
        return {"W": int(w), "L": int(l), "P": int(pu), "win_pct": round(float(w / (w + l)), 4) if w + l else None,
                "units": round(float(x[ucol].sum()), 2), "bets": int(w + l + pu)}
    daily = (p.groupby("game_date").agg(units=("units_line", "sum"), bets=("game_id", "size"))
             .sort_index().assign(cum_units=lambda x: x.units.cumsum()).round(2).reset_index())
    return {"vs_open": rec(p, "result_open", "units_open"), "vs_line_at_pick": rec(p, "result_line", "units_line"),
            "by_market": {m: rec(p[p.market == m], "result_open", "units_open") for m in ("total", "spread")},
            "pending_plays": int(((ledger.play) & (ledger.status == "pending")).sum()),
            "daily": daily.to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="day to log picks for (default: today, US Eastern)")
    ap.add_argument("--skip-refresh", action="store_true")
    ap.add_argument("--relog", action="store_true", help="drop the day's still-pending rows and pick again")
    args = ap.parse_args()
    day = args.date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    PAPER.mkdir(parents=True, exist_ok=True)
    if not args.skip_refresh:
        print("Refreshing data and ratings...")
        refresh()
    ledger = migrate(pd.read_csv(LEDGER, dtype={"final": str})) if LEDGER.exists() else pd.DataFrame()
    if not ledger.empty:
        ledger["play"] = ledger.play.astype(str).str.lower().eq("true")
        if args.relog:
            ledger = ledger[~((ledger.game_date == day) & (ledger.status == "pending"))]
    new = candidates(day)
    if not new.empty:
        already = 0
        if not ledger.empty:
            new = new[~new.key.isin(ledger.key)]  # picks are locked once logged
            already = int((ledger.play & (ledger.game_date == day)).sum())
        new = choose_plays(new, already)
        ledger = pd.concat([ledger, new], ignore_index=True)
        print(f"Logged {len(new)} new lines for {day} ({int(new.play.sum())} plays).")
    if ledger.empty:
        return
    ledger["play"] = ledger.play.astype(bool)
    ledger = grade(ledger)
    ledger.to_csv(LEDGER, index=False)
    feed = {"updated": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
            "rule": f"Opening totals and spreads, confidence >= {MIN_PROB}, top {MAX_PLAYS} a day, 1 unit per play",
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
        print(today_plays[["away", "home", "market", "pick", "open_total", "line_at_pick", "model_total", "prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
