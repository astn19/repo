"""Project upcoming games with current ratings + the ML layer; flag edges vs. posted lines.

Usage:
    python -m cbb.predict                       # next 3 days of games
    python -m cbb.predict --days 1 --lines data/lines/today.csv --min-prob 0.55

Run `python -m cbb.data --refresh-current && python -m cbb.ratings` first to pick up the
latest results.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from .features import build
from .lines import TeamMatcher, load_lines
from .model import fit_final
from .ratings import OTHER, RatingParams, project


def upcoming_games(days, start=None):
    s = pd.read_parquet(C.RAW / f"schedule_{C.LAST_SEASON}.parquet")
    s["game_date"] = pd.to_datetime(s.game_date)
    start = pd.Timestamp(start) if start else pd.Timestamp.today().normalize()
    s = s[(s.status_type_completed != True) & (s.game_date >= start) &  # noqa: E712
          (s.game_date < start + pd.Timedelta(days=days))]
    return pd.DataFrame({
        "game_id": s.game_id.astype("int64"), "season": C.LAST_SEASON, "game_date": s.game_date,
        "home_id": s.home_id.astype("int64"), "away_id": s.away_id.astype("int64"),
        "home_name": s.home_display_name, "away_name": s.away_display_name,
        "neutral_site": s.neutral_site.fillna(False).astype(bool),
        "conference_competition": s.conference_competition.fillna(False).astype(bool),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--lines", default=None, help="lines CSV for these games")
    ap.add_argument("--min-prob", type=float, default=0.55)
    args = ap.parse_args()

    rated = pd.read_parquet(C.PROCESSED / "games_rated.parquet")
    finals = pd.read_pickle(C.PROCESSED / "final_ratings.pkl")
    p = RatingParams()
    cur, meta = finals.get(C.LAST_SEASON, (None, None))
    prev, _ = finals[C.LAST_SEASON - 1]
    if cur is None:  # season hasn't started: use regressed last-season ratings
        cur = {t: (p.prior_regress * o, p.prior_regress * d, p.pace_regress * pc) for t, (o, d, pc) in prev.items()}
        meta = {**finals[C.LAST_SEASON - 1][1], "games_played": {}}

    up = upcoming_games(args.days, args.start)
    if up.empty:
        print("No upcoming games found in the schedule for that window.")
        return
    d1 = {t for t in list(cur) + list(prev) if t != OTHER}
    up["home_d1"] = up.home_id.isin(d1)
    up["away_d1"] = up.away_id.isin(d1)

    def rating(tid, is_d1):
        key = tid if is_d1 else OTHER
        if key in cur:
            return cur[key]
        if key in prev:
            o, dd, pc = prev[key]
            return p.prior_regress * o, p.prior_regress * dd, p.pace_regress * pc
        return p.new_team_prior / 2, -p.new_team_prior / 2, 0.0

    rh = np.array([rating(t, k) for t, k in zip(up.home_id, up.home_d1)])
    ra = np.array([rating(t, k) for t, k in zip(up.away_id, up.away_d1)])
    up["O_h"], up["D_h"], up["P_h"] = rh.T
    up["O_a"], up["D_a"], up["P_a"] = ra.T
    up["mu"], up["hca"], up["mu_p"] = meta["mu"], meta["hca"], meta["mu_p"]
    gp = meta.get("games_played", {})
    up["gp_h"] = up.home_id.map(gp).fillna(0)
    up["gp_a"] = up.away_id.map(gp).fillna(0)
    up = up.join(project(up))

    # Features need team history -> append upcoming rows (no box score) to completed games.
    box = [c for c in rated.columns if c.endswith(("_h", "_a")) and c not in up.columns]
    for c in box:
        up[c] = 0.0
    up["poss"] = np.nan
    combo = pd.concat([rated, up], ignore_index=True)
    df, feats = build(combo)
    df_hist = df[df.margin.notna()]
    mm, mt, sd_m, sd_t = fit_final(df_hist, feats)
    f = df[df.game_id.isin(up.game_id) & df.margin.isna()].copy()
    f["ml_margin"] = f.proj_margin + mm.predict(f[feats])
    f["ml_total"] = f.proj_total + mt.predict(f[feats])

    out = f[["game_id", "game_date", "away_name", "home_name", "neutral_site", "ml_margin", "ml_total"]].copy()
    out["model_home_spread"] = -out.ml_margin.round(1)
    out["model_total"] = out.ml_total.round(1)

    if args.lines:
        matcher = TeamMatcher(sorted(C.RAW.glob("team_box_*.parquet")))
        L, stats = load_lines(args.lines, f, matcher)
        print(f"Lines matched: {stats['matched']}/{stats['rows']}")
        if len(stats["unmatched_names"]):
            print("Unmatched names:", ", ".join(stats["unmatched_names"].index[:20]))
        out = out.merge(L, on="game_id", how="left")
        if "spread" in out:
            ph = norm.cdf((out.ml_margin + out.spread) / sd_m)
            out["spread_pick"] = np.where(ph >= 0.5, out.home_name, out.away_name)
            out["spread_prob"] = np.maximum(ph, 1 - ph).round(3)
        if "total" in out:
            po = norm.cdf((out.ml_total - out.total) / sd_t)
            out["total_pick"] = np.where(po >= 0.5, "OVER", "UNDER")
            out["total_prob"] = np.maximum(po, 1 - po).round(3)

    pd.set_option("display.width", 220)
    show = [c for c in ["game_date", "away_name", "home_name", "neutral_site", "model_home_spread", "spread",
                        "spread_pick", "spread_prob", "model_total", "total", "total_pick", "total_prob"] if c in out]
    print(out[show].sort_values("game_date").to_string(index=False))
    if "spread_prob" in out or "total_prob" in out:
        print(f"\n*** Plays (model probability >= {args.min_prob}) ***")
        if "spread_prob" in out:
            print(out[out.spread_prob >= args.min_prob][["game_date", "away_name", "home_name", "spread",
                                                         "model_home_spread", "spread_pick", "spread_prob"]].to_string(index=False))
        if "total_prob" in out:
            print(out[out.total_prob >= args.min_prob][["game_date", "away_name", "home_name", "total",
                                                        "model_total", "total_pick", "total_prob"]].to_string(index=False))
    C.REPORTS.mkdir(exist_ok=True)
    path = C.REPORTS / f"predictions_{pd.Timestamp.today():%Y%m%d}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
