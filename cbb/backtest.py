"""Backtest the model's out-of-sample predictions against historical lines.

Usage:
    python -m cbb.backtest data/lines/my_lines.csv [--min-prob 0.55] [--seasons 2019-2026]

Every prediction used here comes from the walk-forward run (cbb.model): the model for
season S was trained only on seasons < S, and ratings for each game use only prior games.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from . import config as C
from .lines import TeamMatcher, load_lines


def payout(odds):
    """Profit per 1 unit risked at American odds."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 100 / -odds, odds / 100)


def grade_spreads(d):
    d = d[d.spread.notna()].copy()
    d["edge"] = d.ml_margin + d.spread                       # >0: model likes home side
    d["p_home"] = norm.cdf(d.edge / d.sd_margin)
    d["side"] = np.where(d.p_home >= 0.5, "home", "away")
    d["prob"] = np.maximum(d.p_home, 1 - d.p_home)
    ats = d.margin + d.spread                                # >0 home covered
    won = np.where(d.side == "home", ats > 0, ats < 0)
    d["result"] = np.where(ats == 0, "P", np.where(won, "W", "L"))
    price = np.where(d.side == "home", d.get("home_price", pd.Series(-110, index=d.index)).fillna(-110),
                     d.get("away_price", pd.Series(-110, index=d.index)).fillna(-110))
    d["profit"] = np.select([d.result == "W", d.result == "L"], [payout(price), -1.0], 0.0)
    if "spread_open" in d:
        # CLV in points: how far the line moved toward our side from open to close
        d["clv"] = np.where(d.side == "home", d.spread_open - d.spread, d.spread - d.spread_open)
    d["market"] = "spread"
    return d


def grade_totals(d):
    d = d[d.total_line.notna()].copy()
    d["edge"] = d.ml_total - d.total_line                    # >0: model likes over
    d["p_over"] = norm.cdf(d.edge / d.sd_total)
    d["side"] = np.where(d.p_over >= 0.5, "over", "under")
    d["prob"] = np.maximum(d.p_over, 1 - d.p_over)
    diff = d.total - d.total_line
    won = np.where(d.side == "over", diff > 0, diff < 0)
    d["result"] = np.where(diff == 0, "P", np.where(won, "W", "L"))
    price = np.where(d.side == "over", d.get("over_price", pd.Series(-110, index=d.index)).fillna(-110),
                     d.get("under_price", pd.Series(-110, index=d.index)).fillna(-110))
    d["profit"] = np.select([d.result == "W", d.result == "L"], [payout(price), -1.0], 0.0)
    if "total_open" in d:
        d["clv"] = np.where(d.side == "over", d.total_line - d.total_open, d.total_open - d.total_line)
    d["edge"] = d.edge.abs()
    d["market"] = "total"
    return d


def summarize(b):
    w, l, p = (b.result == "W").sum(), (b.result == "L").sum(), (b.result == "P").sum()
    n = w + l
    out = {"bets": len(b), "W": w, "L": l, "P": p,
           "win_pct": w / n if n else np.nan,
           "roi": b.profit.sum() / len(b) if len(b) else np.nan,
           "units": b.profit.sum(),
           "p_value_vs_52.4": binomtest(w, n, C.BREAKEVEN, alternative="greater").pvalue if n else np.nan}
    if "clv" in b and b.clv.notna().any():
        out["avg_clv_pts"] = b.clv.mean()
    return pd.Series(out)


def report(graded, label):
    lines = [f"\n=== {label}: {len(graded):,} games with lines ==="]
    lines.append("Market sanity check - closing-line error vs model error (MAE, pts):")
    tgt = "margin" if label == "SPREADS" else "total"
    if label == "SPREADS":
        lines.append(f"  line MAE {(graded.margin + graded.spread).abs().mean():.2f}  |  "
                     f"model MAE {(graded.margin - graded.ml_margin).abs().mean():.2f}")
    else:
        lines.append(f"  line MAE {(graded.total - graded.total_line).abs().mean():.2f}  |  "
                     f"model MAE {(graded.total - graded.ml_total).abs().mean():.2f}")
    rows = {}
    for t in [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]:
        rows[f"prob>={t:.2f}"] = summarize(graded[graded.prob >= t])
    lines.append("\nBy model confidence threshold:")
    lines.append(pd.DataFrame(rows).T.round(3).to_string())
    graded = graded.assign(edge_bucket=pd.cut(graded.edge.abs(), [0, 1, 2, 3, 4, 6, 100]))
    lines.append("\nBy edge size (pts vs line):")
    lines.append(graded.groupby("edge_bucket", observed=True).apply(summarize).round(3).to_string())
    return "\n".join(lines), graded


def segment_report(graded, min_prob):
    b = graded[graded.prob >= min_prob].copy()
    if b.empty:
        return ""
    b["month"] = b.game_date.dt.month_name().str[:3]
    out = [f"\n--- Segments for bets with prob >= {min_prob} ---"]
    for col in ["season", "month", "conf"]:
        out.append(f"\nBy {col}:")
        out.append(b.groupby(col).apply(summarize).round(3).to_string())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lines_csv")
    ap.add_argument("--min-prob", type=float, default=0.55)
    ap.add_argument("--seasons", default=None, help="e.g. 2019-2026")
    args = ap.parse_args()

    pred = pd.read_parquet(C.PROCESSED / "predictions.parquet")
    pred = pred[pred.ml_margin.notna() & pred.margin.notna()]
    if args.seasons:
        a, b = map(int, args.seasons.split("-"))
        pred = pred[pred.season.between(a, b)]
    matcher = TeamMatcher(sorted(C.RAW.glob("team_box_*.parquet")))
    L, stats = load_lines(args.lines_csv, pred, matcher)
    print(f"Lines file rows: {stats['rows']:,}  matched to games: {stats['matched']:,}")
    if len(stats["unmatched_names"]):
        print("Unmatched team names (add to data/lines/team_aliases.csv as alias,team_id):")
        print(stats["unmatched_names"].head(25).to_string())

    d = pred.merge(L, on="game_id")
    if d.empty:
        print("No lines matched any games with out-of-sample predictions - check dates/team names/seasons.")
        return
    if "total" in L:
        d = d.rename(columns={"total_x": "total", "total_y": "total_line"})
    text = []
    graded_all = []
    if "spread" in d:
        t, g = report(grade_spreads(d), "SPREADS"); text.append(t); text.append(segment_report(g, args.min_prob))
        graded_all.append(g)
    if "total_line" in d:
        t, g = report(grade_totals(d), "TOTALS"); text.append(t); text.append(segment_report(g, args.min_prob))
        graded_all.append(g)
    out = "\n".join(text)
    print(out)
    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "backtest.txt").write_text(out)
    cols = ["game_id", "game_date", "season", "home_name", "away_name", "market", "spread", "total_line",
            "ml_margin", "ml_total", "margin", "total", "side", "prob", "edge", "result", "profit"]
    pd.concat(graded_all)[lambda x: [c for c in cols if c in x]].to_csv(C.REPORTS / "backtest_bets.csv", index=False)
    print(f"\nSaved reports/backtest.txt and reports/backtest_bets.csv")


if __name__ == "__main__":
    main()
