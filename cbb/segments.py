"""Find where the model has an edge: segment analysis with discovery/validation split.

Discovery  = SBR lines 2014-2021.  Validation = ESPN lines 2022-2026 (opening totals 2024-26).
A segment only "holds" if it beats break-even in BOTH periods. Slicing data many ways will always
produce some lucky-looking slices; requiring out-of-period confirmation filters most of them.

Usage: python -m cbb.segments
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from . import config as C
from .lines import TeamMatcher, load_lines

POWER = {2, 4, 7, 8, 21, 23}            # ACC, Big East, Big Ten, Big 12, Pac-12, SEC
HIGH_MID = {3, 18, 29, 44, 62}          # A-10, MVC, WCC, Mountain West, American
MIN_PROB = 0.55
DISCOVERY = (2014, 2021)
VALIDATION = (2022, 2026)


def conference_tiers():
    rows = []
    for y in range(C.FIRST_SEASON, C.LAST_SEASON + 1):
        p = C.RAW / f"schedule_{y}.parquet"
        if not p.exists():
            continue
        cols = ["game_id", "home_conference_id", "away_conference_id"]
        extra = ["home_current_rank", "away_current_rank"]
        s = pd.read_parquet(p)
        s = s[[c for c in cols + extra if c in s.columns]]
        rows.append(s)
    s = pd.concat(rows)
    s["game_id"] = s.game_id.astype("int64")

    def tier(c):
        return np.where(pd.Series(c).isin(POWER), "power", np.where(pd.Series(c).isin(HIGH_MID), "high-mid", "low-major"))
    s["tier_h"] = tier(s.home_conference_id)
    s["tier_a"] = tier(s.away_conference_id)
    return s.drop(columns=["home_conference_id", "away_conference_id"])


def top25_cutoffs():
    finals = pd.read_pickle(C.PROCESSED / "final_ratings.pkl")
    cut = {}
    for season, (final, _) in finals.items():
        nets = sorted((o - d for t, (o, d, _) in final.items() if t != -1), reverse=True)
        cut[season + 1] = nets[24]  # use LAST season's 25th-best rating -> no look-ahead
    return cut


def add_segments(d):
    d = d.copy()
    cut = d.season.map(top25_cutoffs())
    t25h, t25a = d.net_h >= cut, d.net_a >= cut
    d["top25"] = np.select([t25h & t25a, t25h | t25a], ["both top-25", "one top-25"], "neither top-25")
    if "home_current_rank" in d:
        rh = d.home_current_rank.fillna(99) <= 25
        ra = d.away_current_rank.fillna(99) <= 25
        d["ap_ranked"] = np.where(d.home_current_rank.isna(), None,
                                  np.select([rh & ra, rh | ra], ["both AP", "one AP"], "no AP"))
    d["game_type"] = np.select([d.season_type == 3, d.conference_competition], ["postseason", "conference"], "non-conference")
    pair = np.sort(np.column_stack([d.tier_h.fillna("low-major"), d.tier_a.fillna("low-major")]), axis=1)
    d["tiers"] = [f"{a} v {b}" for a, b in pair]
    d["any_power"] = np.where((d.tier_h == "power") | (d.tier_a == "power"), "has power team", "no power team")
    d["month"] = d.game_date.dt.month.map({11: "1 Nov", 12: "2 Dec", 1: "3 Jan", 2: "4 Feb", 3: "5 Mar", 4: "5 Mar"})
    d["site"] = np.where(d.neutral_site, "neutral", "home/away")
    d["weekday"] = np.where(d.game_date.dt.dayofweek >= 5, "weekend", "weekday")
    d["pace"] = pd.cut(d.proj_poss, [0, 66, 69, 72, 200], labels=["slow <66", "66-69", "69-72", "fast 72+"])
    d["rest"] = np.where((d.rest_h <= 1) | (d.rest_a <= 1), "back-to-back", "normal rest")
    d["early_season"] = np.where(d.min_gp < 5, "<5 games played", "5+ games")
    d["mismatch"] = pd.cut((d.net_h - d.net_a).abs(), [0, 5, 12, 20, 200],
                           labels=["close <5", "5-12", "12-20", "blowout 20+"])
    return d


def grade_total(d, line_col):
    x = d.dropna(subset=[line_col]).copy()
    po = norm.cdf((x.ml_total - x[line_col]) / x.sd_total)
    x["prob"] = np.maximum(po, 1 - po)
    x["pick"] = np.where(po >= 0.5, "over", "under")
    x["edge"] = (x.ml_total - x[line_col]).abs()
    diff = x.total - x[line_col]
    x["win"] = np.where(diff == 0, np.nan, np.where(np.where(po >= 0.5, diff > 0, diff < 0), 1.0, 0.0))
    x["line_bucket"] = pd.cut(x[line_col], [0, 130, 140, 150, 160, 300],
                              labels=["<130", "130-140", "140-150", "150-160", "160+"])
    return x


def grade_spread(d):
    x = d.dropna(subset=["spread"]).copy()
    ph = norm.cdf((x.ml_margin + x.spread) / x.sd_margin)
    x["prob"] = np.maximum(ph, 1 - ph)
    home_pick = ph >= 0.5
    fav_home = x.spread < 0
    x["pick"] = np.where(home_pick == fav_home, "favorite", "underdog") + np.where(home_pick, " (home)", " (away)")
    x["edge"] = (x.ml_margin + x.spread).abs()
    ats = x.margin + x.spread
    x["win"] = np.where(ats == 0, np.nan, np.where(np.where(home_pick, ats > 0, ats < 0), 1.0, 0.0))
    x["line_bucket"] = pd.cut(x.spread.abs(), [-1, 3, 7, 12, 100], labels=["pk-3", "3.5-7", "7.5-12", "12.5+"])
    return x


SEGMENTS = ["top25", "game_type", "tiers", "any_power", "month", "site", "weekday", "pace", "rest",
            "early_season", "mismatch", "pick", "line_bucket"]


def _stats(x):
    x = x[x.win.notna()]
    n, w = len(x), int(x.win.sum())
    if n == 0:
        return pd.Series({"n": 0, "win%": np.nan, "p": np.nan})
    return pd.Series({"n": n, "win%": w / n, "p": binomtest(w, n, C.BREAKEVEN, alternative="greater").pvalue})


def segment_table(disc, val):
    out = []
    disc, val = disc[disc.prob >= MIN_PROB], val[val.prob >= MIN_PROB]
    for seg in SEGMENTS + (["ap_ranked"] if "ap_ranked" in val else []):
        if seg not in val:
            continue
        for level in sorted(set(val[seg].dropna().astype(str)) | set(disc[seg].dropna().astype(str)) if seg in disc else set(val[seg].dropna().astype(str))):
            a = _stats(disc[disc[seg].astype(str) == level]) if seg in disc else _stats(disc.iloc[0:0])
            b = _stats(val[val[seg].astype(str) == level])
            out.append({"segment": seg, "level": level,
                        "disc_n": a.n, "disc_win%": a["win%"], "disc_p": a.p,
                        "val_n": b.n, "val_win%": b["win%"], "val_p": b.p})
    t = pd.DataFrame(out)
    t["verdict"] = np.select(
        [(t["disc_win%"] > C.BREAKEVEN) & (t["val_win%"] > C.BREAKEVEN) & (t.disc_p < 0.05) & (t.val_n >= 150),
         (t["disc_win%"] < 0.50) & (t["val_win%"] < 0.50)],
        ["HOLDS", "AVOID"], "")
    return t


def load():
    pred = pd.read_parquet(C.PROCESSED / "predictions.parquet")
    pred = pred[pred.ml_margin.notna() & pred.margin.notna()]
    pred = pred.merge(conference_tiers(), on="game_id", how="left")
    matcher = TeamMatcher(sorted(C.RAW.glob("team_box_*.parquet")))
    sbr, _ = load_lines(C.LINES / "sbr_2014_2021.csv", pred[pred.season.between(*DISCOVERY)], matcher)
    espn, _ = load_lines(C.LINES / "espn_2022_2026.csv", pred[pred.season.between(*VALIDATION)], None)
    lines = pd.concat([sbr, espn]).drop_duplicates("game_id")
    d = pred.merge(lines.rename(columns={"total": "total_line"}), on="game_id")
    return add_segments(d)


def main():
    d = load()
    disc = d[d.season.between(*DISCOVERY)]
    val = d[d.season.between(*VALIDATION)]
    pd.set_option("display.width", 200)
    fmt = {"disc_win%": "{:.1%}".format, "val_win%": "{:.1%}".format, "disc_p": "{:.3f}".format, "val_p": "{:.3f}".format}
    results = {}
    for name, gd, gv in [
        ("TOTALS vs OPEN", grade_total(disc, "total_open"), grade_total(val, "total_open")),
        ("TOTALS vs CLOSE", grade_total(disc, "total_line"), grade_total(val, "total_line")),
        ("SPREADS vs CLOSE", grade_spread(disc), grade_spread(val)),
    ]:
        t = segment_table(gd, gv)
        results[name] = t
        print(f"\n===== {name}  (bets with model prob >= {MIN_PROB}) =====")
        allrow = segment_table(gd.assign(all="all"), gv.assign(all="all"))
        print(t.to_string(index=False, formatters=fmt))
        t.to_csv(C.REPORTS / f"segments_{name.split()[0].lower()}_{name.split()[-1].lower()}.csv", index=False)
    return results


if __name__ == "__main__":
    main()
