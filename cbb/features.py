"""Pregame features (strictly from information available before tip-off)."""
import numpy as np
import pandas as pd


def _long(g):
    """One row per team-game with that team's offensive/defensive box stats."""
    def side(s, o, is_home):
        d = pd.DataFrame({
            "game_id": g.game_id, "season": g.season, "game_date": g.game_date,
            "team": g[f"{'home' if s == 'h' else 'away'}_id"], "is_home": is_home,
            "fgm": g[f"fgm_{s}"], "fga": g[f"fga_{s}"], "fg3m": g[f"fg3m_{s}"], "fg3a": g[f"fg3a_{s}"],
            "fta": g[f"fta_{s}"], "ftm": g[f"ftm_{s}"], "orb": g[f"orb_{s}"], "tov": g[f"tov_{s}"],
            "o_fgm": g[f"fgm_{o}"], "o_fga": g[f"fga_{o}"], "o_fg3m": g[f"fg3m_{o}"], "o_fg3a": g[f"fg3a_{o}"],
            "o_fta": g[f"fta_{o}"], "o_orb": g[f"orb_{o}"], "o_tov": g[f"tov_{o}"],
            "drb": g[f"drb_{s}"], "o_drb": g[f"drb_{o}"], "poss": g.poss,
        })
        sign = 1 if s == "h" else -1
        if "proj_margin" in g:
            d["resid"] = sign * (g.margin - g.proj_margin)       # beat the projection by
            d["resid_t"] = g.total - g.proj_total
        return d
    return pd.concat([side("h", "a", 1), side("a", "h", 0)], ignore_index=True)


SUM_COLS = ["fgm", "fga", "fg3m", "fg3a", "fta", "ftm", "orb", "tov", "o_fgm", "o_fga", "o_fg3m",
            "o_fg3a", "o_fta", "o_orb", "o_tov", "drb", "o_drb", "poss"]


def team_form(g):
    L = _long(g).sort_values(["team", "game_date", "game_id"])
    grp = L.groupby(["season", "team"], sort=False)
    cum = grp[SUM_COLS].cumsum() - L[SUM_COLS]  # season-to-date, excluding current game
    f = pd.DataFrame({"game_id": L.game_id, "team": L.team})
    eps = 1e-9
    f["efg"] = (cum.fgm + 0.5 * cum.fg3m) / (cum.fga + eps)
    f["d_efg"] = (cum.o_fgm + 0.5 * cum.o_fg3m) / (cum.o_fga + eps)
    f["tov_r"] = cum.tov / (cum.poss + eps)
    f["d_tov_r"] = cum.o_tov / (cum.poss + eps)
    f["orb_r"] = cum.orb / (cum.orb + cum.o_drb + eps)
    f["drb_r"] = cum.drb / (cum.drb + cum.o_orb + eps)
    f["ftr"] = cum.fta / (cum.fga + eps)
    f["d_ftr"] = cum.o_fta / (cum.o_fga + eps)
    f["three_rate"] = cum.fg3a / (cum.fga + eps)
    f["d_three_rate"] = cum.o_fg3a / (cum.o_fga + eps)
    f["ft_pct"] = cum.ftm / (cum.fta + eps)
    f.loc[cum.fga < 1, f.columns[2:]] = np.nan
    # rest days
    prev = grp.game_date.shift(1)
    f["rest"] = (L.game_date - prev).dt.days.clip(upper=10)
    # recent form vs. projection (ratings lag); last 5 games, excluding current
    if "resid" in L:
        f["form5"] = grp.resid.transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        f["form5_t"] = grp.resid_t.transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    # games in last 7 days (fatigue)
    f["g7"] = grp.game_date.transform(lambda s: s.diff().dt.days.rolling(3, min_periods=1).sum()).le(7).astype(float)
    return f


def build(rated):
    """Attach features to the rated games table. Returns (df, feature_cols)."""
    g = rated
    tf = team_form(g)
    h = tf.add_suffix("_h").rename(columns={"game_id_h": "game_id", "team_h": "home_id"})
    a = tf.add_suffix("_a").rename(columns={"game_id_a": "game_id", "team_a": "away_id"})
    df = g.merge(h, on=["game_id", "home_id"], how="left").merge(a, on=["game_id", "away_id"], how="left")
    df["home"] = (~df.neutral_site).astype(float)
    df["net_h"] = df.O_h - df.D_h
    df["net_a"] = df.O_a - df.D_a
    df["net_diff"] = df.net_h - df.net_a
    df["min_gp"] = np.minimum(df.gp_h, df.gp_a)
    season_start = df.groupby("season").game_date.transform("min")
    df["day_of_season"] = (df.game_date - season_start).dt.days
    df["conf"] = df.conference_competition.astype(float)
    df["rest_diff"] = df.rest_h.fillna(7) - df.rest_a.fillna(7)
    df["d1_both"] = (df.home_d1 & df.away_d1).astype(float)
    feats = ["proj_margin", "proj_total", "proj_poss", "net_h", "net_a", "net_diff", "O_h", "D_h", "O_a", "D_a",
             "P_h", "P_a", "home", "conf", "gp_h", "gp_a", "min_gp", "day_of_season", "rest_h", "rest_a",
             "rest_diff", "d1_both"]
    for s in ("h", "a"):
        feats += [f"{c}_{s}" for c in ["efg", "d_efg", "tov_r", "d_tov_r", "orb_r", "drb_r", "ftr", "d_ftr",
                                        "three_rate", "d_three_rate", "ft_pct", "form5", "form5_t"]]
    return df, feats
