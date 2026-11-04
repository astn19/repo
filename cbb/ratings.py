"""Walk-forward adjusted efficiency & tempo ratings (KenPom-style, ridge regression).

For every game date we fit ratings using ONLY games played before that date,
shrunk toward a preseason prior (last season's ratings, regressed to the mean).
Each game then gets the *pregame* ratings of both teams -> no look-ahead.

Efficiency model (points per 100 possessions for the team on offense):
    eff = mu + O[offense] + D[defense] + hca * (+1 home offense / -1 away offense / 0 neutral)
Tempo model (possessions per 40 min):
    poss = mu_p + P[home] + P[away]

Usage: python -m cbb.ratings
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C

OTHER = -1  # pooled non-D1 opponent id


@dataclass
class RatingParams:
    prior_games: float = 3.0       # strength of preseason prior, in "games" of data
    prior_regress: float = 0.8     # keep this fraction of last season's rating
    new_team_prior: float = -6.0   # net rating prior for teams with no history (O and D split)
    half_life_days: float = 200.0   # recency weighting within season
    pace_prior_games: float = 6.0
    pace_regress: float = 0.6
    hca_prior: float = 2.6         # pts/100 for home offense (and -2.6 away) -> ~3.6 pt margin; tuned on 2011-16
    hca_prior_games: float = 20000.0


def _team_index(g):
    ids = pd.unique(np.concatenate([g.home_key.values, g.away_key.values]))
    return {t: i for i, t in enumerate(ids)}


def _season_ratings(g, prior, p: RatingParams):
    """Run walk-forward fits through one season.

    g: games for one season (with home_key/away_key: team id or OTHER)
    prior: dict team -> (O, D, P) preseason prior offsets
    Returns (pregame DataFrame indexed like g, final ratings dict)
    """
    idx = _team_index(g)
    n = len(idx)
    # Efficiency design: cols [O_0..O_n-1, D_0..D_n-1, mu, hca]
    ng = len(g)
    hi = g.home_key.map(idx).values
    ai = g.away_key.map(idx).values
    home_flag = np.where(g.neutral_site.values, 0.0, 1.0)

    rows = np.arange(2 * ng)
    off = np.concatenate([hi, ai])
    dfn = np.concatenate([ai, hi])
    hca_col = np.concatenate([home_flag, -home_flag])
    data = np.concatenate([np.ones(2 * ng), np.ones(2 * ng), np.ones(2 * ng), hca_col])
    r_idx = np.concatenate([rows, rows, rows, rows])
    c_idx = np.concatenate([off, n + dfn, np.full(2 * ng, 2 * n), np.full(2 * ng, 2 * n + 1)])
    X = sp.csr_matrix((data, (r_idx, c_idx)), shape=(2 * ng, 2 * n + 2))
    y = np.concatenate([100 * g.pts_h.values / g.poss.values, 100 * g.pts_a.values / g.poss.values])

    # Tempo design: cols [P_0..P_n-1, mu_p]
    Xp = sp.csr_matrix((np.ones(3 * ng), (np.concatenate([np.arange(ng)] * 3),
                                          np.concatenate([hi, ai, np.full(ng, n)]))), shape=(ng, n + 1))
    yp = g.poss40.values

    # Priors
    O0 = np.zeros(n); D0 = np.zeros(n); P0 = np.zeros(n)
    inv = {i: t for t, i in idx.items()}
    for i in range(n):
        t = inv[i]
        if t in prior:
            O0[i], D0[i], P0[i] = prior[t]
        elif t == OTHER:
            O0[i], D0[i] = -8.0, 8.0  # non-D1 opponents are much worse
        else:
            O0[i], D0[i] = p.new_team_prior / 2, -p.new_team_prior / 2
    mu0 = y.mean() if ng else 103.0
    beta0 = np.concatenate([O0, D0, [mu0, p.hca_prior]])
    lam = np.concatenate([np.full(2 * n, p.prior_games), [1e-3, p.hca_prior_games]])
    beta0p = np.concatenate([P0, [yp.mean() if ng else 68.0]])
    lamp = np.concatenate([np.full(n, p.pace_prior_games), [1e-3]])
    # OTHER team gets a strong prior (few, noisy games)
    if OTHER in idx:
        j = idx[OTHER]
        lam[j] = lam[n + j] = 50.0
        lamp[j] = 50.0

    dates = g.game_date.values
    udates = np.unique(dates)
    out = np.zeros((ng, 8))  # O_h D_h O_a D_a P_h P_a mu hca  (+ mu_p stored separately)
    mup_out = np.zeros(ng)
    gp_out = np.zeros((ng, 2))
    games_played = np.zeros(n)
    decay = np.log(2) / p.half_life_days

    beta, betap = beta0.copy(), beta0p.copy()
    for d in udates:
        today = dates == d
        past = dates < d
        if past.any():
            age = (d - dates[past]).astype("timedelta64[D]").astype(float)
            w = np.exp(-decay * age)
            W2 = np.concatenate([w, w])
            mask2 = np.concatenate([past, past])
            Xs = X[mask2]
            A = (Xs.T.multiply(W2)).dot(Xs).toarray() + np.diag(lam)
            b = Xs.T.dot(W2 * y[mask2]) + lam * beta0
            beta = np.linalg.solve(A, b)
            Xps = Xp[past]
            Ap = (Xps.T.multiply(w)).dot(Xps).toarray() + np.diag(lamp)
            bp = Xps.T.dot(w * yp[past]) + lamp * beta0p
            betap = np.linalg.solve(Ap, bp)
        O, D, mu, hca = beta[:n], beta[n:2 * n], beta[2 * n], beta[2 * n + 1]
        P, mup = betap[:n], betap[n]
        th, ta = hi[today], ai[today]
        out[today] = np.column_stack([O[th], D[th], O[ta], D[ta], P[th], P[ta],
                                      np.full(len(th), mu), np.full(len(th), hca)])
        mup_out[today] = mup
        gp_out[today] = np.column_stack([games_played[th], games_played[ta]])
        np.add.at(games_played, th, 1)
        np.add.at(games_played, ta, 1)

    pre = pd.DataFrame(out, columns=["O_h", "D_h", "O_a", "D_a", "P_h", "P_a", "mu", "hca"], index=g.index)
    pre["mu_p"] = mup_out
    pre["gp_h"], pre["gp_a"] = gp_out[:, 0], gp_out[:, 1]

    # Final (end of season, all games) ratings -> next season's prior and current ratings.
    w = np.exp(-decay * (dates.max() - dates).astype("timedelta64[D]").astype(float))
    W2 = np.concatenate([w, w])
    A = (X.T.multiply(W2)).dot(X).toarray() + np.diag(lam)
    beta = np.linalg.solve(A, X.T.dot(W2 * y) + lam * beta0)
    Ap = (Xp.T.multiply(w)).dot(Xp).toarray() + np.diag(lamp)
    betap = np.linalg.solve(Ap, Xp.T.dot(w * yp) + lamp * beta0p)
    final = {inv[i]: (beta[i], beta[n + i], betap[i]) for i in range(n)}
    meta = {"mu": beta[2 * n], "hca": beta[2 * n + 1], "mu_p": betap[n], "games_played": dict(zip(inv.values(), games_played))}
    return pre, final, meta


def project(df):
    """Projected score/margin/total from pregame ratings columns."""
    home = np.where(df.neutral_site.values, 0.0, 1.0)
    poss = df.mu_p + df.P_h + df.P_a
    eff_h = df.mu + df.O_h + df.D_a + df.hca * home
    eff_a = df.mu + df.O_a + df.D_h - df.hca * home
    out = pd.DataFrame(index=df.index)
    out["proj_poss"] = poss
    out["proj_pts_h"] = eff_h * poss / 100
    out["proj_pts_a"] = eff_a * poss / 100
    out["proj_margin"] = out.proj_pts_h - out.proj_pts_a
    out["proj_total"] = out.proj_pts_h + out.proj_pts_a
    return out


def add_keys(g):
    g = g.copy()
    g["home_key"] = np.where(g.home_d1, g.home_id, OTHER)
    g["away_key"] = np.where(g.away_d1, g.away_id, OTHER)
    return g


def run(g, p: RatingParams = RatingParams(), verbose=True):
    g = add_keys(g)
    prior = {}
    pres, finals = [], {}
    for season in sorted(g.season.unique()):
        gs = g[g.season == season]
        pre, final, meta = _season_ratings(gs, prior, p)
        pres.append(pre)
        finals[season] = (final, meta)
        prior = {t: (p.prior_regress * o, p.prior_regress * d, p.pace_regress * pc)
                 for t, (o, d, pc) in final.items() if t != OTHER}
        if verbose:
            print(f"  ratings {season}: {len(gs)} games, hca={meta['hca']:.2f}/100")
    pre = pd.concat(pres)
    out = g.join(pre)
    out = out.join(project(out))
    return out, finals


def ratings_table(finals, season, names):
    final, meta = finals[season]
    rows = [{"team_id": t, "team": names.get(t, t), "adj_o": meta["mu"] + o, "adj_d": meta["mu"] + d,
             "net": o - d, "tempo": meta["mu_p"] + pc, "games": meta["games_played"].get(t, 0)}
            for t, (o, d, pc) in final.items() if t != OTHER]
    return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def team_names(g):
    n = pd.concat([g[["home_id", "home_name"]].set_axis(["id", "name"], axis=1),
                   g[["away_id", "away_name"]].set_axis(["id", "name"], axis=1)])
    return n.drop_duplicates("id", keep="last").set_index("id").name.to_dict()


def main():
    g = pd.read_parquet(C.PROCESSED / "games.parquet")
    out, finals = run(g)
    out.to_parquet(C.PROCESSED / "games_rated.parquet")
    names = team_names(g)
    pd.to_pickle(finals, C.PROCESSED / "final_ratings.pkl")
    last_full = C.LAST_SEASON - 1
    t = ratings_table(finals, last_full, names)
    print(f"\nTop 15, end of {last_full}:")
    print(t.head(15).round(1).to_string())


if __name__ == "__main__":
    main()
