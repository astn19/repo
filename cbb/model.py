"""ML layer: LightGBM corrects the ratings projection, trained walk-forward by season.

For test season S we train only on seasons < S, so every prediction is out-of-sample.
Targets are residuals (actual - ratings projection), which is a more stable target.

Usage: python -m cbb.model
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import build

FIRST_TRAIN = 2011   # 2010 is a warm-up season for priors
FIRST_TEST = 2014

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def _fit(X, y):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X, y)
    return m


def walk_forward(df, feats, test_seasons):
    preds = []
    for s in test_seasons:
        tr = df[(df.season >= FIRST_TRAIN) & (df.season < s)]
        te = df[df.season == s]
        if te.empty:
            continue
        mm = _fit(tr[feats], tr.margin - tr.proj_margin)
        mt = _fit(tr[feats], tr.total - tr.proj_total)
        out = te[["game_id"]].copy()
        out["ml_margin"] = te.proj_margin + mm.predict(te[feats])
        out["ml_total"] = te.proj_total + mt.predict(te[feats])
        # residual std from training set (for cover probabilities)
        out["sd_margin"] = np.std(tr.margin - (tr.proj_margin + mm.predict(tr[feats])))
        out["sd_total"] = np.std(tr.total - (tr.proj_total + mt.predict(tr[feats])))
        preds.append(out)
        print(f"  walk-forward {s}: trained on {len(tr):,} games")
    return pd.concat(preds)


def fit_final(df, feats):
    """Models trained on all completed seasons, for live predictions."""
    tr = df[(df.season >= FIRST_TRAIN) & (df.season < C.LAST_SEASON)]
    mm = _fit(tr[feats], tr.margin - tr.proj_margin)
    mt = _fit(tr[feats], tr.total - tr.proj_total)
    sd_m = np.std(tr.margin - (tr.proj_margin + mm.predict(tr[feats])))
    sd_t = np.std(tr.total - (tr.proj_total + mt.predict(tr[feats])))
    return mm, mt, sd_m, sd_t


def accuracy_report(df):
    r = df[(df.d1_both == 1) & df.ml_margin.notna()].copy()
    def f(d):
        return pd.Series({
            "games": len(d),
            "base_mae_margin": (d.margin - d.proj_margin).abs().mean(),
            "ml_mae_margin": (d.margin - d.ml_margin).abs().mean(),
            "base_mae_total": (d.total - d.proj_total).abs().mean(),
            "ml_mae_total": (d.total - d.ml_total).abs().mean(),
            "ml_bias_margin": (d.margin - d.ml_margin).mean(),
            "ml_bias_total": (d.total - d.ml_total).mean(),
        })
    return r.groupby("season").apply(f), f(r)


def main():
    rated = pd.read_parquet(C.PROCESSED / "games_rated.parquet")
    df, feats = build(rated)
    seasons = list(range(FIRST_TEST, C.LAST_SEASON + 1))
    preds = walk_forward(df, feats, seasons)
    df = df.merge(preds, on="game_id", how="left")
    df.to_parquet(C.PROCESSED / "predictions.parquet")
    by_season, overall = accuracy_report(df[df.season < C.LAST_SEASON])
    print("\nOut-of-sample accuracy, D1 vs D1 (points):")
    print(by_season.round(2).to_string())
    print("\nOverall:")
    print(overall.round(3).to_string())
    C.REPORTS.mkdir(exist_ok=True)
    by_season.round(3).to_csv(C.REPORTS / "accuracy_by_season.csv")


if __name__ == "__main__":
    main()
