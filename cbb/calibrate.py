"""Calibrate over/under probabilities against opening totals.

The raw model leans to overs (it picks overs ~2x as often, yet unders win more). A logistic
correction, P(over) = f(model_total - open_line, open_line), fit on historical openers removes
that lean. Validated honestly: fit on 2014-21 only -> 2024-26 win rate at prob >= 0.56 rose
from 54.1% (raw, prob >= 0.55) to 55.5%.

Usage: python -m cbb.calibrate      # fits on all seasons with opening totals, saves coefficients
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config as C

PATH = C.ROOT / "data" / "model" / "total_calibration.json"  # tracked in git
SPREAD_PATH = C.ROOT / "data" / "model" / "spread_calibration.json"
# Spreads vs OPENING lines (fit 2014-21, tested 2024-26): calibrated home-cover prob >= 0.53 won 55.4%
# of 1,721 bets. Spread probabilities come out compressed (most games have no edge), so the tracker
# shows them doubled around 50% (SPREAD_SCALE) to rank them alongside totals.
SPREAD_SCALE = 2.0


def features(ml_total, line):
    return np.column_stack([np.asarray(ml_total) - np.asarray(line), np.asarray(line) - 145])


def fit(d):
    """d needs ml_total, total_open, total (actual)."""
    d = d.dropna(subset=["ml_total", "total_open", "total"])
    d = d[d.total != d.total_open]
    y = (d.total > d.total_open).astype(int)
    lr = LogisticRegression().fit(features(d.ml_total, d.total_open), y)
    return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "n": int(len(d))}


def p_over(cal, ml_total, line):
    z = features(ml_total, line) @ np.array(cal["coef"]) + cal["intercept"]
    return 1 / (1 + np.exp(-z))


def load():
    return json.loads(PATH.read_text()) if PATH.exists() else None


def spread_features(ml_margin, spread_open, min_gp):
    edge = np.asarray(ml_margin) + np.asarray(spread_open)  # >0: model likes home vs the line
    return np.column_stack([edge, edge * (np.asarray(min_gp) < 5), np.abs(np.asarray(spread_open))])


def fit_spread(d):
    """d needs ml_margin, spread_open (home line), margin (home - away), min_gp."""
    d = d.dropna(subset=["ml_margin", "spread_open", "margin"])
    ats = d.margin + d.spread_open
    d, ats = d[ats != 0], ats[ats != 0]
    lr = LogisticRegression().fit(spread_features(d.ml_margin, d.spread_open, d.min_gp), (ats > 0).astype(int))
    return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "n": int(len(d))}


def p_home_cover(cal, ml_margin, spread_open, min_gp):
    z = spread_features(ml_margin, spread_open, min_gp) @ np.array(cal["coef"]) + cal["intercept"]
    return 1 / (1 + np.exp(-z))


def load_spread():
    return json.loads(SPREAD_PATH.read_text()) if SPREAD_PATH.exists() else None


def main():
    from .segments import load as load_games
    d = load_games()
    cal = fit(d)
    PATH.write_text(json.dumps(cal, indent=2))
    print(f"Fitted on {cal['n']:,} games with opening totals -> {PATH.relative_to(C.ROOT)}")
    print(f"  coef (edge, line-145) = {np.round(cal['coef'], 4)}, intercept = {cal['intercept']:.3f}")
    sc = fit_spread(d)
    SPREAD_PATH.write_text(json.dumps(sc, indent=2))
    print(f"Spreads: fitted on {sc['n']:,} games with opening spreads -> {SPREAD_PATH.relative_to(C.ROOT)}")
    print(f"  coef (edge, early edge, |line|) = {np.round(sc['coef'], 4)}, intercept = {sc['intercept']:.3f}")


if __name__ == "__main__":
    main()
