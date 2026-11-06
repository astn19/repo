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


def main():
    from .segments import load as load_games
    d = load_games()
    cal = fit(d)
    PATH.write_text(json.dumps(cal, indent=2))
    print(f"Fitted on {cal['n']:,} games with opening totals -> {PATH.relative_to(C.ROOT)}")
    print(f"  coef (edge, line-145) = {np.round(cal['coef'], 4)}, intercept = {cal['intercept']:.3f}")


if __name__ == "__main__":
    main()
