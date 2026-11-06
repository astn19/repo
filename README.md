# CBB Model: college basketball spread and total predictions

A walk-forward model for NCAA men's basketball. It projects margins and totals, and it can be
backtested against historical lines to see whether it beats the **52.38% break-even** at -110.

```
ESPN box scores (2010→today)
   └─ cbb.data      → one row per game, possessions, neutral/OT flags
   └─ cbb.ratings   → adjusted offense/defense/tempo ratings, refit before EVERY game date
                       using only earlier games (ridge regression, preseason priors)
   └─ cbb.features  → season-to-date four factors, rest, recent form vs. projection
   └─ cbb.model     → LightGBM correction layer, trained season-by-season (walk-forward)
   └─ cbb.backtest  → grades predictions vs. YOUR historical lines: win %, ROI, CLV, edge buckets
   └─ cbb.predict   → projections + picks for upcoming games
```

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                     # download, rate, train, project the next 2 days
```

Or step by step:

```bash
python -m cbb.data               # downloads ~36 parquet files (≈25 MB) to data/raw
python -m cbb.ratings            # ~1 min
python -m cbb.model              # ~1 min, prints out-of-sample accuracy
python -m cbb.backtest data/lines/your_lines.csv --min-prob 0.55
python -m cbb.predict --days 1 --lines data/lines/today.csv
```

## Historical lines (required for a betting backtest)

The free game data does **not** include betting lines, so you need to supply them. Any source works if you
can export a CSV like `data/lines/TEMPLATE_lines.csv`:

| column | meaning |
|---|---|
| `date` | YYYY-MM-DD |
| `home`, `away` | team names in any common spelling (fuzzy-matched to ESPN) |
| `spread` | **home** spread (closing), e.g. `-6.5` = home favored by 6.5 |
| `total` | closing over/under |
| `spread_open`, `total_open` | optional: enables closing-line-value (CLV) analysis |
| `home_price`, `away_price`, `over_price`, `under_price` | optional American odds (default -110) |

Unmatched team names get printed. Add them to `data/lines/team_aliases.csv` as `alias,team_id`.

Common sources: SportsbookReviewsOnline's NCAAB archive (free xlsx, older seasons), The Odds API
(paid historical, 2020+), Kaggle CBB odds datasets, or your own line logs going forward.

## Current results (out-of-sample, D1 vs D1, 2014–2026, 70,850 games)

| | ratings only | + ML layer |
|---|---|---|
| Margin MAE | 9.00 pts | **8.94 pts** |
| Total MAE | 13.49 pts | **13.30 pts** |

For context, closing college lines are generally believed to land around 8–8.7 MAE on margins. The
model is **not yet sharper than the closing market overall**, which is normal at this stage. The
places to look for an edge are where the market is weakest: **early season, low-major games,
totals, and opening lines**. `cbb.backtest` breaks results down by month, season, conference and
non-conference games, and edge size so you can find them.

Tuning was done only on 2011–2016. Treat 2017+ as the clean test period.

## Backtest vs real lines (SportsbookReviewsOnline, 2014–2021, ~34,000 games)

Convert the archive with `python -m cbb.sbr <xlsx files> -o data/lines/sbr_2014_2021.csv`. Team
names are learned automatically by matching final scores (`cbb.sbr.learn_aliases`).

| Market | Bets (prob ≥ 0.55) | Win % | Note |
|---|---|---|---|
| Spreads vs close | 18,767 | 50.8% | **no edge** |
| Totals vs close | 16,431 | 54.0% | edge, shrinking in recent seasons |
| Totals vs **open** | 15,283 | **56.8%** | above 55% in every season, 2014–2021 |

Caveats: opening lines carry low limits, SBR data has errors (a sign-convention bug that created a
fake spread edge was found and fixed), and the most recent season in this data is 2021. The
totals edge needs to be confirmed on 2022+ lines before you bet real money.

## How to read a backtest honestly

* Look at the **p-value vs 52.4%**. Fewer than ~1,000 bets at 54% is still within noise.
* **CLV beats win %.** If your bets consistently get better numbers than the close, the edge is
  probably real even before the wins show up.
* Don't cherry-pick the best threshold or segment after looking. Choose rules on older seasons,
  then confirm on newer ones.
* The pipeline has been checked with a null test (lines = the model's own number). It goes 50.5%
  and loses the juice, as it should.

## Next improvements (roughly by expected value)

1. Historical lines, then backtest with an opening vs. closing split and CLV.
2. Use the market line as a feature, and model *when* the model and the market disagree productively.
3. Injury and availability data (player box scores are already in the same data source).
4. Roster continuity and transfer-portal adjustments to preseason priors (the biggest November error).
5. Per-team or per-venue home-court advantage; travel distance and altitude.
6. Live line feed (e.g. The Odds API) plus automated daily picks and bet logging.

## Data

ESPN data via the [sportsdataverse / hoopR](https://github.com/sportsdataverse/hoopR-mbb-data) releases.
