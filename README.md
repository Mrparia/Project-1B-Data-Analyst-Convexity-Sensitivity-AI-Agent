# Project 1B — Bond Portfolio Convexity Sensitivity AI Agent

Built on the uploaded 300-bond portfolio, 1,000-scenario Monte Carlo set, and 2022–2024 INR yield curve history.

## Key portfolio findings
- Total market value: ₹32,059,346 | Portfolio Modified Duration: **4.20 yrs** | Convexity: **44.06** | DV01: ₹8,948/bp
- 95% Monte Carlo VaR: **₹1.98M** (6.17%) · 95% CVaR (expected shortfall): **₹2.38M** (7.39%)
- Duration effect explains 94.1% of simulated P&L magnitude; convexity explains the remaining 5.9% — material mainly in large (>150bps) shocks
- Government bonds are 57.6% of the book and dominate both duration and convexity exposure (20Y+ SOV paper carries the highest convexity)
- The analytical duration+convexity Taylor approximation correlates **1.0000** with simulated P&L for pure parallel shifts — validating the closed-form model

## ML model results
**Model A — bond-level Modified Duration & Convexity prediction** (from coupon, maturity, YTM, spread, sector, rating):
| Model | Duration R² | Convexity R² |
|---|---|---|
| Random Forest (best) | 0.930 | 0.838 |
| XGBoost | 0.915 | 0.811 |
| Neural Net (MLP) | 0.916 | 0.806 |

YearsToMaturity dominates feature importance (98.3%), consistent with duration theory.

**Model B — portfolio P&L prediction from scenario factors** (parallel/twist/butterfly):
| Model | R² | RMSE (INR) |
|---|---|---|
| Random Forest (best) | 0.990 | 137,052 |
| XGBoost | 0.990 | 141,779 |
| Neural Net (MLP) | -0.02 (underfit — needs more tuning/data) | 1,406,774 |

ML-implied 95% VaR (₹1.978M) matches the historical simulated 95% VaR (₹1.976M) almost exactly.

## Files in this delivery

| File | What it is |
|---|---|
| `python_engine/01_analytics_engine.py` | Duration/convexity portfolio analytics, key-rate bucket & sector exposure, Monte Carlo VaR/CVaR engine |
| `python_engine/02_ml_models.py` | Random Forest / XGBoost / Neural Network models for bond-level sensitivity and portfolio P&L, with saved models |
| `python_engine/03_visualizations.py` | Generates all 8 charts in `figures/` |
| `figures/*.png` | Yield curve evolution, duration-convexity scatter, bucket exposure, Monte Carlo P&L distribution, model comparisons, feature importance, sector concentration |
| `analytics_outputs/*.csv, *.json` | All computed tables (bucket exposure, sector risk, VaR/CVaR, model comparisons, worst-case scenarios) |
| `bond_risk_analytics.R` | R companion script mirroring the Python duration/convexity/VaR engine (R not installed in this sandbox — run locally in RStudio; requires `dplyr`, `readr`, `ggplot2`, optionally `randomForest`) |
| `Bond_Risk_Model.xlsx` | Formula-driven Excel workbook: live KPIs, bucket/sector SUMIFS breakdowns, VaR/CVaR via PERCENTILE/AVERAGEIF, shift-sensitivity chart, and a `PowerBI_DAX_Measures` tab with ready-to-paste DAX code + a suggested 5-page Power BI report layout |
| `Bond_Risk_Lab.html` | **Gamified Bond Risk Lab** — an interactive training simulator (open in any browser). Traders drag Parallel/Twist/Butterfly shock sliders and watch a live price–yield curve compare the duration-only tangent line against the true convex price. A "Predict the P&L" challenge quizzes trainees against 150 real Monte Carlo scenarios, scoring accuracy and promoting them through Trainee → Associate → Senior Risk Analyst → Chief Risk Officer tiers. |

## Notes & caveats
- The Neural Network in Model B underperformed on this dataset (small sample, need more architecture tuning or more scenarios) — Random Forest is recommended for production use on both tasks.
- `Bond_Risk_Model.xlsx` was recalculated and verified with **zero formula errors** across 1,001 formulas.
- The Power BI file itself (`.pbix`) can't be authored outside Power BI Desktop — the `PowerBI_DAX_Measures` tab gives copy-paste-ready DAX measures and report-page guidance so you can wire up the same model in Power BI Desktop directly from `Bond_Data`/`Scenario_Data`.
