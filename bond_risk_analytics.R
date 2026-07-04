# =============================================================================
# Bond Portfolio Convexity Sensitivity AI Agent -- R Companion
# Duration/Convexity Analytics, Key-Rate Exposure, and Monte Carlo VaR/CVaR
# Mirrors the Python engine (01_analytics_engine.py) for cross-platform parity.
# =============================================================================
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

bonds     <- read_csv("data/bond_portfolio_data.csv", show_col_types = FALSE)
scenarios <- read_csv("data/monte_carlo_scenarios.csv", show_col_types = FALSE)
curve     <- read_csv("data/yield_curve_history.csv", show_col_types = FALSE)

# ---- Portfolio-level duration / convexity -----------------------------------
total_mv <- sum(bonds$MarketValue_INR)
bonds <- bonds %>% mutate(w = MarketValue_INR / total_mv)

port_mod_dur   <- sum(bonds$w * bonds$ModifiedDuration)
port_mac_dur   <- sum(bonds$w * bonds$MacaulayDuration)
port_convexity <- sum(bonds$w * bonds$Convexity)
port_dv01      <- sum(bonds$DV01_Per100Face * bonds$Quantity / 100)
port_ytm       <- sum(bonds$w * bonds$YieldToMaturity)

cat("=== PORTFOLIO-LEVEL RISK SUMMARY (R) ===\n")
cat(sprintf("Total Market Value (INR): %s\n", format(total_mv, big.mark = ",")))
cat(sprintf("Portfolio Modified Duration: %.4f years\n", port_mod_dur))
cat(sprintf("Portfolio Macaulay Duration: %.4f years\n", port_mac_dur))
cat(sprintf("Portfolio Convexity: %.4f\n", port_convexity))
cat(sprintf("Portfolio DV01 (INR/bp): %s\n", format(round(port_dv01, 2), big.mark = ",")))
cat(sprintf("Portfolio YTM (weighted): %.3f%%\n\n", port_ytm * 100))

# Second-order (duration + convexity) Taylor approximation of price change
price_change_pct <- function(delta_y, mod_dur, convexity) {
  -mod_dur * delta_y + 0.5 * convexity * delta_y^2
}

shocks <- c(-200, -100, -50, 50, 100, 200)
cat("Parallel-shift price sensitivity:\n")
for (bp in shocks) {
  dy <- bp / 10000
  pct <- price_change_pct(dy, port_mod_dur, port_convexity)
  cat(sprintf("  %+dbps -> %+.3f%% (~INR %+.0f)\n", bp, pct * 100, pct * total_mv))
}

# ---- Key-rate bucket exposure -------------------------------------------------
bucket_exposure <- bonds %>%
  group_by(KeyRateBucket) %>%
  summarise(
    MarketValue_INR = sum(MarketValue_INR),
    Weight_pct = 100 * sum(MarketValue_INR) / total_mv,
    Contribution_to_PortDuration = sum(w * ModifiedDuration),
    Contribution_to_PortConvexity = sum(w * Convexity),
    .groups = "drop"
  )
write_csv(bucket_exposure, "outputs/r_bucket_exposure.csv")
cat("\n=== KEY-RATE BUCKET EXPOSURE ===\n"); print(bucket_exposure)

# ---- Sector / rating concentration -------------------------------------------
sector_risk <- bonds %>%
  group_by(Sector) %>%
  summarise(
    MarketValue_INR = sum(MarketValue_INR),
    Weight_pct = 100 * sum(MarketValue_INR) / total_mv,
    AvgModDuration = weighted.mean(ModifiedDuration, MarketValue_INR),
    AvgConvexity = weighted.mean(Convexity, MarketValue_INR),
    .groups = "drop"
  ) %>% arrange(desc(Weight_pct))
write_csv(sector_risk, "outputs/r_sector_risk.csv")
cat("\n=== SECTOR CONCENTRATION ===\n"); print(sector_risk)

# ---- Monte Carlo VaR / CVaR ----------------------------------------------------
var_cvar <- function(pnl, conf) {
  q <- quantile(pnl, probs = 1 - conf, type = 1)
  cvar <- mean(pnl[pnl <= q])
  c(VaR = -as.numeric(q), CVaR = -cvar)
}

cat("\n=== MONTE CARLO VaR / CVaR ===\n")
risk_tbl <- lapply(c(0.90, 0.95, 0.99), function(cf) {
  r <- var_cvar(scenarios$PnL_Total_INR, cf)
  cat(sprintf("  %.0f%% -> VaR: INR %s | CVaR: INR %s\n", cf * 100,
              format(round(r["VaR"]), big.mark = ","), format(round(r["CVaR"]), big.mark = ",")))
  data.frame(Confidence = cf, VaR_INR = r["VaR"], CVaR_INR = r["CVaR"])
}) %>% bind_rows()
write_csv(risk_tbl, "outputs/r_var_cvar_table.csv")

# ---- Random Forest: bond-level duration/convexity prediction (R parity) -------
if (requireNamespace("randomForest", quietly = TRUE)) {
  library(randomForest)
  set.seed(42)
  model_df <- bonds %>%
    select(ModifiedDuration, Convexity, CouponRate, YearsToMaturity, YieldToMaturity,
           OAS_bps, Sector, CreditRating) %>%
    mutate(Sector = as.factor(Sector), CreditRating = as.factor(CreditRating))

  idx <- sample(seq_len(nrow(model_df)), size = 0.8 * nrow(model_df))
  train <- model_df[idx, ]; test <- model_df[-idx, ]

  rf_dur  <- randomForest(ModifiedDuration ~ CouponRate + YearsToMaturity + YieldToMaturity +
                             OAS_bps + Sector + CreditRating, data = train, ntree = 400)
  rf_conv <- randomForest(Convexity ~ CouponRate + YearsToMaturity + YieldToMaturity +
                             OAS_bps + Sector + CreditRating, data = train, ntree = 400)

  pred_dur  <- predict(rf_dur, test)
  pred_conv <- predict(rf_conv, test)
  r2 <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)

  cat(sprintf("\n[R RandomForest] Duration R2:  %.4f\n", r2(test$ModifiedDuration, pred_dur)))
  cat(sprintf("[R RandomForest] Convexity R2: %.4f\n", r2(test$Convexity, pred_conv)))
} else {
  cat("\n[Note] Install 'randomForest' package to run the R ML parity model:\n")
  cat("  install.packages('randomForest')\n")
}

cat("\nR analytics complete. Outputs written to outputs/r_*.csv\n")
