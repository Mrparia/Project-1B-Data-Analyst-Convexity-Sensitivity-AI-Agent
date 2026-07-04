"""
Bond Portfolio Convexity Sensitivity AI Agent
Part 1: Data Loading + Duration/Convexity Analytics + Monte Carlo Risk Engine
"""
import pandas as pd
import numpy as np
import json

pd.set_option('display.width', 140)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
bonds = pd.read_csv('data/bond_portfolio_data.csv')
scenarios = pd.read_csv('data/monte_carlo_scenarios.csv')
curve = pd.read_csv('data/yield_curve_history.csv', parse_dates=['CurveDate'])

bonds['IssueDate'] = pd.to_datetime(bonds['IssueDate'])
bonds['MaturityDate'] = pd.to_datetime(bonds['MaturityDate'])
bonds['ValuationDate'] = pd.to_datetime(bonds['ValuationDate'])

print(f"Bonds: {bonds.shape}, Scenarios: {scenarios.shape}, Curve points: {curve.shape}")

# ---------------------------------------------------------------------------
# 2. PORTFOLIO-LEVEL DURATION / CONVEXITY ANALYTICS
# ---------------------------------------------------------------------------
total_mv = bonds['MarketValue_INR'].sum()
bonds['w'] = bonds['MarketValue_INR'] / total_mv

port_mod_dur = (bonds['w'] * bonds['ModifiedDuration']).sum()
port_mac_dur = (bonds['w'] * bonds['MacaulayDuration']).sum()
port_convexity = (bonds['w'] * bonds['Convexity']).sum()
port_eff_dur = (bonds['w'] * bonds['EffectiveDuration']).sum()
port_eff_conv = (bonds['w'] * bonds['EffectiveConvexity']).sum()
port_dv01 = (bonds['DV01_Per100Face'] * bonds['Quantity'] / 100).sum()
port_ytm = (bonds['w'] * bonds['YieldToMaturity']).sum()

print("\n=== PORTFOLIO-LEVEL RISK SUMMARY ===")
print(f"Total Market Value (INR): {total_mv:,.0f}")
print(f"Portfolio Modified Duration: {port_mod_dur:.4f} years")
print(f"Portfolio Macaulay Duration: {port_mac_dur:.4f} years")
print(f"Portfolio Effective Duration: {port_eff_dur:.4f} years")
print(f"Portfolio Convexity: {port_convexity:.4f}")
print(f"Portfolio Effective Convexity: {port_eff_conv:.4f}")
print(f"Portfolio DV01 (INR per 1bp): {port_dv01:,.2f}")
print(f"Portfolio Yield to Maturity (wtd): {port_ytm*100:.3f}%")

# Duration/convexity price approximation for a parallel shift (in decimal, e.g. 0.01 = 100bps)
def price_change_pct(delta_y, mod_dur, convexity):
    """Second-order (duration + convexity) Taylor approximation of % price change."""
    return -mod_dur * delta_y + 0.5 * convexity * delta_y ** 2

for bp_shift in [-200, -100, -50, 50, 100, 200]:
    dy = bp_shift / 10000
    pct = price_change_pct(dy, port_mod_dur, port_convexity)
    print(f"  Parallel shift {bp_shift:+d}bps -> Est. portfolio price change: {pct*100:+.3f}%  "
          f"(~INR {pct*total_mv:+,.0f})")

# ---------------------------------------------------------------------------
# 3. KEY RATE / BUCKET EXPOSURE (partial duration by maturity bucket)
# ---------------------------------------------------------------------------
bucket_order = ['0-1Y', '1-2Y', '2-3Y', '3-5Y', '5-7Y', '7-10Y', '10-15Y', '15-20Y', '20Y+']
bucket_exposure = bonds.groupby('KeyRateBucket').apply(
    lambda g: pd.Series({
        'MarketValue_INR': g['MarketValue_INR'].sum(),
        'Weight_%': 100 * g['MarketValue_INR'].sum() / total_mv,
        'Contribution_to_PortDuration': (g['w'] * g['ModifiedDuration']).sum(),
        'Contribution_to_PortConvexity': (g['w'] * g['Convexity']).sum(),
    }), include_groups=False
).reindex(bucket_order)
print("\n=== KEY-RATE BUCKET EXPOSURE ===")
print(bucket_exposure.round(3))
bucket_exposure.to_csv('outputs/bucket_exposure.csv')

# Sector / rating risk concentration
sector_risk = bonds.groupby('Sector').apply(
    lambda g: pd.Series({
        'MarketValue_INR': g['MarketValue_INR'].sum(),
        'Weight_%': 100 * g['MarketValue_INR'].sum() / total_mv,
        'AvgModDuration': np.average(g['ModifiedDuration'], weights=g['MarketValue_INR']),
        'AvgConvexity': np.average(g['Convexity'], weights=g['MarketValue_INR']),
        'AvgOAS_bps': np.average(g['OAS_bps'], weights=g['MarketValue_INR']),
    }), include_groups=False
).sort_values('Weight_%', ascending=False)
print("\n=== SECTOR RISK CONCENTRATION ===")
print(sector_risk.round(3))
sector_risk.to_csv('outputs/sector_risk.csv')

rating_risk = bonds.groupby('CreditRating').apply(
    lambda g: pd.Series({
        'MarketValue_INR': g['MarketValue_INR'].sum(),
        'Weight_%': 100 * g['MarketValue_INR'].sum() / total_mv,
        'AvgModDuration': np.average(g['ModifiedDuration'], weights=g['MarketValue_INR']),
    }), include_groups=False
).sort_values('Weight_%', ascending=False)
print("\n=== CREDIT RATING CONCENTRATION ===")
print(rating_risk.round(3))

# ---------------------------------------------------------------------------
# 4. MONTE CARLO SCENARIO RISK ENGINE (VaR / CVaR / stress)
# ---------------------------------------------------------------------------
pnl = scenarios['PnL_Total_INR'].sort_values().reset_index(drop=True)
pnl_pct = scenarios['PnL_Pct'].sort_values().reset_index(drop=True)

def var_cvar(series, conf):
    idx = int((1 - conf) * len(series))
    var = -series.iloc[idx]
    cvar = -series.iloc[:idx+1].mean()
    return var, cvar

print("\n=== MONTE CARLO VaR / CVaR (1000 scenarios) ===")
risk_table = {}
for conf in [0.90, 0.95, 0.99]:
    var_inr, cvar_inr = var_cvar(pnl, conf)
    var_pct, cvar_pct = var_cvar(pnl_pct, conf)
    risk_table[f'{int(conf*100)}%'] = {
        'VaR_INR': var_inr, 'CVaR_INR': cvar_inr,
        'VaR_%': var_pct, 'CVaR_%': cvar_pct
    }
    print(f"  {int(conf*100)}% confidence -> VaR: INR {var_inr:,.0f} ({var_pct:.3f}%), "
          f"CVaR (Expected Shortfall): INR {cvar_inr:,.0f} ({cvar_pct:.3f}%)")

pd.DataFrame(risk_table).T.to_csv('outputs/var_cvar_table.csv')

# Decompose duration effect vs convexity effect contribution to total variance
dur_effect_share = scenarios['PnL_DurationEffect_INR'].abs().sum() / (
    scenarios['PnL_DurationEffect_INR'].abs().sum() + scenarios['PnL_ConvexityEffect_INR'].abs().sum())
print(f"\nDuration effect explains {dur_effect_share*100:.1f}% of |P&L| magnitude; "
      f"Convexity effect explains {(1-dur_effect_share)*100:.1f}%")

# Worst 10 scenarios (stress cases)
worst10 = scenarios.nsmallest(10, 'PnL_Total_INR')[
    ['ScenarioID', 'ParallelShift_bps', 'TwistFactor_bps', 'ButterflyFactor_bps', 'PnL_Total_INR', 'PnL_Pct']]
print("\n=== TOP 10 WORST SCENARIOS ===")
print(worst10.to_string(index=False))
worst10.to_csv('outputs/worst_10_scenarios.csv', index=False)

# Sanity-check: analytical duration/convexity approx vs Monte Carlo realized P&L for parallel-only scenarios
scenarios['approx_pct'] = price_change_pct(scenarios['ParallelShift_bps']/10000, port_mod_dur, port_convexity) * 100
corr = scenarios[['approx_pct', 'PnL_Pct']].corr().iloc[0, 1]
print(f"\nCorrelation between analytical (duration+convexity) approx and simulated PnL_Pct: {corr:.4f}")

summary = {
    'total_market_value_inr': float(total_mv),
    'portfolio_modified_duration': float(port_mod_dur),
    'portfolio_convexity': float(port_convexity),
    'portfolio_effective_duration': float(port_eff_dur),
    'portfolio_dv01_inr': float(port_dv01),
    'portfolio_ytm_pct': float(port_ytm * 100),
    'var_cvar': risk_table,
    'duration_effect_share_pct': float(dur_effect_share * 100),
    'analytical_vs_mc_correlation': float(corr),
}
with open('outputs/portfolio_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSaved: outputs/bucket_exposure.csv, sector_risk.csv, var_cvar_table.csv, "
      "worst_10_scenarios.csv, portfolio_summary.json")
