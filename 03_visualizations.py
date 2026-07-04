import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({'font.size': 10, 'figure.dpi': 140, 'font.family': 'DejaVu Sans'})
COLORS = {'primary': '#1f4e5f', 'accent': '#e07a5f', 'good': '#3d8361', 'bad': '#c1121f', 'grid': '#dddddd'}

bonds = pd.read_csv('data/bond_portfolio_data.csv')
scenarios = pd.read_csv('outputs/scenarios_with_ml_predictions.csv')
curve = pd.read_csv('data/yield_curve_history.csv', parse_dates=['CurveDate'])
bucket_exp = pd.read_csv('outputs/bucket_exposure.csv', index_col=0)
sector_risk = pd.read_csv('outputs/sector_risk.csv', index_col=0)
model_a = pd.read_csv('outputs/model_a_comparison.csv', index_col=0)
model_b = pd.read_csv('outputs/model_b_comparison.csv', index_col=0)
fi = pd.read_csv('outputs/model_a_feature_importance.csv', index_col=0).iloc[:, 0]

# 1. Yield curve evolution over time -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
dates = sorted(curve['CurveDate'].unique())
sample_dates = [dates[0], dates[len(dates)//3], dates[2*len(dates)//3], dates[-1]]
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sample_dates)))
for d, c in zip(sample_dates, cmap):
    sub = curve[curve['CurveDate'] == d].sort_values('Tenor_Years')
    ax.plot(sub['Tenor_Years'], sub['Yield']*100, marker='o', ms=3, color=c,
            label=pd.Timestamp(d).strftime('%b %Y'))
ax.set_xlabel('Tenor (Years)'); ax.set_ylabel('Yield (%)')
ax.set_title('INR Sovereign Yield Curve Evolution', fontweight='bold')
ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/01_yield_curve_evolution.png'); plt.close()

# 2. Duration vs Convexity scatter (bubble = market value, color = sector) -----------
fig, ax = plt.subplots(figsize=(8, 5.5))
sectors = bonds['Sector'].unique()
palette = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
for s, c in zip(sectors, palette):
    sub = bonds[bonds['Sector'] == s]
    ax.scatter(sub['ModifiedDuration'], sub['Convexity'], s=sub['MarketValue_INR']/8000,
               alpha=0.6, color=c, label=s, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Modified Duration (years)'); ax.set_ylabel('Convexity')
ax.set_title('Bond Universe: Duration vs Convexity (bubble = market value)', fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/02_duration_convexity_scatter.png'); plt.close()

# 3. Key-rate bucket exposure (duration contribution) --------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
be = bucket_exp.dropna()
ax.bar(be.index, be['Contribution_to_PortDuration'], color=COLORS['primary'])
ax.set_ylabel('Contribution to Portfolio Modified Duration (yrs)')
ax.set_title('Key-Rate Bucket Duration Contribution', fontweight='bold')
ax.tick_params(axis='x', rotation=40)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('figures/03_bucket_duration_contribution.png'); plt.close()

# 4. Monte Carlo P&L distribution with VaR/CVaR lines --------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(scenarios['PnL_Total_INR']/1e6, bins=40, color=COLORS['primary'], alpha=0.8)
var95 = -np.percentile(scenarios['PnL_Total_INR'], 5)
ax.axvline(-var95/1e6, color=COLORS['bad'], linestyle='--', linewidth=1.5, label=f'95% VaR: INR {var95/1e6:.2f}M')
ax.set_xlabel('Portfolio P&L (INR millions)'); ax.set_ylabel('Frequency')
ax.set_title('Monte Carlo Simulated Portfolio P&L Distribution (1,000 scenarios)', fontweight='bold')
ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/04_montecarlo_pnl_distribution.png'); plt.close()

# 5. Duration effect vs Convexity effect scatter --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(scenarios['ParallelShift_bps'], scenarios['PnL_ConvexityEffect_INR']/1000,
                 c=scenarios['PnL_Total_INR']/1e6, cmap='RdYlGn', s=14, alpha=0.8)
cb = plt.colorbar(sc); cb.set_label('Total P&L (INR mm)')
ax.set_xlabel('Parallel Shift (bps)'); ax.set_ylabel('Convexity Effect (INR thousands)')
ax.set_title('Convexity Effect Grows Non-linearly with Rate Shift Size', fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/05_convexity_effect_nonlinearity.png'); plt.close()

# 6. Model comparison - Model A (duration/convexity prediction) ----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
model_a[['R2_Duration', 'R2_Convexity']].plot(kind='bar', ax=axes[0],
    color=[COLORS['primary'], COLORS['accent']])
axes[0].set_title('Model A: R² by Algorithm', fontweight='bold'); axes[0].set_ylabel('R²')
axes[0].legend(['Duration', 'Convexity'], frameon=False); axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(alpha=0.3, axis='y')
model_b['R2'].plot(kind='bar', ax=axes[1], color=COLORS['good'])
axes[1].set_title('Model B: Portfolio P&L R² by Algorithm', fontweight='bold'); axes[1].set_ylabel('R²')
axes[1].tick_params(axis='x', rotation=0); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('figures/06_model_comparison.png'); plt.close()

# 7. Feature importance ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
top_fi = fi.sort_values(ascending=True).tail(8)
ax.barh(top_fi.index, top_fi.values, color=COLORS['primary'])
ax.set_xlabel('Importance'); ax.set_title('Random Forest Feature Importance — Duration Prediction', fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig('figures/07_feature_importance.png'); plt.close()

# 8. Sector risk concentration (pie + duration) ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].pie(sector_risk['MarketValue_INR'], labels=sector_risk.index, autopct='%1.0f%%',
            colors=plt.cm.Set2(np.linspace(0, 1, len(sector_risk))))
axes[0].set_title('Portfolio Weight by Sector', fontweight='bold')
axes[1].bar(sector_risk.index, sector_risk['AvgModDuration'], color=COLORS['accent'])
axes[1].set_ylabel('Avg Modified Duration (yrs)'); axes[1].set_title('Avg Duration by Sector', fontweight='bold')
axes[1].tick_params(axis='x', rotation=40); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('figures/08_sector_concentration.png'); plt.close()

print("Saved 8 figures to figures/")
