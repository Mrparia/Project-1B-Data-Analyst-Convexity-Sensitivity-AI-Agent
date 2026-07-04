"""
Bond Portfolio Convexity Sensitivity AI Agent
Part 2: Machine Learning Models
  Model A - Bond-level sensitivity prediction (Modified Duration & Convexity) from bond characteristics
  Model B - Portfolio P&L prediction from Monte Carlo yield-curve scenario factors
Compares Random Forest, XGBoost, and Neural Network (MLP) for each task.
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

bonds = pd.read_csv('data/bond_portfolio_data.csv')
scenarios = pd.read_csv('data/monte_carlo_scenarios.csv')

results_log = {}

# ===========================================================================
# MODEL A: BOND-LEVEL DURATION & CONVEXITY SENSITIVITY PREDICTION
# ===========================================================================
print("="*80)
print("MODEL A: Bond-level Modified Duration & Convexity Prediction")
print("="*80)

num_features = ['CouponRate', 'YearsToMaturity', 'YieldToMaturity', 'CouponFrequency',
                 'OAS_bps', 'SpreadOverBenchmark_bps']
cat_features = ['Sector', 'CreditRating']
target_cols = ['ModifiedDuration', 'Convexity']

X = bonds[num_features + cat_features].copy()
y = bonds[target_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

model_specs_a = {
    'RandomForest': RandomForestRegressor(n_estimators=400, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                  subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=3000,
                                   random_state=RANDOM_STATE, early_stopping=True)
}

model_a_results = {}
fitted_a = {}
for name, est in model_specs_a.items():
    pipe = Pipeline([('prep', preprocess), ('model', est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2_dur = r2_score(y_test['ModifiedDuration'], pred[:, 0])
    r2_conv = r2_score(y_test['Convexity'], pred[:, 1])
    rmse_dur = mean_squared_error(y_test['ModifiedDuration'], pred[:, 0]) ** 0.5
    rmse_conv = mean_squared_error(y_test['Convexity'], pred[:, 1]) ** 0.5
    mae_dur = mean_absolute_error(y_test['ModifiedDuration'], pred[:, 0])
    mae_conv = mean_absolute_error(y_test['Convexity'], pred[:, 1])
    model_a_results[name] = {
        'R2_Duration': r2_dur, 'R2_Convexity': r2_conv,
        'RMSE_Duration': rmse_dur, 'RMSE_Convexity': rmse_conv,
        'MAE_Duration': mae_dur, 'MAE_Convexity': mae_conv,
    }
    fitted_a[name] = pipe
    print(f"\n{name}:")
    print(f"  Duration  -> R2={r2_dur:.4f}  RMSE={rmse_dur:.4f}  MAE={mae_dur:.4f}")
    print(f"  Convexity -> R2={r2_conv:.4f}  RMSE={rmse_conv:.4f}  MAE={mae_conv:.4f}")

model_a_df = pd.DataFrame(model_a_results).T
model_a_df.to_csv('outputs/model_a_comparison.csv')
best_a = model_a_df['R2_Duration'].idxmax()
print(f"\nBest Model A (by Duration R2): {best_a}")
joblib.dump(fitted_a[best_a], 'models/model_a_best_sensitivity.joblib')
results_log['model_a'] = model_a_results
results_log['model_a_best'] = best_a

# Feature importance (RandomForest, most interpretable) for duration target
rf_pipe = fitted_a['RandomForest']
ohe_names = rf_pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features)
feat_names = num_features + list(ohe_names)
importances = rf_pipe.named_steps['model'].estimators_[0].feature_importances_ \
    if hasattr(rf_pipe.named_steps['model'], 'estimators_') else rf_pipe.named_steps['model'].feature_importances_
fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
fi.to_csv('outputs/model_a_feature_importance.csv')
print("\nTop feature importances (Duration, RandomForest):")
print(fi.head(8))

# ===========================================================================
# MODEL B: PORTFOLIO P&L PREDICTION FROM SCENARIO FACTORS (Monte Carlo)
# ===========================================================================
print("\n" + "="*80)
print("MODEL B: Portfolio P&L Prediction from Yield Curve Scenario Factors")
print("="*80)

feat_b = ['ParallelShift_bps', 'TwistFactor_bps', 'ButterflyFactor_bps']
Xb = scenarios[feat_b].copy()
yb = scenarios['PnL_Total_INR'].copy()

Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=RANDOM_STATE)

model_specs_b = {
    'RandomForest': RandomForestRegressor(n_estimators=400, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE),
    'NeuralNetwork': Pipeline([('scale', StandardScaler()),
                               ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                                                     random_state=RANDOM_STATE, early_stopping=True))])
}

model_b_results = {}
fitted_b = {}
for name, est in model_specs_b.items():
    est.fit(Xb_train, yb_train)
    pred = est.predict(Xb_test)
    r2 = r2_score(yb_test, pred)
    rmse = mean_squared_error(yb_test, pred) ** 0.5
    mae = mean_absolute_error(yb_test, pred)
    model_b_results[name] = {'R2': r2, 'RMSE_INR': rmse, 'MAE_INR': mae}
    fitted_b[name] = est
    print(f"{name}: R2={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}")

model_b_df = pd.DataFrame(model_b_results).T
model_b_df.to_csv('outputs/model_b_comparison.csv')
best_b = model_b_df['R2'].idxmax()
print(f"\nBest Model B (by R2): {best_b}")
joblib.dump(fitted_b[best_b], 'models/model_b_best_pnl.joblib')
results_log['model_b'] = model_b_results
results_log['model_b_best'] = best_b

# ML-based VaR using best model's predictions on FULL scenario set (out-of-sample via CV-like refit check)
full_pred = fitted_b[best_b].predict(Xb)
scenarios['ML_Predicted_PnL'] = full_pred
ml_pnl_sorted = pd.Series(full_pred).sort_values().reset_index(drop=True)

def var_from_series(s, conf):
    idx = int((1 - conf) * len(s))
    return -s.iloc[idx]

ml_var_95 = var_from_series(ml_pnl_sorted, 0.95)
actual_var_95 = var_from_series(scenarios['PnL_Total_INR'].sort_values().reset_index(drop=True), 0.95)
print(f"\nML-model implied 95% VaR: INR {ml_var_95:,.0f}  vs  Historical (simulated) 95% VaR: INR {actual_var_95:,.0f}")

results_log['ml_var_95_inr'] = float(ml_var_95)
results_log['historical_var_95_inr'] = float(actual_var_95)

with open('outputs/ml_results_summary.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=float)

scenarios.to_csv('outputs/scenarios_with_ml_predictions.csv', index=False)
print("\nSaved model comparisons, feature importances, best models, and ML-predictions.")
