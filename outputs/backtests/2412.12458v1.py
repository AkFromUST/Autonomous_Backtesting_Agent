#!/usr/bin/env python3
"""
OU Pairs‑Trading Backtest – replicates the algorithm described in the brief.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
DATA_DIR = Path("/Users/aarav/Developer/Applications/NineMasts_ColdEmail/prototype_1/data/paper1")
with open(DATA_DIR / "data_manifest.json") as f:
    manifest = json.load(f)

prices = pd.read_parquet(DATA_DIR / "prices.parquet")
# pivot to wide format: rows = dates, columns = tickers, values = adj_close
close = (
    prices.pivot(index="date", columns="ticker", values="adj_close")
    .sort_index()
    .ffill()
    .dropna(axis=1, how="all")
)
# ensure DateTime index
close.index = pd.to_datetime(close.index)

print("Data loaded – shape:", close.shape)

# ----------------------------------------------------------------------
# 2. Define training / test windows
# ----------------------------------------------------------------------
train_start = pd.Timestamp("2018-01-01")
train_end = pd.Timestamp("2020-12-31")
test_start = pd.Timestamp("2021-01-01")

close_train = close.loc[train_start:train_end]
close_test = close.loc[test_start:]

# ----------------------------------------------------------------------
# 3. Pair selection – MSD on returns, one‑to‑one mapping
# ----------------------------------------------------------------------
rets_train = close_train.pct_change().dropna()

tickers = rets_train.columns.tolist()
n = len(tickers)

# compute MSD for each unordered pair
msd = {}
for i in range(n):
    for j in range(i + 1, n):
        ti, tj = tickers[i], tickers[j]
        diff = rets_train[ti] - rets_train[tj]
        msd_val = np.mean(diff ** 2)
        msd[(ti, tj)] = msd_val

# sort pairs by smallest MSD
sorted_pairs = sorted(msd.items(), key=lambda x: x[1])

# greedy one‑to‑one selection
selected_pairs = []
used = set()
for (pair, _) in sorted_pairs:
    if pair[0] not in used and pair[1] not in used:
        selected_pairs.append(pair)
        used.update(pair)

print("PAIR SELECTION – number of selected pairs:", len(selected_pairs))

# ----------------------------------------------------------------------
# 4. Optional cointegration filter (ADF on OLS residuals)
# ----------------------------------------------------------------------

def passes_cointegration(x, y):
    """Return True if residuals of x~y are stationary (ADF p<0.05)."""
    # Align the two series on the same dates and drop any NaNs
    df = pd.concat([x, y], axis=1, join='inner')
    df.columns = ['x', 'y']
    df = df.dropna()
    if len(df) < 30:
        return False
    model = sm.OLS(df['x'], sm.add_constant(df['y'])).fit()
    resid = model.resid
    adf_res = sm.tsa.stattools.adfuller(resid, autolag="AIC")
    pvalue = adf_res[1]
    return pvalue < 0.05

filtered_pairs = []
for i, j in selected_pairs:
    if passes_cointegration(close_train[i], close_train[j]):
        filtered_pairs.append((i, j))
print("COINTEGRATION FILTER – pairs after filter:", len(filtered_pairs))

# ----------------------------------------------------------------------
# 5. Estimate OU parameters on training spreads (once)
# ----------------------------------------------------------------------

def estimate_ou_params(spread_series):
    """Fit AR(1): S_{t+1} = a*S_t + b + eps, return lambda, mu, sigma, a, b."""
    s = spread_series.dropna()
    s_lag = s.shift(1).iloc[1:]
    s_cur = s.iloc[1:]
    X = sm.add_constant(s_lag)
    model = sm.OLS(s_cur, X).fit()
    a = model.params.iloc[1]
    b = model.params.iloc[0]
    eps = model.resid
    delta = 1.0
    lam = -np.log(a) / delta
    mu = b / (1 - a)
    sigma = np.std(eps, ddof=1) * np.sqrt(-2 * np.log(a) / (delta * (1 - a ** 2)))
    return lam, mu, sigma, a, b

pair_params = {}
for i, j in filtered_pairs:
    spread = close_train[i] - close_train[j]
    lam, mu, sigma, a, b = estimate_ou_params(spread)
    pair_params[(i, j)] = {"lam": lam, "mu": mu, "sigma": sigma, "a": a, "b": b}

lam_vals = [v["lam"] for v in pair_params.values()]
print(
    "OU PARAMETERS – lambda stats:",
    f"count={len(lam_vals)} mean={np.mean(lam_vals):.4f} min={np.min(lam_vals):.4f} max={np.max(lam_vals):.4f}",
)

# ----------------------------------------------------------------------
# 6. Build Z‑score series for the test period (using fixed params)
# ----------------------------------------------------------------------
z_series = {}
for (i, j), pars in pair_params.items():
    spread_test = close_test[i] - close_test[j]
    mu, sigma = pars["mu"], pars["sigma"]
    z = (spread_test - mu) / sigma
    z_series[(i, j)] = z

# ----------------------------------------------------------------------
# 7. Rolling 90‑day percentile (look‑back only)
# ----------------------------------------------------------------------
window = 90
percentile_series = {}
for pair, z in z_series.items():
    perc = (
        z.rolling(window, min_periods=window)
        .apply(
            lambda s: stats.percentileofscore(s[:-1], s[-1]) if len(s) > 1 else np.nan,
            raw=False,
        )
        .shift(1)
    )
    percentile_series[pair] = perc

# ----------------------------------------------------------------------
# 8. Signal generation & backtest
# ----------------------------------------------------------------------
open_positions = {}
portfolio_returns = []

dates_test = close_test.index
for t_idx, today in enumerate(dates_test):
    daily_ret = 0.0
    # Exit logic – evaluate using today's percentile
    for pair in list(open_positions.keys()):
        perc = percentile_series[pair].loc[today]
        if np.isnan(perc):
            continue
        direction = open_positions[pair]["direction"]
        if (direction == 1 and perc >= 50) or (direction == -1 and perc <= 50):
            i, j = pair
            ret_i = close_test[i].pct_change().loc[today]
            ret_j = close_test[j].pct_change().loc[today]
            pair_ret = direction * (ret_i - ret_j)
            daily_ret += pair_ret
            del open_positions[pair]
    # Entry signals – use yesterday's percentile to avoid look‑ahead
    if t_idx > 0:
        yesterday = dates_test[t_idx - 1]
        for pair in percentile_series.keys():
            if pair in open_positions:
                continue
            perc_y = percentile_series[pair].loc[yesterday]
            if np.isnan(perc_y):
                continue
            if perc_y <= 25:
                direction = 1
            elif perc_y >= 75:
                direction = -1
            else:
                continue
            # Enter position; P&L will be realized on next day's price move
            open_positions[pair] = {"direction": direction, "entry_date": today}
    portfolio_returns.append(daily_ret)

# Close any remaining positions on the final day
final_day = dates_test[-1]
for pair, pos in open_positions.items():
    i, j = pair
    ret_i = close_test[i].pct_change().loc[final_day]
    ret_j = close_test[j].pct_change().loc[final_day]
    pair_ret = pos["direction"] * (ret_i - ret_j)
    portfolio_returns[-1] += pair_ret

# ----------------------------------------------------------------------
# 9. Equity curve & diagnostics
# ----------------------------------------------------------------------
portfolio_returns = np.array(portfolio_returns, dtype=float)

equity = np.cumprod(1 + portfolio_returns)
print("EQUITY SERIES – NaN check:", np.isnan(equity).any())

# ----------------------------------------------------------------------
# 10. Performance metrics
# ----------------------------------------------------------------------

daily_ret_series = pd.Series(portfolio_returns, index=dates_test)
sharpe = np.mean(daily_ret_series) / np.std(daily_ret_series, ddof=1) * np.sqrt(252)
total_return = equity[-1] - 1.0
annualized_return = equity[-1] ** (252 / len(daily_ret_series)) - 1

print(f"Sharpe Ratio: {sharpe:.6f}")
print(f"Total Return: {total_return:.6f}")
print(f"Annualized Return: {annualized_return:.6f}")

# ----------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------
