# Quant Forex AI — Phases 1–11

Educational **quantitative forex research** platform. Phases 1–10 cover data through a simulated paper book. Phase 11 is a **read-only Streamlit dashboard** over local reports. It cannot place broker orders.

Live trading is **disabled**. Setting `LIVE_TRADING=true` or `execution.live_trading: true` raises an error.

---

## What you can do right now

1. Install Python 3.12+ and this package.
2. Run the unit tests.
3. Ingest the bundled **synthetic** EURUSD sample (offline).
4. Or ingest public daily EURUSD from Yahoo Finance (needs internet).
5. Or drop in your own CSV (Histdata, Dukascopy, broker export) and point the config at it.
6. Open the data-quality report under `reports/` and the processed CSV under `data/processed/`.
7. Run exploration: charts plus a Markdown report under `reports/`.
8. Build a feature dataset (`python -m quant_forex features`) and read the feature report.
9. Run baseline strategies (`python -m quant_forex baselines`) and inspect signals.
10. Run a costed backtest (`python -m quant_forex backtest`).
11. Inspect conservative sizing (`python -m quant_forex risk-preview`).
12. Train logistic / random forest / gradient boosting (`python -m quant_forex train`). Model choice uses **validation ROC-AUC only**.
13. Run walk-forward evaluation (`python -m quant_forex walk-forward`).
14. Run overfitting / statistical diagnostics (`python -m quant_forex diagnose`).
15. Replay a **simulated** paper book (`python -m quant_forex paper --enable-paper`).
16. Open the research dashboard (`python -m quant_forex dashboard`).

What you **cannot** do: neural nets, or any live broker order.

---

## 1. Setup (Windows PowerShell)

From the project root (`ForexTrendsProj`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,dashboard]"
```

If `py -3.12` is not found, use `python` as long as `python --version` is 3.12 or newer.

Optional extras (not required for Phase 1):

```powershell
python -m pip install -e ".[ml,api,dashboard]"
```

Copy environment defaults:

```powershell
copy .env.example .env
```

Leave `LIVE_TRADING=false`.

---

## 2. Run the tests

Always from the project root, with the venv active:

```powershell
pytest
```

You should see Phase 1–6 tests pass. They use synthetic bars and **do not** need the network.

---

## 3. Run the offline example (bundled sample)

```powershell
python -m quant_forex ingest --config configs/default.yaml
```

Equivalent:

```powershell
python scripts/ingest_example.py
```

This reads `data/external/sample_eurusd_daily.csv` (synthetic random-walk OHLC, **not** real EURUSD), writes:

| Output | Where |
| --- | --- |
| Unchanged source CSV | `data/external/sample_eurusd_daily.csv` |
| Raw snapshot (copy) | `data/raw/<dataset_id>_raw.csv` |
| Processed CSV | `data/processed/<dataset_id>_processed.csv` |
| SQLite | `data/processed/market_data.sqlite` |
| Quality report (Markdown + JSON) | `reports/<dataset_id>_data_quality.md` |

Read the Markdown report. It lists row counts, date range, missing values, duplicates, abnormal OHLC, inferred frequency, timezone, and **every transformation** that was applied.

---

## 3b. Explore the processed data (Phase 2)

After ingest:

```powershell
python -m quant_forex explore --config configs/default.yaml
```

Or:

```powershell
python scripts/explore_example.py
```

Point at a specific file:

```powershell
python -m quant_forex explore --input data/processed\YOURFILE_processed.csv
```

Outputs:

| Output | Where |
| --- | --- |
| Exploration report | `reports/<id>_exploration.md` |
| JSON summary | `reports/<id>_exploration.json` |
| PNG charts | `reports/figures/<id>/` |
| Interactive candles | `reports/figures/<id>/candlestick.html` |

Charts include close, return histogram, rolling volatility (no future bars), weekday return boxes, cumulative **spot** log return (labelled as not P&L), and a Plotly candlestick.

Daily bars cannot identify London vs New York sessions (timestamps sit at midnight UTC). Session flags only activate on intraday timestamps.

---

## 3c. Build features (Phase 3)

After ingest:

```powershell
python -m quant_forex features --config configs/default.yaml
```

Or:

```powershell
python scripts/build_features.py
```

This reads the newest `data/processed/*_processed.csv` and writes:

| Output | Where |
| --- | --- |
| Feature table | `data/processed/<id>_features.csv` |
| Feature report | `reports/<id>_features.md` |
| JSON (columns + univariate correlations) | `reports/<id>_features.json` |

Groups (toggle in `configs/default.yaml` under `features.groups`):

- **price** — simple/log returns, rolling returns, range, body, wicks
- **trend** — SMA, EMA, close vs MA, MACD
- **momentum** — RSI (Wilder), ROC, momentum
- **volatility** — rolling return std, ATR, Bollinger width, high-vol regime vs a *rolling* median of vol (not a full-sample quantile — that would leak)
- **time** — hour, weekday, London/NY/overlap flags (meaningful on intraday bars only)

Rolling windows include the current bar and **no future bars**. Warmup NaNs at the start are kept unless `drop_warmup_rows: true`.

The report lists Pearson correlation of each feature at time *t* with the **next** bar’s return. That is a crude screen, not proof of an edge, and not model feature importance (that comes with fitted models later). **Do not** assume every column helps.

Set `attach_next_return_target: true` only if you want a clearly named `target_next_simple_return` column (last row is empty by construction). Default is off so the feature file is not silently a labeled dataset.

---

## 3d. Baseline strategies (Phase 4)

```powershell
python -m quant_forex baselines --config configs/default.yaml
```

Or `python scripts/run_baselines.py`.

| Output | Where |
| --- | --- |
| Positions (every bar) | `data/processed/<id>_baseline_positions.csv` |
| Entry signals (on position change) | `data/processed/<id>_baseline_signals.csv` |
| Report | `reports/<id>_baselines.md` |

Rules (all causal; shorts optional via `strategies.allow_short`):

1. **Buy-and-hold** — always long the pair (FX benchmark, not an equity hold)
2. **SMA crossover** — long if fast SMA > slow SMA
3. **Momentum** — sign of N-bar return vs a threshold
4. **Mean reversion** — fade a rolling z-score (rolling mean/std, not the full sample)
5. **Breakout** — Donchian of the *previous* N highs/lows, then hold until the opposite break

Position size on baseline *signals* is still empty: size is computed at fill time from then-current equity. See backtest trades for actual units.

The comparison table uses **next-bar** simple returns with **zero costs**. It exists so you can see the rules fire. It is not evidence of profitability.

---

## 3e. Costed backtest (Phase 5)

```powershell
python -m quant_forex backtest --config configs/default.yaml
python -m quant_forex backtest --strategy ma_crossover
```

Or `python scripts/run_backtest.py`.

| Output | Where |
| --- | --- |
| Report | `reports/<id>_backtest.md` |
| Trades | `data/processed/<id>_backtest_trades.csv` |
| Equity | `data/processed/<id>_backtest_equity.csv` |
| Charts | `reports/figures/<id>_backtest/` |

**Fill rule:** a position decided at bar *t* close is filled at bar *t+1* **open**.

**Costs:** half-spread plus slippage on every entry and exit; optional commission per lot. Defaults are **not** zero (`spread_pips: 1.0`, `slippage_pips: 0.2`).

**Stops / targets:** ATR multiples from strategy config. If both would trade in the same OHLC bar, the engine takes the **stop** (worst case). A gap through the stop fills at the open, still with costs.

**Sizing / limits:** the risk manager takes the **minimum** of (a) fixed-fraction cash at risk vs the stop and (b) volatility targeting, then applies `max_risk_per_trade`, `max_gross_exposure`, `max_leverage`, and `max_position_size`. `max_daily_loss` flattens and blocks new entries until the next day. `max_drawdown` flattens and **halts** further trading. These caps are not tuned for return.

Account currency is assumed to be the pair’s quote (EURUSD → USD). Crosses such as USDJPY convert JPY PnL at the exit rate when the account is USD.

This is an OHLC model, not tick data. A profitable backtest is not an edge.

---

## 3f. Risk management (Phase 6)

```powershell
python -m quant_forex risk-preview --config configs/default.yaml
```

That prints an **indicative** size at `backtest.initial_capital` for the latest processed close (not a live order). Actual backtest size is path-dependent (equity at each fill).

`configs/default.yaml` → `risk`:

| Key | Default | Role |
| --- | --- | --- |
| `sizing_method` | `min` | `fixed_fraction`, `volatility`, or `min` of both |
| `risk_per_trade` | 0.5% | Target cash at risk vs stop |
| `max_risk_per_trade` | 0.5% | Hard cap (cannot be looser than the target) |
| `annual_vol_target` | 10% | Vol targeting uses *causal* rolling return std |
| `max_gross_exposure` | 2× | Notional / equity; cannot exceed `max_leverage` |
| `max_concurrent_positions` | 1 | One open position per market in this engine |

Never raise these to “make the backtest look better.”

---

## 3g. Machine-learning classifiers (Phase 7)

```powershell
python -m quant_forex train --config configs/default.yaml
```

Or `python scripts/train_models.py`.

This fits **logistic regression** (with a train-only scaler), **random forest**, and **sklearn gradient boosting**. There is no neural net and no XGBoost in this phase.

| Rule | Why |
| --- | --- |
| Target is `close[t+horizon] / close[t] - 1 > threshold` | Label uses the future; that column is **never** in `X`. |
| Chronological train / validation / test | No shuffle. Config dates that miss this sample fall back to time **fractions**. |
| Select by validation ROC-AUC | Test metrics are reported but **not** used to pick a model or tune depth/trees. |
| Signals use real probabilities | LONG if `P(up) ≥ probability_long`, SHORT if `P(up) ≤ probability_short`, else FLAT. |
| Optional test-fold backtest | Train and validation bars stay flat; only the test timestamps get ML positions, then Phase 5 costs apply. |

| Output | Where |
| --- | --- |
| Report | `reports/<id>_ml.md` |
| Metrics JSON | `reports/<id>_ml.json` |
| Test predictions | `data/processed/<id>_ml_predictions.csv` |
| Fitted model | `data/processed/<id>_ml_<name>.joblib` |

A high train score with a weak test score is a warning, not a strategy. The synthetic sample is not real EURUSD.

---

## 3h. Walk-forward evaluation (Phase 8)

```powershell
python -m quant_forex walk-forward --config configs/default.yaml
```

Or `python scripts/run_walk_forward.py`.

A single train/validation/test split (Phase 7) can look good by luck. Walk-forward **refits** on past bars only, chooses the model on a **validation** slice that is still before the test window, then scores the next `test_bars`. Then it steps forward and repeats.

| Rule | Default | Why |
| --- | --- | --- |
| `mode` | `expanding` | Training origin stays at the start (`rolling` uses a fixed-length window). |
| `train_bars` / `validation_bars` / `test_bars` | 80 / 20 / 20 | Sizes in **bars**, not calendar years. |
| Purge | `horizon + embargo_bars` | Labels use future closes; those rows are not in the test window. |
| Selection | validation ROC-AUC **per fold** | Test bars are never used to pick logistic vs forest vs boosting. |
| OOS backtest | concatenated test positions | Fit/validation bars stay flat; Phase 5 costs still apply. |

| Output | Where |
| --- | --- |
| Report | `reports/<id>_walkforward.md` |
| Metrics JSON | `reports/<id>_walkforward.json` |
| OOS predictions | `data/processed/<id>_walkforward_predictions.csv` |
| AUC chart | `reports/figures/<id>_walkforward/oos_auc.png` |

If `step_bars` is smaller than `test_bars`, test windows overlap and later folds overwrite earlier positions at the same timestamp.

This is still not live trading and does not prove an edge.

---

## 3i. Overfitting and statistical diagnostics (Phase 9)

```powershell
python -m quant_forex diagnose --config configs/default.yaml
```

Or `python scripts/run_diagnostics.py`.

A costed backtest can look good by luck, by a magical parameter pair, or by leakage. This command does **not** approve a strategy. It reports:

| Check | What it does |
| --- | --- |
| Moving-block bootstrap | Confidence interval on mean equity-bar return and Sharpe (blocks keep some serial dependence). |
| Circular-shift permutation | Keeps the mix of long/short/flat but breaks timing, then re-runs the **costed** engine. |
| Return / drawdown distributions | Percentiles of bar returns and of the drawdown path. |
| Monthly / annual tables | Calendar returns from the equity curve. |
| Parameter sensitivity | Nearby MA windows and momentum lookbacks; an isolated Sharpe spike is flagged. |
| Train vs test AUC | Reads existing `*_ml.json` / `*_walkforward.json` if you already ran those commands. |
| Too-good-to-be-true | High return with tiny drawdown prints: *Potential overfitting or unrealistic assumptions. Investigate before proceeding.* |

| Output | Where |
| --- | --- |
| Report | `reports/<id>_diagnostics.md` |
| JSON | `reports/<id>_diagnostics.json` |
| Charts | `reports/figures/<id>_diagnostics/` |

Hyperparameters are still **not** searched on the test fold. Raising `max_depth` to chase in-sample accuracy is a complexity knob, not a feature.

---

## 3j. Paper trading (Phase 10, still not live)

```powershell
python -m quant_forex paper --config configs/default.yaml --enable-paper --strategy ma_crossover
```

Or `python scripts/run_paper.py`.

This **replays local OHLCV** into a simulated book. It uses the same next-open fills, spread, slippage, and risk manager as the backtest. Differences:

- You must opt in (`--enable-paper` or `execution.paper_trading: true`).
- The last bar is **not** flattened. An open simulated position can remain.
- A signal on the last close is a **pending next-open** intent until a newer bar exists.
- Fills are stored in `data/processed/paper_ledger.sqlite` with `venue=paper_simulator`.
- `submit_live_order` always raises. There is no broker adapter.

A paper PnL is not a live result and not an edge.

---

## 3k. Research dashboard (Phase 11)

```powershell
python -m pip install -e ".[dashboard]"
python -m quant_forex dashboard --config configs/default.yaml
```

Or `python scripts/run_dashboard.py`.

Streamlit opens a **read-only** viewer of local files: processed prices, backtest equity, ML/walk-forward JSON, diagnostic flags, and the simulated paper snapshot. There is no order ticket. The Overview tab’s live-order button is expected to fail.

Run ingest (and any later commands you care about) first, or the tabs will show empty-state messages.

---

## 4. Download public EURUSD from Yahoo Finance

Yahoo is convenient for a first look. It is **not** tick data, often has empty FX volume, and can differ from a broker’s feed.

```powershell
python -m quant_forex ingest --config configs/yfinance_eurusd.yaml
```

Change pair without editing code:

```powershell
python -m quant_forex ingest --config configs/yfinance_eurusd.yaml --symbol GBPUSD
```

If you change `--symbol`, also set a matching ticker in YAML (`GBPUSD=X`, `USDJPY=X`, …) or set `yfinance_ticker` to `null` so the code derives `{SYMBOL}=X`.

Yahoo daily FX typically covers weekdays only. Weekend gaps are expected.

---

## 5. Use your own historical data (recommended for real research)

Put a CSV in `data/external/` and set `data.csv_path` in `configs/default.yaml` (or pass `--csv-path`).

### Required shape

Columns (case-insensitive; common aliases work):

- `timestamp` (or `date`, `time`, `datetime`)
- `open`, `high`, `low`, `close`
- `volume` optional (often missing for FX — that is fine; it is recorded, not invented)

Example:

```text
timestamp,open,high,low,close,volume
2020-01-02,1.1220,1.1228,1.1160,1.1172,
2020-01-03,1.1171,1.1179,1.1124,1.1145,
```

If timestamps have **no timezone**, they are treated as `data.timezone` in the YAML (default `UTC`) and then stored as UTC. Wrong timezone is a research bug — set it explicitly if your file is London local time (`Europe/London`).

### Where to get data

Use sources you are allowed to use for a university project. Check each site’s terms.

| Source | Typical use | Notes |
| --- | --- | --- |
| **Your own CSV** | Best control | Export from a broker MT4/MT5 history center if you have an account. |
| **[HistData.com](https://www.histdata.com/)** | M1 / tick ZIP files | Popular for FX research. You must convert their format to the CSV columns above (Phase 1 does not parse their native `DAT_ASCII` layout yet). |
| **[Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)** | Tick / bar exports | High resolution; large files. Convert to OHLCV CSV. |
| **Yahoo Finance via yfinance** | Daily FX | Built-in (`configs/yfinance_eurusd.yaml`). Easy, lower quality. |
| **OANDA / Twelve Data / Polygon / Alpha Vantage** | REST APIs | Need API keys. Not wired in Phase 1; `httpx` is available for a later adapter. Do not commit keys. |
| **TrueFX / other tick archives** | Tick research | Convert to bars yourself if you need OHLC. |

**Do not** scrape a broker in violation of their terms. **Do not** expect free daily Yahoo bars to match a live account.

After you save `data/external/eurusd_daily.csv`:

```powershell
python -m quant_forex ingest --config configs/default.yaml --csv-path data/external/eurusd_daily.csv --symbol EURUSD
```

---

## 6. Configuration

Edit YAML rather than code. Start with `configs/default.yaml`.

Important keys:

- `instrument.symbol` — e.g. `EURUSD` or `GBP/USD` (normalized to six letters)
- `timeframe` — currently informational for daily CSV (`1d`)
- `data.source` — `csv` or `yfinance`
- `data.missing_value_policy` — `flag` (default: **do not fill**), `drop`, or `ffill` (forward-fill only, logged)
- `data.abnormal_return_threshold` — flags large close-to-close moves (default 5%)
- `costs` / `risk` — documented now so later backtests do not assume zero spread; unused in Phase 1
- `execution.live_trading` — must stay `false`

Majors listed in `configs/instruments.yaml`: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD. JPY pairs use `pip_size: 0.01` in that catalog.

---

## 7. How the pipeline treats data

Design rule: **never silently rewrite history**.

1. Load from CSV or yfinance.
2. Standardize column names (logged).
3. Build a **raw** quality report (issues found, nothing “fixed” yet).
4. Save a raw snapshot under `data/raw/`. The file you provided is left untouched.
5. Apply **explicit** cleaning from config: UTC timestamps, numeric coercion, optional sort, optional duplicate drop, missing-value policy, optional resample.
6. Flag abnormal OHLC (high < low, non-positive prices) without deleting those rows unless you chose a drop policy.
7. Save processed CSV, SQLite bars, and a quality report.

If a step removes rows, the log records `rows_before` → `rows_after`. If the default `flag` policy is used, missing prices stay missing.

---

## 8. Jupyter

```powershell
python -m pip install -e ".[notebooks]"
jupyter notebook notebooks/08_dashboard.ipynb
```

Run ingestion from the project root first, or execute the notebook cells which call the same pipeline.

---

## 9. Project layout

```text
configs/            YAML settings
data/raw/           snapshots of loaded data
data/processed/     cleaned CSV + SQLite
data/external/      your files + bundled synthetic sample
notebooks/          01 … 07 paper, 08 dashboard
src/quant_forex/    library (… `execution` paper simulator, no broker)
tests/              pytest
reports/            quality, exploration, feature, baseline, and backtest reports
scripts/            thin CLI wrappers
```

Research stays in `notebooks/`. Reusable logic stays in `src/`.

---

## 10. Roadmap (not built yet)

| Phase | Topic |
| --- | --- |
| 1 | Project setup + data pipeline (**done**) |
| 2 | Data exploration and visualization (**done**) |
| 3 | Feature engineering (**done**) |
| 4 | Baseline strategies (**done**) |
| 5 | Backtesting engine (**done**) |
| 6 | Risk management (**done**) |
| 7 | Machine-learning models (**done**) |
| 8 | Walk-forward testing (**done**) |
| 9 | Overfitting / statistical analysis (**done**) |
| 10 | Paper trading (**done**, still not live) |
| 11 | Dashboard (**this release**) |

---

## Academic honesty

The sample EURUSD file is **synthetic**. A passing test suite does not mean a strategy works. Do not submit backtests or dashboard screenshots as evidence of an edge unless you used real (or clearly cited) data with transaction costs and out-of-sample evaluation.
