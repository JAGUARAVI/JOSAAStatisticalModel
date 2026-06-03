# JoSAA Rank Analyzer: Data & ML Pipeline Workflow

This document outlines the end-to-end process for updating the admission predictions, from scraping historical data to exporting the final CatBoost-powered probability matrix.

## Core Stack
- **Data Engineering**: Python 3.10+ (Scrapy, Pandas)
- **ML Engine**: CatBoost (Gradient Boosting on Decision Trees)
- **Uncertainty**: Bayesian Residual Analysis + Monte Carlo Simulation (10k iterations)
- **Frontend**: React + TypeScript (Static lookup)

---

## 🚀 Execution Workflow

### 1. Environment Setup
Ensure you have the required dependencies installed (managed via `uv` or `pip`):
```bash
cd scraper
# Using uv (recommended)
uv sync
# Or using pip
pip install -r requirements.txt
```

### 2. Scrape/Update Historical Data
If new JoSAA rounds are released or you need to re-fetch data:
```bash
# This will fetch all data from 2016-2025 and save to app/public/data/ranks.json
python cleaner.py
```
*Note: `cleaner.py` now includes the `load_all_data()` logic to parse HTML tables from the `/data/` subdirectory.*

### 3. Feature Engineering & Training
Run the ML pipeline to train the CatBoost model and generate probabilistic forecasts:
```bash
# Trains model, runs 10k Monte Carlo simulations, and exports predictions.json
python -m ml.train
```
**What happens here?**
- Loads historical ranks from `ranks.json`.
- Computes lag features, rolling statistics, and trend slopes (`ml/features.py`).
- Trains a `CatBoostRegressor` with Bayesian residual modeling.
- Runs 10,000 simulations per institute-program-category-gender combination.
- **Output**: `app/public/data/predictions.json`

### 4. Backtesting (Optional)
To verify model accuracy against historical "future" years (e.g., predicting 2025 using 2016-2024):
```bash
python -m ml.backtest
```
This ensures the model maintains low calibration error before production.

### 5. Frontend Build
Once `predictions.json` is updated, build the static site:
```bash
cd ../app
npm run build
```

---

## 🛠 Troubleshooting

- **Memory Issues**: If the training process crashes, reduce `N_SIMULATIONS` in `ml/train.py` (currently set to 10,000).
- **Missing Clusters**: If a new institute is added to JoSAA, ensure `cleaner.py` and the scraper handle its naming convention consistently with previous years to avoid feature gaps.
- **Fallbacks**: If `predictions.json` is missing or fails to load in the browser, the frontend automatically falls back to the **Weighted Linear Regression (WLR)** engine.

---

## 📊 Model Methodology
The engine treats admission prediction as a **Probabilistic Forecasting** problem:
1. **Point Estimate**: CatBoost predicts the most likely closing rank.
2. **Distribution**: Bayesian residuals define a normal distribution around that estimate.
3. **Monte Carlo**: We sample 10,000 points from this distribution to see how often the user's rank "fits" inside the cutoff.
4. **Rounding**: Probabilities are computed per round to account for late-stage volatility.
