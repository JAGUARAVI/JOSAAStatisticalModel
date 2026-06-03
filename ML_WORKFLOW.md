# 🔬 JoSAA Forensics: Data & ML Pipeline Workflow

This document outlines the end-to-end process for updating admission predictions, from scraping raw historical tables to exporting the final CatBoost-powered probability matrix.

## 🛠️ Core Stack
- **Data Collection**: [Playwright](https://playwright.dev/) + Python (Headless automation for JoSAA portal)
- **Data Cleaning**: Python 3.13+ (Pandas, StringIO)
- **ML Engine**: [CatBoost](https://catboost.ai/) (Gradient Boosting on Decision Trees)
- **Probabilistic Modeling**: Bayesian Residual Analysis + Monte Carlo Simulation (10k iterations)
- **Frontend**: React 19 + TypeScript 5 (In-browser Bayesian failover & static lookup)

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

### 2. Scrape Historical Data
If new JoSAA rounds are released or you need to re-fetch historical tables directly from the source:
```bash
# This uses Playwright to navigate the JoSAA archive and save HTML tables to raw_data/
python crawler.py
```
*Note: The crawler implements `wait_for_stable_table` logic to ensure 100% data capture from dynamic ASP.NET pages.*

### 3. Normalize & Clean Data
Convert the raw HTML tables into the optimized JSON format used by the frontend:
```bash
# This parses raw_data/*.html and exports app/public/data/ranks.json + metadata.json
python cleaner.py
```

### 4. Feature Engineering & ML Training
Run the ML pipeline to train the CatBoost model and generate probabilistic forecasts:
```bash
# Trains the regressor, computes Bayesian residuals, and runs 10k Monte Carlo simulations
python -m ml.train
```
**Pipeline Details**:
- **Feature Extraction**: Computes lag features, rolling statistics, and trend slopes (`ml/features.py`).
- **Bayesian Modeling**: Trains a `CatBoostRegressor` and models prediction error (residuals).
- **Simulation**: Conducts 10,000 simulations per combination to determine the probabilistic frontier.
- **Output**: Generates `app/public/data/predictions.json`.

### 5. Deployment
Build the static frontend once the data files are updated:
```bash
cd ../app
npm run build
```

---

## 🛠️ Troubleshooting & Tuning

- **Simulation Volatility**: If prediction probabilities feel too optimistic, increase the noise floor in `ml/train.py`.
- **System Memory**: If training crashes, reduce `N_SIMULATIONS` in `ml/train.py` (default: 10,000).
- **GitHub Push Errors**: The `.json` files in `app/public/data/` are ignored by Git to avoid exceeding file size limits. Ensure these are uploaded to a CDN or served locally during deployment.
- **Failover**: If `predictions.json` is missing, the React frontend automatically triggers an in-browser **Weighted Linear Regression (WLR)** engine.

---

## 📊 Methodology
The engine treats admission as a **Probabilistic Forecasting** problem:
1. **Point Estimate**: CatBoost predicts the most likely closing rank based on historical shifts.
2. **Distribution**: Bayesian analysis defines a normal distribution around that estimate based on model residuals.
3. **Monte Carlo Sampling**: We sample 10,000 points from this distribution to calculate the percentage chance of your rank falling inside the expected cutoff.
4. **Temporal Awareness**: Probabilities are computed per round to account for the gradual "thinning" of the seat pool.
