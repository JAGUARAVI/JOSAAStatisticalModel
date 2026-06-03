# 🔬 JoSAABae: JoSAA Bayesian Pipeline Workflow

This document outlines the end-to-end process for updating admission predictions, from scraping raw historical tables to exporting the final PyMC-powered probability matrix.

## 🛠️ Core Stack
- **Data Collection**: [Playwright](https://playwright.dev/) + python (Headless automation for JoSAA portal)
- **Data Cleaning**: python 3.13+ (Pandas, StringIO)
- **ML Engine**: [PyMC](https://www.pymc.io/) hierarchical Bayesian model
- **Probabilistic Modeling**: Posterior predictive rank distributions with 90% credible intervals
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
uv run crawler.py
```
*Note: The crawler implements `wait_for_stable_table` logic to ensure 100% data capture from dynamic ASP.NET pages.*

Optional current seat matrix enrichment:
```bash
uv run crawler.py --kind seats
```
Historic seat matrices are not required. If no seat data is present, the model sets `seats_missing=True` and trains from rank history.

### 3. Normalize & Clean Data
Convert the raw HTML tables into the optimized JSON format used by the frontend:
```bash
# This parses raw_data/*.html and exports app/public/data/ranks.json + metadata.json
uv run cleaner.py
```

### 4. Feature Engineering & ML Training
Run the ML pipeline to train the PyMC hierarchical model and generate probabilistic forecasts:
```bash
# Quick smoke-test fit
uv run python -m ml.train --quick

# Checkpointed ADVI with validation-based model selection
uv run python -m ml.train --checkpoint-advi

# Checkpointed full NUTS run with validation-based checkpoint selection
uv run python -m ml.train

# Full posterior fit
uv run python -m ml.train
```
**Pipeline Details**:
- **Feature Extraction**: Computes lag features, rolling statistics, opening-rank signals, sparse-data indicators, and optional seat-count features (`ml/features.py`).
- **Bayesian Modeling**: Fits global, tier, institute, program, quota, category, gender, and round effects with partial pooling.
- **Checkpointing**: ADVI and full NUTS runs can save intermediate checkpoints under `scraper/ml/checkpoints/` and select the best one using a held-out historical validation split.
- **Posterior Export**: Converts posterior predictive samples into median ranks, uncertainty, and 90% credible intervals.
- **Output**: Generates `app/public/data/predictions.json`.

### 5. Deployment
Build the static frontend once the data files are updated:
```bash
cd ../app
npm run build
```

---

## 🛠️ Troubleshooting & Tuning

- **Sampling Runtime**: Use `uv run python -m ml.train --quick` for smoke tests; full NUTS sampling is intentionally slower.
- **System Memory**: Reduce `--draws`, `--chains`, or use `--quick` if posterior fitting is too heavy.
- **GitHub Push Errors**: The `.json` files in `app/public/data/` are ignored by Git to avoid exceeding file size limits. Ensure these are uploaded to a CDN or served locally during deployment.
- **Failover**: If `predictions.json` is missing, the React frontend automatically triggers an in-browser **Weighted Linear Regression (WLR)** engine.

---

## 📊 Methodology
The engine treats admission as a **Probabilistic Forecasting** problem:
1. **Point Estimate**: The posterior median predicts the most likely closing rank based on historical shifts.
2. **Distribution**: The model directly estimates posterior predictive uncertainty in rank space.
3. **Posterior Sampling**: We calculate the percentage chance of your rank falling inside the expected cutoff from posterior samples.
4. **Temporal Awareness**: Probabilities are computed per round to account for the gradual "thinning" of the seat pool.
