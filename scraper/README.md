# 🐍 JoSAA Forensics - Scraper & ML Pipeline

This directory contains the automated data collection and machine learning pipeline used to power JoSAA Forensics.

## 🛠 Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Install dependencies
uv sync
# Or using pip
pip install -r requirements.txt
```

## 🚀 Pipeline Workflow

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Crawl** | `python crawler.py` | Fetches raw HTML tables from JoSAA server using Playwright. |
| **2. Clean** | `python cleaner.py` | Parses HTML and generates `ranks.json` and `metadata.json`. |
| **3. Train** | `python -m ml.train` | Trains CatBoost model and runs Monte Carlo simulations. |
| **4. Verify** | `python -m ml.backtest` | Runs historical backtests to ensure model calibration. |

## 🧬 ML Architecture

- **Model**: CatBoost Regressor.
- **Features**: Lagged closing ranks, rolling trends, categorical embeddings for Institutes/Programs.
- **Uncertainty**: The model predicts residuals (error distribution), which are then sampled via 10,000 Monte Carlo iterations to compute the final admission probability percentage.

## 📂 Directory Structure

- `/raw_data`: Local cache of JoSAA HTML tables.
- `/ml`: Source code for feature engineering, training, and backtesting.
- `cleaner.py`: The main data normalization script.
- `crawler.py`: Playwright-based automation script.
