# 🐍 JoSAABae - JoSAA Bayesian Pipeline

This directory contains the automated data collection and machine learning pipeline used to power **JoSAABae**.

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
| **1. Crawl** | `python crawler.py` | Fetches raw opening/closing rank HTML tables from JoSAA server using Playwright. |
| **1b. Seats** | `python crawler.py --kind seats` | Optionally fetches the currently available seat matrix. Historic seat matrices are not required. |
| **2. Clean** | `python cleaner.py` | Parses HTML and generates `ranks.json` and `metadata.json`. |
| **3. Train** | `python -m ml.train` | Trains the PyMC hierarchical Bayesian model and exports posterior predictions. |
| **4. Verify** | `python -m ml.backtest` | Runs historical backtests to ensure model calibration. |

## 🧬 ML Architecture

- **Model**: PyMC hierarchical log-rank model.
- **Features**: Opening/closing rank history, lagged closing ranks, rolling trends, sparse-data indicators, and optional current seat counts.
- **Uncertainty**: The model exports posterior median ranks, rank-space posterior standard deviation, and 90% credible intervals.

## 📂 Directory Structure

- `/raw_data`: Local cache of JoSAA HTML tables.
- `/ml`: Source code for feature engineering, training, and backtesting.
- `cleaner.py`: The main data normalization script.
- `crawler.py`: Playwright-based automation script.
