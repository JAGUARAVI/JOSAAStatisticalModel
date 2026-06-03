# PyMC Hierarchical Bayesian Model Handoff

## Summary
This handoff documents the replacement of the CatBoost-based JoSAA prediction pipeline with a PyMC hierarchical Bayesian model. The implementation keeps the React frontend's existing `predictions.json` lookup contract intact while adding deterministic encoders, optional seat-matrix enrichment, posterior predictive intervals, and backtesting against the existing weighted linear regression baseline.

The model is designed to work without historic seat matrices. If no seat data exists for a rank row, the pipeline sets `seats = 0`, `log_seats = 0`, and `seats_missing = True`, allowing rank history to remain the primary signal.

## Original Plan
- Preserve both opening and closing ranks from scraped OR/CR tables.
- Add optional current seat-matrix scraping and normalize seats into `(year, institute, program, quota, category, gender, seats)`.
- Make historical seats optional, not required.
- Replace CatBoost with PyMC and ArviZ.
- Save deterministic encoders as `app/public/data/encoders.json`.
- Train a hierarchical Bayesian log-rank model with:
  - global intercept and year trend
  - institute-tier, institute, and program partial pooling
  - quota, category, gender, and ordered round effects
  - covariates for opening rank, previous closing rank, rolling history, sparse data, and optional seats
  - heteroskedastic uncertainty from sparse history, new programs, and missing seats
- Export `app/public/data/predictions.json` with the existing frontend-compatible schema:
  - `pred` as posterior median rank
  - `ci_low` / `ci_high` as 90% posterior interval
  - `mu` as posterior mean minus median
  - `sigma` as posterior rank-space standard deviation
- Update docs and app-facing model language from CatBoost/residual wording to hierarchical Bayesian posterior wording.

## Implemented
- `scraper/cleaner.py`
  - Now preserves `opening_rank` in loaded modeling data.
  - Frontend `ranks.json` export remains compatible with the existing app shape.
- `scraper/ml/features.py`
  - Rewritten for PyMC-oriented feature engineering.
  - Adds deterministic encoders for institute, program, quota, category, gender, and tier.
  - Adds leakage-safe temporal features including previous log closing rank, rolling mean/std, YoY change, observation counts, sparse/new-program flags, and year index.
  - Adds optional seat merge and missing-seat indicators.
- `scraper/ml/train.py`
  - Replaced CatBoost training with a PyMC hierarchical Bayesian model.
  - Supports full NUTS by default and ADVI smoke/fast mode via `--quick`.
  - Saves `predictions.json`, `encoders.json`, and a posterior trace when possible.
  - Fixed PyMC 6 dimension-name conflict by using dimension `row` and likelihood variable `closing_rank_obs`.
- `scraper/ml/backtest.py`
  - Rewritten to compare the PyMC model against WLR for held-out years.
  - Uses the same posterior prediction path as production artifact generation.
- `scraper/seats.py`
  - Added optional saved seat-matrix HTML parser and normalized CSV writer.
  - Normalizes JoSAA seat columns such as `GEN-EWS`, `OPEN-PwD`, and `OBC-NCL-PwD`.
- `scraper/crawler.py`
  - Added `--kind ranks|seats|all`.
  - `--kind seats` performs a best-effort scrape of the currently available JoSAA seat matrix.
- `scraper/pyproject.toml` and `scraper/uv.lock`
  - Removed CatBoost as the primary ML dependency.
  - Added PyMC, ArviZ, and pytest.
- Tests
  - Added feature tests for missing-seat behavior, temporal leakage, and current-seat merge.
  - Added seat parser normalization test.
- Docs and UI copy
  - Updated `README.md`, `ML_WORKFLOW.md`, `scraper/README.md`, `JoSAABae_Article.md`, `app/index.html`, and model-facing copy in `app/src/App.tsx`.

## Current Generated Artifacts
- `app/public/data/predictions.json`
  - `model`: `pymc-hierarchical-lognormal`
  - `fit_method`: `advi`
  - prediction count: `66663`
- `app/public/data/encoders.json`
  - Contains deterministic encoder groups for `category`, `gender`, `institute`, `program`, `quota`, and `tier`.
- `scraper/seat_data/`
  - Present as an optional generated/source directory for current seat matrix enrichment.

## Commands
Use `uv` from the `scraper` directory.

```bash
uv sync
uv run python crawler.py                 # scrape OR/CR rank tables
uv run python crawler.py --kind seats    # optional current seat matrix scrape
uv run python cleaner.py                 # regenerate ranks.json + metadata.json
uv run python -m ml.train --quick        # ADVI fit, practical smoke/fast artifact generation
uv run python -m ml.train                # full NUTS fit, much slower
uv run python -m ml.train --validate     # validate predictions.json
uv run python -m ml.backtest --quick     # quick historical backtest
```

Frontend verification:

```bash
cd app
npm run build
```

## Verification Completed
- `uv run python -m pytest tests`
  - 4 tests passed under the project `uv` environment.
- PyMC runtime import verified:
  - `pymc 6.0.1`
- Tiny PyMC model-construction smoke test passed after the `row`/`closing_rank_obs` rename.
- `uv run python -m ml.train --validate`
  - Validated the generated `predictions.json`.
- `npm run build`
  - TypeScript and Vite production build completed successfully.

## Notes and Caveats
- Full NUTS on the current full dataset is expensive. The observed loaded dataset was about `514719` training records, so `--quick` is the practical development path.
- The app still performs some client-side fallback and distribution visualization logic in `app/src/lib/regression.ts`; this is intentionally retained for compatibility when ML artifacts are missing.
- Historic seat matrices are intentionally not required. Current or saved seat data can improve features, but missing seats are a modeled condition.
- The frontend prediction consumer still expects human-readable keys in `predictions.json`, not encoded IDs.
- `predictions.json` can be large and should be regenerated only after scraper/model changes are intentional.

## Suggested Next Steps
- Run `uv run python -m ml.backtest --quick` and record metrics in `ML_WORKFLOW.md` once the desired artifact fit is finalized.
- Consider adding a smaller stratified training option for faster local experimentation before attempting full NUTS.
- If full NUTS is required, run it on a machine with enough time and memory, then inspect ArviZ diagnostics before publishing artifacts.
