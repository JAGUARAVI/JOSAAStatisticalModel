# JOSAA Rank Analyzer & Predictor – Handoff Document

## Project Vision

Build a React application that:

1. Compares JEE Main / JEE Advanced ranks against historical JoSAA opening and closing rank data.
2. Recommends the best attainable colleges and branches based on predicted cutoffs.
3. Predicts admission probability using multi-year linear regression and variance modeling.
4. Works as a static, high-performance client-side deployment.

---

# Architecture Decisions

## Chosen Stack

### Data Collection
- **Python + Playwright**: Robust automation for ASP.NET WebForms.
- **Robust Crawler**: Implements `wait_for_stable_table` logic to prevent partial data capture and has year-level recovery.

### Data Processing
- **Pandas**: Efficient cleaning and normalization.
- **JSON Pipeline**: Custom NaN-safe export to `ranks.json`.

### Frontend
- **React (Vite) + TypeScript**: Modern, type-safe development.
- **Tailwind CSS**: Premium "Glassmorphism" UI with dark mode.
- **Lucide Icons**: Modern iconography.

### Analytics
- **simple-statistics**: Client-side linear regression.
- **Logistic CDF Probability**: Probability modeled via cumulative distribution function around forecasted cutoffs.

### Visualization
- **Recharts**: Interactive historical trend charting for every college/program.

---

# Project Status

## Phase 1 – Data Scraping
**Status: COMPLETED**
- Automated discovery of all years (2016–2025) and rounds.
- Robust handling of hidden "Chosen" dropdowns.
- Logic implemented to wait for full table stabilization (fixing historical partial-capture bugs).
- Retry/Recovery system at the round and year levels.

## Phase 2 – Data Normalization
**Status: COMPLETED**
- Consolidated 10 years of data into a unified, optimized schema.
- Automatic handling of schema changes (e.g., missing Gender columns in older datasets).
- **NaN-safe serialization**: Fixed JSON parsing errors caused by null/invalid values in raw data.
- Extraction of "Last Round" data for accurate year-over-year forecasting.

## Phase 3 – Frontend & UI
**Status: COMPLETED**
- **Premium Dashboard**: High-quality dark theme with grid/list toggles.
- **Advanced Filtering**: Probability-based filtering (Safe/Target/Dream).
- **Responsive Charts**: Trends correctly span full-width in grid view when expanded.
- **Performance**: Instant client-side search and regression on ~70k+ records.

## Phase 4 – Prediction Engine
**Status: COMPLETED**
- Linear regression with trend detection (Aggressive/Recessive).
- Standard deviation-based variance modeling.
- Automatic target-year detection.

---

# Notable Technical Improvements

### 1. Robust Scraper
Successfully addressed the issue where JoSAA's dynamic loading caused rounds to return inconsistent row counts. The new `wait_for_stable_table()` function ensures we wait until the server has finished sending the entire dataset before saving the HTML.

### 2. JSON Integrity
Fixed a `JSON.parse` crash caused by Python/Pandas `NaN` values leaking into the output. The cleaner now uses a custom `safe_value` serializer.

### 3. Grid Visualization
Fixed a CSS layout bug where expanded charts were cramped in grid mode. Added `col-span-full` logic to allow detail views to utilize the full screen width.

---

# Remaining Work / Future Roadmap

### 1. Data Optimization
Currently, the `ranks.json` is large. We could optimize this by:
- Storing institute and program names as integer IDs in a separate dictionary.
- Compressing data files using Brotli or Gzip for faster initial load.

### 2. Quota Logic
Implement more complex quota handling (Home State vs Other State) if specific state data is added to the scraper.

### 3. Deployment
The application is ready for deployment on **GitHub Pages** or **Vercel** as a static site. No backend is required.

---

# Recommended Next Steps
1. Perform one last "Full Scrape" to verify 100% row stabilization across all 2024/2025 rounds.
2. Build and deploy for production use.
