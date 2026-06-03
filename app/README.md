# 📱 JoSAABae - Frontend (JoSAA Bayesian)

This is the React-based frontend for **JoSAABae**. It provides a high-performance, interactive dashboard for JEE rank analysis.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm run dev
   ```

3. **Production Build**:
   ```bash
   npm run build
   ```

## 🏗 Key Components

- **`App.tsx`**: The main application logic, state management, and primary UI layout.
- **`lib/regression.ts`**: The client-side fallback prediction engine using weighted linear regression.
- **`index.css`**: The core design system, colors, and animations.

## 📊 Data Dependencies

The frontend expects the following files in `public/data/`:
- `ranks.json`: Historical opening/closing ranks (2016-2025).
- `metadata.json`: Category, Gender, and Round mappings.
- `predictions.json`: ML-generated probability matrices from the Python pipeline.

## 🎨 UI Features

- **Theme Toggle**: Switch between Dark and Light modes.
- **Search & Filters**: Real-time filtering by institute, program, category, and match status (Safe/Target/Dream).
- **Monte Carlo Charts**: Detailed rank distribution visualizations.
- **Mobile Responsive**: Fully optimized for mobile counseling sessions.
