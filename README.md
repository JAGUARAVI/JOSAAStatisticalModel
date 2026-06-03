# 🔍 JoSAA Forensics

### **Advanced Admission Analytics & ML-Powered Rank Prediction for JEE Aspirants**

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-CatBoost-ff69b4?style=for-the-badge)](https://catboost.ai/)

**JoSAA Forensics** is a premium, open-source diagnostic suite designed to help JEE Main and Advanced students navigate the complexities of JoSAA/CSAB counseling. Unlike simple lookup tools, it uses **Bayesian Machine Learning** and **Monte Carlo Simulations** to provide probabilistic admission insights.

---

## ✨ Key Features

- **🧠 ML Prediction Engine**: Powered by CatBoost and Bayesian residuals, providing 90% Confidence Intervals for every institute-program pair.
- **🎲 Monte Carlo Simulations**: Runs 10,000+ simulations per query in real-time to calculate exact admission probabilities.
- **📊 Historical Forensics**: Visualize 10 years (2016–2025) of opening and closing rank data with interactive Recharts.
- **🎨 Premium UI**: A modern "Glassmorphism" interface with **Dark/Light mode** support, optimized for desktop and mobile.
- **⚡ Client-Side Performance**: Instant filtering and searching on 70k+ records using an optimized local dataset.
- **🛡️ Legal & Privacy**: Fully client-side processing. Your rank data never leaves your browser.

---

## 📂 Project Structure

```bash
├── app/              # React + Vite Frontend (TypeScript, Tailwind CSS)
├── scraper/          # Python Scraper & ML Pipeline (Playwright, CatBoost)
├── ML_WORKFLOW.md    # Detailed guide on the ML training & data pipeline
└── ARCHITECTURE.md   # System design and technical handoff documentation
```

---

## 🚀 Quick Start

### 1. Frontend Development (React)
```bash
cd app
npm install
npm run dev
```

### 2. Data Pipeline (Python)
The repository comes with pre-processed data. To update it:
```bash
cd scraper
uv sync  # or pip install -r requirements.txt
python cleaner.py     # Process raw HTML to JSON
python -m ml.train    # Train ML models and export predictions.json
```

---

## 🛠 Tech Stack

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS, Lucide Icons.
- **Visualization**: Recharts (Custom SVG charting).
- **ML/Data**: Python, CatBoost, Pandas, Scrapy/Playwright.
- **Analytics**: Simple Statistics (Client-side failover engine).

---

## 📜 Legal & Disclaimer

*JoSAA Forensics is an independent community project and is NOT affiliated with JoSAA, CSAB, or any IIT/NIT. Predictions are probabilistic and should be used for informational purposes only.*

---

## 🤝 Contributing

Contributions are welcome! Whether it's improving the ML model, refining the UI, or updating the datasets, feel free to open a PR.

---

**Built with ❤️ for JEE Aspirants.**
