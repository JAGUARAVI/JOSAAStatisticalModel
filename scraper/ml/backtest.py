"""Historical backtesting: compare PyMC hierarchy vs WLR baseline.

Trains on data up to year T-1, predicts year T for T in {2023, 2024, 2025}.

Usage:
    cd scraper && python -m ml.backtest --quick
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaner import load_all_data
from ml.baseline import predict_wlr
from ml.features import engineer_features
from ml.train import predict_frame_for_backtest, train_model


def backtest_bayesian(df_train_features: pd.DataFrame, df_actuals: pd.DataFrame, test_year: int, args) -> list[dict]:
    """Train the Bayesian model, predict test_year, and compare with actuals."""
    fit = train_model(
        df_train_features,
        quick=args.quick,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        advi_steps=args.advi_steps,
    )

    pred_df = predict_frame_for_backtest(fit, df_train_features, test_year)
    if pred_df.empty:
        return []

    actuals = df_actuals.rename(columns={
        "round": "round_num",
        "closing_rank": "actual",
    }).copy()

    merged = pd.merge(
        pred_df,
        actuals,
        on=["institute", "program", "quota", "category", "gender", "round_num", "year"],
        how="inner",
        suffixes=("", "_actual"),
    )
    if merged.empty:
        return []

    merged["in_ci"] = (merged["actual"] >= merged["ci_low"]) & (merged["actual"] <= merged["ci_high"])
    return [{
        "key": (
            row["institute"], row["program"], row["quota"],
            row["category"], row["gender"], row["round_num"],
        ),
        "actual": float(row["actual"]),
        "predicted": float(row["predicted"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "in_ci": bool(row["in_ci"]),
    } for _, row in merged.iterrows()]


def backtest_wlr(df_train: pd.DataFrame, df_actuals: pd.DataFrame, test_year: int) -> list[dict]:
    """Predict using WLR baseline and compare with actuals."""
    wlr_preds_dict = predict_wlr(df_train, test_year)
    if not wlr_preds_dict:
        return []

    rows = []
    for key, pred_info in wlr_preds_dict.items():
        rows.append({
            "institute": key[0],
            "program": key[1],
            "quota": key[2],
            "category": key[3],
            "gender": key[4],
            "round_num": key[5],
            "predicted": pred_info["predicted"],
            "std_dev": pred_info["std_dev"],
            "year": test_year,
        })
    df_preds = pd.DataFrame(rows)
    df_preds["ci_low"] = np.maximum(1, df_preds["predicted"] - 1.645 * df_preds["std_dev"])
    df_preds["ci_high"] = df_preds["predicted"] + 1.645 * df_preds["std_dev"]

    actuals = df_actuals.rename(columns={
        "round": "round_num",
        "closing_rank": "actual",
    }).copy()

    merged = pd.merge(
        df_preds,
        actuals,
        on=["institute", "program", "quota", "category", "gender", "round_num", "year"],
        how="inner",
    )
    if merged.empty:
        return []

    merged["in_ci"] = (merged["actual"] >= merged["ci_low"]) & (merged["actual"] <= merged["ci_high"])
    return [{
        "key": (
            row["institute"], row["program"], row["quota"],
            row["category"], row["gender"], row["round_num"],
        ),
        "actual": float(row["actual"]),
        "predicted": float(row["predicted"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "in_ci": bool(row["in_ci"]),
    } for _, row in merged.iterrows()]


def compute_metrics(results: list[dict]) -> dict:
    """Compute MAE, RMSE, calibration error, and coverage rate."""
    if not results:
        return {"mae": None, "rmse": None, "coverage": None, "cal_error": None, "n": 0}

    errors = np.array([r["actual"] - r["predicted"] for r in results], dtype=float)
    coverage = float(np.mean([r["in_ci"] for r in results]) * 100)
    return {
        "mae": round(float(np.mean(np.abs(errors))), 1),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 1),
        "coverage": round(coverage, 1),
        "cal_error": round(abs(coverage - 90.0), 1),
        "n": len(results),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    if metrics["n"] == 0:
        print(f"    No {label} results")
        return
    print(
        f"    MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, "
        f"Coverage: {metrics['coverage']}%, Cal Error: {metrics['cal_error']}%, "
        f"N: {metrics['n']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest PyMC hierarchy against WLR")
    parser.add_argument("--quick", action="store_true", help="Use ADVI for each backtest fit")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--advi-steps", type=int, default=4000)
    args = parser.parse_args()

    test_years = [2023, 2024, 2025]

    print("=" * 60)
    print("  JoSAA Prediction Engine - Historical Backtest")
    print("  PyMC Hierarchy vs Weighted Linear Regression")
    print("=" * 60)

    df_all = load_all_data()
    print(f"\nTotal records loaded: {len(df_all)}")

    all_results = {"bayesian": {}, "wlr": {}}

    for test_year in test_years:
        print(f"\n{'-' * 50}")
        print(f"  TEST YEAR: {test_year}")
        print(f"  Training on: <= {test_year - 1}")
        print(f"{'-' * 50}")

        df_train = df_all[df_all["year"] < test_year].copy()
        df_test = df_all[df_all["year"] == test_year].copy()
        print(f"  Train records: {len(df_train)}, Test records: {len(df_test)}")

        print("\n  Bayesian hierarchy backtest...")
        df_features = engineer_features(df_train, target_year=test_year)
        bayes_results = backtest_bayesian(df_features, df_test, test_year, args)
        bayes_metrics = compute_metrics(bayes_results)
        all_results["bayesian"][test_year] = bayes_metrics
        _print_metrics("Bayesian", bayes_metrics)

        print("\n  WLR baseline backtest...")
        wlr_results = backtest_wlr(df_train, df_test, test_year)
        wlr_metrics = compute_metrics(wlr_results)
        all_results["wlr"][test_year] = wlr_metrics
        _print_metrics("WLR", wlr_metrics)

    print(f"\n{'=' * 72}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 72}")
    header = f"{'Year':<8} | {'Model':<12} | {'MAE':>8} | {'RMSE':>8} | {'Coverage':>8} | {'CalErr':>6} | {'N':>6}"
    print(header)
    print("-" * len(header))

    for year in test_years:
        bayes = all_results["bayesian"].get(year)
        wlr = all_results["wlr"].get(year)
        if bayes:
            print(f"{year:<8} | {'PyMC':<12} | {bayes['mae']:>8} | {bayes['rmse']:>8} | {bayes['coverage']:>7}% | {bayes['cal_error']:>5}% | {bayes['n']:>6}")
        if wlr:
            print(f"{'':<8} | {'WLR':<12} | {wlr['mae']:>8} | {wlr['rmse']:>8} | {wlr['coverage']:>7}% | {wlr['cal_error']:>5}% | {wlr['n']:>6}")
        print()


if __name__ == "__main__":
    main()
