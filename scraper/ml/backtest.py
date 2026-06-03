"""Historical backtesting: compare CatBoost vs WLR baseline.

Trains on data up to year T-1, predicts year T for T ∈ {2023, 2024, 2025}.
Reports MAE, RMSE, calibration error, and coverage rate.

Usage:
    cd scraper && python -m ml.backtest
"""
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaner import load_all_data
from ml.features import engineer_features, get_feature_columns, get_categorical_feature_indices
from ml.baseline import predict_wlr
from ml.train import train_model, compute_confidence_interval, _get_residual_stats


def backtest_catboost(df_train_features, df_actuals, test_year):
    """Train CatBoost on features, predict test_year, compare with actuals."""
    from catboost import Pool
    
    feature_cols = get_feature_columns()
    cat_indices = get_categorical_feature_indices()
    
    # Train
    model, residual_stats = train_model(df_train_features)
    
    # Get prediction rows for test_year
    pred_df = df_train_features[
        (df_train_features["year"] == test_year) & 
        (df_train_features["closing_rank"].isna())
    ].copy()
    pred_df = pred_df.dropna(subset=["lag_1"])
    
    if pred_df.empty:
        return None
    
    X_pred = pred_df[feature_cols]
    pool = Pool(X_pred, cat_features=cat_indices)
    # Predict in log-space and convert back
    predicted_log = model.predict(pool)
    pred_df["predicted"] = np.maximum(np.expm1(predicted_log), 1)
    
    # Bayesian CI using rank-bucket-based residual stats
    def get_ci(row):
        rnd = int(row["round_num"])
        rs = _get_residual_stats(residual_stats, row["predicted"], rnd)
        return compute_confidence_interval(row["predicted"], rs["mu"], rs["sigma"], level=0.90)
    
    cis = pred_df.apply(get_ci, axis=1)
    pred_df["ci_low"] = [c[0] for c in cis]
    pred_df["ci_high"] = [c[1] for c in cis]
    
    # Merge with actuals
    # Rename columns for merge to avoid collisions
    df_actuals_clean = df_actuals.rename(columns={
        "round": "round_num",
        "closing_rank": "actual"
    }).copy()
    
    merged = pd.merge(
        pred_df, 
        df_actuals_clean, 
        on=["institute", "program", "quota", "category", "gender", "round_num", "year"],
        how="inner"
    )
    
    if merged.empty:
        return []
    
    merged["in_ci"] = (merged["actual"] >= merged["ci_low"]) & (merged["actual"] <= merged["ci_high"])
    
    results = []
    for _, row in merged.iterrows():
        results.append({
            "key": (row["institute"], row["program"], row["quota"], row["category"], row["gender"], row["round_num"]),
            "actual": float(row["actual"]),
            "predicted": float(row["predicted"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "in_ci": bool(row["in_ci"]),
        })
    
    return results


def backtest_wlr(df_train, df_actuals, test_year):
    """Predict using WLR baseline and compare with actuals."""
    wlr_preds_dict = predict_wlr(df_train, test_year)
    
    if not wlr_preds_dict:
        return []
        
    # Convert dict to df for merging
    rows = []
    for key, pred_info in wlr_preds_dict.items():
        rows.append({
            "institute": key[0], "program": key[1], "quota": key[2],
            "category": key[3], "gender": key[4], "round_num": key[5],
            "predicted": pred_info["predicted"],
            "std_dev": pred_info["std_dev"],
            "year": test_year
        })
    df_preds = pd.DataFrame(rows)
    
    # 90% CI for WLR
    df_preds["ci_low"] = np.maximum(1, df_preds["predicted"] - 1.645 * df_preds["std_dev"])
    df_preds["ci_high"] = df_preds["predicted"] + 1.645 * df_preds["std_dev"]
    
    df_actuals_clean = df_actuals.rename(columns={
        "round": "round_num",
        "closing_rank": "actual"
    }).copy()
    
    merged = pd.merge(
        df_preds, 
        df_actuals_clean, 
        on=["institute", "program", "quota", "category", "gender", "round_num", "year"],
        how="inner"
    )
    
    if merged.empty:
        return []
        
    merged["in_ci"] = (merged["actual"] >= merged["ci_low"]) & (merged["actual"] <= merged["ci_high"])
    
    results = []
    for _, row in merged.iterrows():
        results.append({
            "key": (row["institute"], row["program"], row["quota"], row["category"], row["gender"], row["round_num"]),
            "actual": float(row["actual"]),
            "predicted": float(row["predicted"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "in_ci": bool(row["in_ci"]),
        })
    
    return results


def compute_metrics(results):
    """Compute MAE, RMSE, calibration error, and coverage rate."""
    if not results:
        return {"mae": None, "rmse": None, "coverage": None, "cal_error": None, "n": 0}
    
    errors = [r["actual"] - r["predicted"] for r in results]
    abs_errors = [abs(e) for e in errors]
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
    coverage = float(np.mean([r["in_ci"] for r in results]) * 100)
    cal_error = abs(coverage - 90.0)  # Deviation from nominal 90% coverage
    
    return {
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "coverage": round(coverage, 1),
        "cal_error": round(cal_error, 1),
        "n": len(results),
    }


def main():
    test_years = [2023, 2024, 2025]
    
    print("═══════════════════════════════════════════════════")
    print("  JoSAA Prediction Engine — Historical Backtest   ")
    print("  CatBoost + Bayesian vs Weighted Linear Regression")
    print("═══════════════════════════════════════════════════")
    
    # Load all data
    df_all = load_all_data()
    print(f"\nTotal records loaded: {len(df_all)}")
    
    all_results = {"catboost": {}, "wlr": {}}
    
    for test_year in test_years:
        print(f"\n{'─'*50}")
        print(f"  TEST YEAR: {test_year}")
        print(f"  Training on: 2016–{test_year - 1}")
        print(f"{'─'*50}")
        
        # Split data
        df_train = df_all[df_all["year"] < test_year].copy()
        df_test = df_all[df_all["year"] == test_year].copy()
        
        print(f"  Train records: {len(df_train)}, Test records: {len(df_test)}")
        
        # --- CatBoost ---
        print(f"\n  ▶ CatBoost backtest...")
        df_features = engineer_features(df_train, target_year=test_year)
        cb_results = backtest_catboost(df_features, df_test, test_year)
        
        if cb_results:
            cb_metrics = compute_metrics(cb_results)
            all_results["catboost"][test_year] = cb_metrics
            print(f"    MAE: {cb_metrics['mae']}, RMSE: {cb_metrics['rmse']}, "
                  f"Coverage: {cb_metrics['coverage']}%, Cal Error: {cb_metrics['cal_error']}%, "
                  f"N: {cb_metrics['n']}")
        else:
            print("    ⚠ No CatBoost results")
        
        # --- WLR Baseline ---
        print(f"\n  ▶ WLR baseline backtest...")
        wlr_results = backtest_wlr(df_train, df_test, test_year)
        
        if wlr_results:
            wlr_metrics = compute_metrics(wlr_results)
            all_results["wlr"][test_year] = wlr_metrics
            print(f"    MAE: {wlr_metrics['mae']}, RMSE: {wlr_metrics['rmse']}, "
                  f"Coverage: {wlr_metrics['coverage']}%, Cal Error: {wlr_metrics['cal_error']}%, "
                  f"N: {wlr_metrics['n']}")
        else:
            print("    ⚠ No WLR results")
    
    # --- Final Comparison ---
    print(f"\n{'═'*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'═'*60}")
    
    header = f"{'Year':<8} | {'Model':<12} | {'MAE':>8} | {'RMSE':>8} | {'Coverage':>8} | {'CalErr':>6} | {'N':>6}"
    print(header)
    print("─" * len(header))
    
    cb_wins = 0
    total_years = 0
    
    for year in test_years:
        if year in all_results["catboost"] and year in all_results["wlr"]:
            cb = all_results["catboost"][year]
            wlr = all_results["wlr"][year]
            
            cb_marker = " ✓" if cb["mae"] < wlr["mae"] else ""
            wlr_marker = " ✓" if wlr["mae"] < cb["mae"] else ""
            
            print(f"{year:<8} | {'CatBoost':<12} | {cb['mae']:>8} | {cb['rmse']:>8} | {cb['coverage']:>7}% | {cb['cal_error']:>5}% | {cb['n']:>6}{cb_marker}")
            print(f"{'':<8} | {'WLR':<12} | {wlr['mae']:>8} | {wlr['rmse']:>8} | {wlr['coverage']:>7}% | {wlr['cal_error']:>5}% | {wlr['n']:>6}{wlr_marker}")
            print()
            
            if cb["mae"] < wlr["mae"]:
                cb_wins += 1
            total_years += 1
    
    # Aggregate
    if total_years > 0:
        avg_cb_mae = np.mean([all_results["catboost"][y]["mae"] for y in test_years if y in all_results["catboost"]])
        avg_wlr_mae = np.mean([all_results["wlr"][y]["mae"] for y in test_years if y in all_results["wlr"]])
        
        print(f"\n  Aggregate MAE: CatBoost={avg_cb_mae:.1f} vs WLR={avg_wlr_mae:.1f}")
        print(f"  CatBoost wins on MAE: {cb_wins}/{total_years} test years")
        
        if cb_wins >= 2:
            print(f"\n  ✅ DECISION: ADOPT CatBoost (wins {cb_wins}/{total_years} years)")
        else:
            print(f"\n  ❌ DECISION: KEEP WLR (CatBoost wins only {cb_wins}/{total_years} years)")


if __name__ == "__main__":
    main()
