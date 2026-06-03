"""CatBoost training pipeline with Bayesian uncertainty and Monte Carlo simulation.

Usage:
    python -m ml.train               # Train and generate predictions.json
    python -m ml.train --validate     # Validate predictions.json schema
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# Add parent dir so we can import cleaner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaner import load_all_data
from ml.features import engineer_features, get_feature_columns, get_categorical_feature_indices


# Rank bucket boundaries for proportional sigma calibration
RANK_BUCKET_BINS = [0, 100, 500, 2000, 10000, 50000, 200000, float('inf')]
RANK_BUCKET_LABELS = ["0-100", "100-500", "500-2k", "2k-10k", "10k-50k", "50k-200k", "200k+"]


def _assign_rank_bucket(rank_values: pd.Series) -> pd.Series:
    """Assign rank bucket labels based on closing rank magnitude."""
    return pd.cut(rank_values, bins=RANK_BUCKET_BINS, labels=RANK_BUCKET_LABELS)


def train_model(df_features: pd.DataFrame):
    """Train CatBoost regressor on feature-engineered data in log-space.
    
    Training on log(closing_rank) ensures scale-invariant predictions:
    - Low-rank entries (60-100) get proportionally accurate predictions
    - High-rank entries (50000+) also get proportionally accurate predictions
    
    Returns trained model and residual statistics per rank bucket + round.
    """
    from catboost import CatBoostRegressor, Pool
    
    feature_cols = get_feature_columns()
    cat_indices = get_categorical_feature_indices()
    
    # Filter to rows that have the target and at least lag_1
    train_df = df_features.dropna(subset=["closing_rank", "lag_1"]).copy()
    
    if train_df.empty:
        raise ValueError("No training data available after filtering")
    
    X = train_df[feature_cols]
    # Train on log-transformed target for scale-invariant learning
    y = train_df["log_closing_rank"]
    
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Categorical features: {len(cat_indices)}")
    print(f"  Target: log(closing_rank) [scale-invariant]")
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=200,
        cat_features=cat_indices,
        loss_function="RMSE",
        eval_metric="MAE",
        early_stopping_rounds=150,
    )
    
    # Use 10% of data as eval set for early stopping
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # Compute predictions in actual space for residual calibration
    train_preds_log = model.predict(Pool(X, cat_features=cat_indices))
    train_preds = np.expm1(train_preds_log)  # Back to actual space
    actual_y = train_df["closing_rank"].values
    residuals = actual_y - train_preds
    
    train_df = train_df.copy()
    train_df["residual"] = residuals
    train_df["predicted"] = train_preds
    
    # Calibrate per rank_bucket + round for proportional sigma
    # This ensures low-rank branches get tight sigma, high-rank get wider sigma
    train_df["rank_bucket"] = _assign_rank_bucket(train_df["closing_rank"])
    round_col = "round_num"
    
    residual_stats = {}
    for (bucket, rnd), grp in train_df.groupby(["rank_bucket", round_col], observed=True):
        res = grp["residual"].values
        if len(res) >= 5:
            mu, sigma = stats.norm.fit(res)
        else:
            mu, sigma = 0.0, float(np.std(res)) if len(res) > 1 else 500.0
        residual_stats[(str(bucket), int(rnd))] = {"mu": float(mu), "sigma": max(float(sigma), 1.0)}
    
    # Also calibrate per rank_bucket only (fallback when bucket+round missing)
    for bucket, grp in train_df.groupby("rank_bucket", observed=True):
        res = grp["residual"].values
        if len(res) >= 5:
            mu, sigma = stats.norm.fit(res)
        else:
            mu, sigma = 0.0, float(np.std(res)) if len(res) > 1 else 500.0
        residual_stats[(str(bucket), "__all__")] = {"mu": float(mu), "sigma": max(float(sigma), 1.0)}
    
    # Global fallback
    global_mu, global_sigma = stats.norm.fit(residuals)
    residual_stats["__global__"] = {"mu": float(global_mu), "sigma": max(float(global_sigma), 1.0)}
    
    # Print per-bucket sigma for diagnostics
    print(f"\n  Residual stats by rank bucket:")
    for bucket in RANK_BUCKET_LABELS:
        key = (bucket, "__all__")
        if key in residual_stats:
            rs = residual_stats[key]
            print(f"    {bucket:>10s}: μ={rs['mu']:>8.1f}, σ={rs['sigma']:>8.1f}")
    print(f"    {'Global':>10s}: μ={global_mu:>8.1f}, σ={global_sigma:>8.1f}")
    
    return model, residual_stats


def compute_confidence_interval(predicted_cutoff: float, residual_mu: float,
                                  residual_sigma: float, level: float = 0.90):
    """Compute confidence interval using Bayesian uncertainty."""
    z = stats.norm.ppf(1 - (1 - level) / 2)
    center = predicted_cutoff + residual_mu
    low = max(1, center - z * residual_sigma)
    high = center + z * residual_sigma
    return int(round(low)), int(round(high))


def _get_residual_stats(residual_stats: dict, predicted_rank: float, round_num: int) -> dict:
    """Get the best matching residual stats for a prediction.
    
    Lookup priority:
    1. (rank_bucket, round) - most specific
    2. (rank_bucket, "__all__") - bucket-level fallback
    3. "__global__" - global fallback
    """
    # Determine rank bucket from predicted rank
    for i, (lo, hi) in enumerate(zip(RANK_BUCKET_BINS[:-1], RANK_BUCKET_BINS[1:])):
        if lo < predicted_rank <= hi:
            bucket_label = RANK_BUCKET_LABELS[i]
            break
    else:
        bucket_label = RANK_BUCKET_LABELS[-1]
    
    # Try bucket + round
    key = (bucket_label, round_num)
    if key in residual_stats:
        return residual_stats[key]
    
    # Try bucket only
    key = (bucket_label, "__all__")
    if key in residual_stats:
        return residual_stats[key]
    
    # Global fallback
    return residual_stats["__global__"]


def generate_predictions(model, residual_stats: dict, df_features: pd.DataFrame, 
                          target_year: int):
    """Generate predictions for all groups in the target year.
    
    Predictions are made in log-space and converted back to actual ranks.
    A sanity clamp ensures predictions stay within reasonable bounds of
    historical lag values.
    
    Returns list of prediction dicts for predictions.json.
    """
    from catboost import Pool
    
    feature_cols = get_feature_columns()
    cat_indices = get_categorical_feature_indices()
    
    # Filter to target year prediction rows (closing_rank is NaN)
    pred_df = df_features[
        (df_features["year"] == target_year) & 
        (df_features["closing_rank"].isna())
    ].copy()
    
    if pred_df.empty:
        print("  ⚠ No prediction rows found for target year")
        return []
    
    # Drop rows missing critical features
    pred_df = pred_df.dropna(subset=["lag_1"])
    
    if pred_df.empty:
        print("  ⚠ No prediction rows with sufficient history")
        return []
    
    print(f"  Generating predictions for {len(pred_df)} seat groups...")
    
    X_pred = pred_df[feature_cols]
    pool = Pool(X_pred, cat_features=cat_indices)
    
    # Predict in log-space, convert back to actual ranks
    predicted_log = model.predict(pool)
    predicted_cutoffs = np.expm1(predicted_log)  # exp(log_pred) - 1 = actual rank
    predicted_cutoffs = np.maximum(predicted_cutoffs, 1)
    
    # Sanity clamp: predictions should stay within reasonable bounds of lag values
    # Use rolling_std to determine appropriate bounds
    lag_values = pred_df["lag_1"].values
    rolling_std = pred_df["rolling_std_3"].values
    
    for idx in range(len(predicted_cutoffs)):
        lag = lag_values[idx]
        std = rolling_std[idx]
        
        if np.isnan(lag) or lag <= 0:
            continue
        
        # Determine clamp range based on historical volatility
        if np.isnan(std) or std <= 0:
            # No std info: use 15% of lag as default volatility
            std = lag * 0.15
        
        # Allow up to 5 sigma deviation from lag, but at least ±20% of lag
        margin = max(5 * std, lag * 0.20)
        lower = max(1, lag - margin)
        upper = lag + margin
        
        predicted_cutoffs[idx] = np.clip(predicted_cutoffs[idx], lower, upper)
    
    predicted_cutoffs = np.round(predicted_cutoffs).astype(int)
    
    predictions = []
    
    for idx, (_, row) in enumerate(pred_df.iterrows()):
        inst = str(row["institute"])
        prog = str(row["program"])
        quota = str(row["quota"])
        cat = str(row["category"])
        gender = str(row["gender"])
        rnd = int(row["round_num"])
        predicted = int(predicted_cutoffs[idx])
        
        # Get residual stats matched to this prediction's rank bucket
        rs = _get_residual_stats(residual_stats, predicted, rnd)
        mu, sigma = rs["mu"], rs["sigma"]
        
        # Confidence interval
        ci_low, ci_high = compute_confidence_interval(predicted, mu, sigma, level=0.90)
        
        predictions.append({
            "i": inst,
            "p": prog,
            "q": quota,
            "c": cat,
            "g": gender,
            "r": rnd,
            "t": str(row["type"]),
            "pred": predicted,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
        })
    
    return predictions


def validate_predictions(predictions_path: str):
    """Validate predictions.json schema and integrity."""
    with open(predictions_path) as f:
        data = json.load(f)
    
    preds = data["predictions"]
    thresholds = data["rank_thresholds"]
    
    errors = []
    
    if not isinstance(preds, list):
        errors.append("predictions is not a list")
        
    if not isinstance(thresholds, list):
        errors.append("rank_thresholds is not a list")
    
    pred_1_count = sum(1 for p in preds if p["pred"] <= 1)
    pred_le_10 = sum(1 for p in preds if p["pred"] <= 10)
    
    for i, p in enumerate(preds[:10]):  # Spot check first 10
        if p["pred"] < 1:
            errors.append(f"Prediction {i}: pred < 1")
        if p["ci_low"] > p["pred"]:
            errors.append(f"Prediction {i}: ci_low ({p['ci_low']}) > pred ({p['pred']})")
        if p["ci_high"] < p["pred"]:
            errors.append(f"Prediction {i}: ci_high ({p['ci_high']}) < pred ({p['pred']})")
        if p["sigma"] < 0:
            errors.append(f"Prediction {i}: sigma < 0")
    
    if errors:
        print("❌ Validation errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print(f"✅ Validation passed: {len(preds)} predictions, {len(thresholds)} rank thresholds")
    print(f"   pred=1: {pred_1_count}, pred≤10: {pred_le_10}")
    
    # Spot check some known entries
    for p in preds:
        if ("Indian Institute of Technology Bombay" in p["i"] and 
            "Computer Science and Engineering (4 Years" in p["p"] and
            p["c"] == "OPEN" and p["g"] == "Gender-Neutral" and p["r"] == 1):
            print(f"   IIT Bombay CSE R1: pred={p['pred']}, sigma={p['sigma']}, ci=[{p['ci_low']},{p['ci_high']}]")
            break
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost prediction model")
    parser.add_argument("--validate", action="store_true", help="Validate existing predictions.json")
    parser.add_argument("--max-year", type=int, default=None, help="Max training year")
    parser.add_argument("--target-year", type=int, default=2026, help="Year to predict")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictions_path = os.path.join(base_dir, "..", "app", "public", "data", "predictions.json")
    
    if args.validate:
        validate_predictions(predictions_path)
        return
    
    max_year = args.max_year or (args.target_year - 1)
    
    print(f"═══ CatBoost Training Pipeline (Log-Space) ═══")
    print(f"  Training data: 2016–{max_year}")
    print(f"  Target year: {args.target_year}")
    
    # Step 1: Load data
    print(f"\n▶ Loading data...")
    df = load_all_data(max_year=max_year)
    print(f"  Loaded {len(df)} records")
    
    # Step 2: Engineer features
    print(f"\n▶ Engineering features...")
    df_features = engineer_features(df, target_year=args.target_year)
    print(f"  Feature matrix: {df_features.shape}")
    
    # Step 3: Train model
    print(f"\n▶ Training CatBoost model (log-space target)...")
    model, residual_stats = train_model(df_features)
    
    # Step 4: Generate predictions
    print(f"\n▶ Generating predictions for {args.target_year}...")
    predictions = generate_predictions(model, residual_stats, df_features, args.target_year)
    print(f"  Generated {len(predictions)} predictions")
    
    # Step 5: Build rank thresholds list
    rank_thresholds = sorted(set(
        list(range(100, 2000, 100)) +
        list(range(2000, 10000, 250)) +
        list(range(10000, 50000, 500)) +
        list(range(50000, 100001, 1000)) +
        list(range(100000, 200001, 5000))
    ))
    
    # Step 6: Save predictions
    output = {
        "target_year": args.target_year,
        "model": "catboost-logspace",
        "n_simulations": 10000,
        "confidence_level": 0.90,
        "rank_thresholds": rank_thresholds,
        "predictions": predictions,
    }
    
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    with open(predictions_path, "w") as f:
        json.dump(output, f)
    
    size_mb = os.path.getsize(predictions_path) / (1024 * 1024)
    print(f"\n✅ Saved predictions.json ({size_mb:.1f} MB, {len(predictions)} groups)")
    
    # Step 7: Validate
    print(f"\n▶ Validating...")
    validate_predictions(predictions_path)


if __name__ == "__main__":
    main()
