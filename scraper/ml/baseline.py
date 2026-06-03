"""Weighted Linear Regression baseline — Python port of the client-side regression.ts logic.

Used for apples-to-apples backtesting comparison against CatBoost.
"""
import numpy as np
from collections import defaultdict


def weighted_linear_regression(data: list[tuple[float, float]], decay: float = 0.8):
    """Compute WLR with exponential decay weights. Returns (slope, intercept) or None."""
    if len(data) < 2:
        return None
    
    max_x = max(d[0] for d in data)
    
    sum_w = sum_wx = sum_wy = sum_wxx = sum_wxy = 0.0
    for x, y in data:
        w = decay ** (max_x - x)
        sum_w += w
        sum_wx += w * x
        sum_wy += w * y
        sum_wxx += w * x * x
        sum_wxy += w * x * y
    
    denom = sum_w * sum_wxx - sum_wx * sum_wx
    if abs(denom) < 1e-10:
        return None
    
    m = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    b = (sum_wy - m * sum_wx) / sum_w
    return m, b


def predict_wlr(df, target_year: int) -> dict:
    """Predict closing ranks for target_year using WLR on historical data.
    
    Args:
        df: DataFrame with columns: institute, program, quota, category, gender,
            closing_rank, year, round, type
        target_year: Year to predict.
    
    Returns:
        Dict mapping (institute, program, quota, category, gender, round) -> predicted_close
    """
    predictions = {}
    
    # Group by seat identity + round
    groups = df.groupby(["institute", "program", "quota", "category", "gender", "round"])
    
    for key, group in groups:
        inst, prog, quota, cat, gender, rnd = key
        
        history = sorted(zip(group["year"].values, group["closing_rank"].values))
        
        if not history:
            continue
        
        result = weighted_linear_regression([(float(y), float(cr)) for y, cr in history])
        
        if result:
            m, b = result
            predicted = max(1, round(m * target_year + b))
            
            # Compute std dev of residuals
            residuals = [cr - (m * y + b) for y, cr in history]
            if len(residuals) > 1:
                std_dev = float(np.std(residuals, ddof=1))
            else:
                std_dev = predicted * 0.05
        else:
            predicted = history[-1][1]
            std_dev = predicted * 0.08
        
        predictions[key] = {
            "predicted": int(predicted),
            "std_dev": float(std_dev),
        }
    
    return predictions
