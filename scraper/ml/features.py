"""Feature engineering for JoSAA cutoff prediction."""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, target_year: int | None = None) -> pd.DataFrame:
    """Engineer features from raw JoSAA data for model training/prediction.
    
    Args:
        df: DataFrame with columns: institute, program, quota, category, gender,
            closing_rank, year, round, type
        target_year: If set, also generate a prediction row for this year per group.
    
    Returns:
        DataFrame with engineered features and 'closing_rank' as the target.
    """
    df = df.copy()
    df = df.sort_values(["institute", "program", "quota", "category", "gender", "round", "year"])
    
    # Group key for each unique seat
    df["group_key"] = (
        df["institute"] + "|" + df["program"] + "|" + df["quota"] + "|" +
        df["category"] + "|" + df["gender"]
    )
    
    # ---- Lag features (per group+round) ----
    group_round_key = df["group_key"] + "|R" + df["round"].astype(str)
    df["group_round_key"] = group_round_key
    
    # Last year's final round (R6) as a feature for all rounds of the current year
    r6_data = df[df["round"] == 6][["group_key", "year", "closing_rank"]].rename(columns={"closing_rank": "lag_1_r6"})
    r6_data["year"] = r6_data["year"] + 1 # Align to next year
    df = pd.merge(df, r6_data, on=["group_key", "year"], how="left")
    
    # Sort for proper lag computation
    df = df.sort_values(["group_round_key", "year"])
    
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby("group_round_key")["closing_rank"].shift(lag)
    
    # Year-over-year delta
    df["yoy_delta"] = df["closing_rank"] - df["lag_1"]
    
    # ---- Rolling statistics (per group+round, over years) ----
    for window in [3, 5]:
        rolled = df.groupby("group_round_key")["closing_rank"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).mean()
        )
        df[f"rolling_mean_{window}"] = rolled
        
        rolled_std = df.groupby("group_round_key")["closing_rank"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
        )
        df[f"rolling_std_{window}"] = rolled_std
    
    # ---- Round progression features (per group+year) ----
    group_year_key = df["group_key"] + "|Y" + df["year"].astype(str)
    
    # Max round cutoff within the same year (from prior rounds only)
    df["round_cummax"] = df.groupby(group_year_key)["closing_rank"].cummax()
    
    # Round number as ordinal feature
    df["round_num"] = df["round"]
    
    # ---- Trend features (linear slope over last N years per group+round) ----
    def compute_slope(series):
        """Compute simple linear regression slope for a series."""
        valid = series.dropna()
        if len(valid) < 2:
            return np.nan
        x = np.arange(len(valid))
        slope = np.polyfit(x, valid.values, 1)[0]
        return slope
    
    df["trend_slope_3y"] = df.groupby("group_round_key")["closing_rank"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=2).apply(compute_slope, raw=False)
    )
    
    # ---- Categorical encoding (as category type for CatBoost) ----
    cat_cols = ["institute", "program", "quota", "category", "gender", "type"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    
    # ---- Year as numeric feature ----
    df["year_num"] = df["year"]
    
    # ---- Institute-level aggregate features ----
    # Mean closing rank per institute per year (proxy for institute prestige shift)
    inst_year_mean = df.groupby(["institute", "year"])["closing_rank"].transform("mean")
    df["inst_year_mean_rank"] = inst_year_mean
    
    # ---- Log-transformed features for scale-invariant learning ----
    log_cols = ["lag_1", "lag_2", "lag_3", "lag_1_r6", "rolling_mean_3", "rolling_mean_5", 
                "round_cummax", "inst_year_mean_rank"]
    for col in log_cols:
        df[f"log_{col}"] = np.log1p(df[col])
    
    # Log-transformed target for training
    df["log_closing_rank"] = np.log1p(df["closing_rank"])
    
    # ---- Generate prediction rows for target year if requested ----
    if target_year is not None:
        pred_rows = _generate_prediction_rows(df, target_year)
        if not pred_rows.empty:
            df = pd.concat([df, pred_rows], ignore_index=True)
            # Re-sort 
            df = df.sort_values(["group_round_key", "year"])
    
    # Drop helper columns
    df = df.drop(columns=["group_key", "group_round_key"])
    
    return df


def _generate_prediction_rows(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Generate prediction rows for a target year by extrapolating lag features.
    
    Skips discontinued branches (no data within 2 years of target year) and
    branches with insufficient history.
    """
    # Get unique group+round combinations
    groups = df.groupby(["institute", "program", "quota", "category", "gender", "round", "type"])
    
    pred_rows = []
    skipped_discontinued = 0
    skipped_insufficient = 0
    
    for (inst, prog, quota, cat, gender, rnd, inst_type), group in groups:
        group = group.sort_values("year")
        if group.empty:
            continue
        
        max_data_year = group["year"].max()
        min_data_year = group["year"].min()
        n_years = group["year"].nunique()
        
        # Skip discontinued branches: no data within 2 years of target
        if max_data_year < target_year - 2:
            skipped_discontinued += 1
            continue
        
        # Skip branches with only 1 data point (unreliable predictions)
        if n_years < 2:
            skipped_insufficient += 1
            continue
        
        row = {
            "institute": inst,
            "program": prog,
            "quota": quota,
            "category": cat,
            "gender": gender,
            "round": rnd,
            "type": inst_type,
            "year": target_year,
            "closing_rank": np.nan,  # This is what we want to predict
            "log_closing_rank": np.nan,
            "year_num": target_year,
            "round_num": rnd,
        }
        
        # Lag features from actual recent data
        recent = group.tail(3)["closing_rank"].values
        row["lag_1"] = recent[-1] if len(recent) >= 1 else np.nan
        row["lag_2"] = recent[-2] if len(recent) >= 2 else np.nan
        row["lag_3"] = recent[-3] if len(recent) >= 3 else np.nan
        
        # Last year's R6
        last_r6 = group[group["round"] == 6]["closing_rank"].values
        row["lag_1_r6"] = last_r6[-1] if len(last_r6) >= 1 else np.nan
        
        row["yoy_delta"] = row["lag_1"] - row["lag_2"] if not np.isnan(row.get("lag_2", np.nan)) else np.nan
        
        # Rolling stats from available history
        hist = group["closing_rank"].values
        row["rolling_mean_3"] = np.mean(hist[-3:]) if len(hist) >= 2 else np.nan
        row["rolling_std_3"] = np.std(hist[-3:], ddof=1) if len(hist) >= 2 else np.nan
        row["rolling_mean_5"] = np.mean(hist[-5:]) if len(hist) >= 2 else np.nan
        row["rolling_std_5"] = np.std(hist[-5:], ddof=1) if len(hist) >= 2 else np.nan
        
        # Round cummax (first round of the year, so just lag_1)
        row["round_cummax"] = row["lag_1"] if not np.isnan(row.get("lag_1", np.nan)) else np.nan
        
        # Trend slope
        if len(hist) >= 2:
            x = np.arange(min(3, len(hist)))
            slope_data = hist[-min(3, len(hist)):]
            row["trend_slope_3y"] = np.polyfit(x, slope_data, 1)[0] if len(slope_data) >= 2 else np.nan
        else:
            row["trend_slope_3y"] = np.nan
        
        # Institute-level mean (use last year's)
        last_year_data = group[group["year"] == group["year"].max()]
        row["inst_year_mean_rank"] = last_year_data["closing_rank"].mean() if not last_year_data.empty else np.nan
        
        # Log-transformed features
        for col in ["lag_1", "lag_2", "lag_3", "lag_1_r6", "rolling_mean_3", "rolling_mean_5", 
                     "round_cummax", "inst_year_mean_rank"]:
            val = row.get(col, np.nan)
            row[f"log_{col}"] = np.log1p(val) if not np.isnan(val) else np.nan
        
        pred_rows.append(row)
    
    if skipped_discontinued > 0 or skipped_insufficient > 0:
        print(f"  Skipped {skipped_discontinued} discontinued + {skipped_insufficient} insufficient-data groups")
    
    if not pred_rows:
        return pd.DataFrame()
    
    pred_df = pd.DataFrame(pred_rows)
    
    # Set categorical types
    for col in ["institute", "program", "quota", "category", "gender", "type"]:
        pred_df[col] = pred_df[col].astype("category")
    
    return pred_df


def get_feature_columns():
    """Return the list of feature columns used for training."""
    return [
        "institute", "program", "quota", "category", "gender", "type",
        "year_num", "round_num",
        "lag_1", "lag_2", "lag_3", "lag_1_r6",
        "yoy_delta",
        "rolling_mean_3", "rolling_std_3",
        "rolling_mean_5", "rolling_std_5",
        "round_cummax",
        "trend_slope_3y",
        "inst_year_mean_rank",
        # Log-transformed features for scale-invariant patterns
        "log_lag_1", "log_lag_2", "log_lag_3", "log_lag_1_r6",
        "log_rolling_mean_3", "log_rolling_mean_5",
        "log_round_cummax", "log_inst_year_mean_rank",
    ]


def get_categorical_feature_indices():
    """Return indices of categorical features in the feature column list."""
    feature_cols = get_feature_columns()
    cat_cols = ["institute", "program", "quota", "category", "gender", "type"]
    return [feature_cols.index(c) for c in cat_cols]
