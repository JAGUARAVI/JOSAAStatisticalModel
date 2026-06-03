"""Quick test of ML pipeline components."""
import sys
import os
import pandas as pd
import numpy as np

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaner import load_all_data
from ml.features import engineer_features, get_feature_columns
from ml.baseline import weighted_linear_regression

print("▶ Loading subset of data (2016-2022)...")
df = load_all_data(max_year=2022)
print(f"  Loaded {len(df)} records")

if not df.empty:
    print("\n▶ Testing feature engineering...")
    df_feat = engineer_features(df.head(1000), target_year=2023)
    print(f"  Features shape: {df_feat.shape}")
    print(f"  Columns: {df_feat.columns.tolist()[:10]}...")

    print("\n▶ Testing WLR baseline...")
    test_data = [(2020.0, 1000.0), (2021.0, 1100.0), (2022.0, 1250.0)]
    m, b = weighted_linear_regression(test_data)
    print(f"  WLR Slope: {m:.2f}, Intercept: {b:.2f}")

    print("\n▶ Checking for CatBoost...")
    try:
        from catboost import CatBoostRegressor
        print("  ✅ CatBoost is installed")
    except ImportError:
        print("  ❌ CatBoost NOT found")

print("\n✅ Basic checks passed!")
