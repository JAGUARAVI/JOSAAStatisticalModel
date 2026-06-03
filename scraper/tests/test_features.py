import numpy as np
import pandas as pd

from ml.features import engineer_features


def _sample_rank_data():
    return pd.DataFrame([
        {
            "institute": "Indian Institute of Technology Test",
            "program": "Computer Science",
            "quota": "AI",
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "opening_rank": 10,
            "closing_rank": 100,
            "year": 2023,
            "round": 1,
            "type": "IIT",
        },
        {
            "institute": "Indian Institute of Technology Test",
            "program": "Computer Science",
            "quota": "AI",
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "opening_rank": 12,
            "closing_rank": 120,
            "year": 2024,
            "round": 1,
            "type": "IIT",
        },
        {
            "institute": "Indian Institute of Technology Test",
            "program": "Computer Science",
            "quota": "AI",
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "opening_rank": 11,
            "closing_rank": 110,
            "year": 2025,
            "round": 1,
            "type": "IIT",
        },
    ])


def test_missing_seats_are_explicit_and_non_blocking():
    features = engineer_features(_sample_rank_data(), target_year=2026, seats_df=pd.DataFrame())

    assert "opening_rank" in features.columns
    assert features["seats"].eq(0).all()
    assert features["log_seats"].eq(0).all()
    assert features["seats_missing"].all()


def test_temporal_features_do_not_use_future_or_current_target():
    features = engineer_features(_sample_rank_data(), target_year=2026, seats_df=pd.DataFrame())

    observed_2025 = features[(features["year"] == 2025) & features["closing_rank"].notna()].iloc[0]
    target_2026 = features[(features["year"] == 2026) & features["closing_rank"].isna()].iloc[0]

    assert np.isclose(observed_2025["prev_year_closing"], np.log1p(120))
    assert np.isclose(target_2026["prev_year_closing"], np.log1p(110))


def test_current_seats_merge_when_available():
    seats = pd.DataFrame([{
        "year": 2026,
        "institute": "Indian Institute of Technology Test",
        "program": "Computer Science",
        "quota": "AI",
        "category": "OPEN",
        "gender": "Gender-Neutral",
        "seats": 42,
    }])

    features = engineer_features(_sample_rank_data(), target_year=2026, seats_df=seats)
    target_2026 = features[(features["year"] == 2026) & features["closing_rank"].isna()].iloc[0]
    observed = features[features["year"] < 2026]

    assert target_2026["seats"] == 42
    assert not bool(target_2026["seats_missing"])
    assert observed["seats_missing"].all()
