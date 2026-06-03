"""Feature engineering for the hierarchical Bayesian JoSAA model."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


PREFERRED_CATEGORY_ORDER = [
    "OPEN",
    "OPEN (PwD)",
    "EWS",
    "EWS (PwD)",
    "OBC-NCL",
    "OBC-NCL (PwD)",
    "SC",
    "SC (PwD)",
    "ST",
    "ST (PwD)",
]

TIER_ENCODER = {"IIT": 0, "NIT": 1, "IIIT": 2, "GFTI": 3}

IDENTITY_COLS = ["institute", "program", "quota", "category", "gender"]
GROUP_ROUND_COLS = IDENTITY_COLS + ["round"]


def normalize_text(value: Any) -> str:
    """Normalize JoSAA labels without changing their user-facing wording."""
    return " ".join(str(value).replace("\xa0", " ").split())


def normalize_category(value: Any) -> str:
    """Normalize seat matrix category labels to the OR/CR labels used by the app."""
    label = normalize_text(value)
    replacements = {
        "GEN-EWS": "EWS",
        "GEN-EWS-PwD": "EWS (PwD)",
        "OPEN-PwD": "OPEN (PwD)",
        "OPEN - PwD": "OPEN (PwD)",
        "OBC-NCL-PwD": "OBC-NCL (PwD)",
        "OBC-NCL - PwD": "OBC-NCL (PwD)",
        "SC-PwD": "SC (PwD)",
        "SC - PwD": "SC (PwD)",
        "ST-PwD": "ST (PwD)",
        "ST - PwD": "ST (PwD)",
    }
    return replacements.get(label, label)


def _ordered_unique(values: pd.Series, preferred: list[str] | None = None) -> list[str]:
    present = sorted({normalize_text(v) for v in values.dropna().tolist()})
    if not preferred:
        return present
    ordered = [v for v in preferred if v in present]
    ordered.extend(v for v in present if v not in ordered)
    return ordered


def _target_year_seat_branches(
    seats_df: pd.DataFrame | None,
    target_year: int,
) -> set[tuple[str, str]] | None:
    """Return branch pairs that exist in the target-year seat matrix.

    When seat data is available for the requested target year, we use it as the
    prediction whitelist so the model only exports branches that actually exist
    in the new seat matrix. If no target-year seat rows are present, return
    ``None`` and keep the historical prediction universe unchanged.
    """
    if seats_df is None or seats_df.empty:
        return None
    required = {"year", "institute", "program"}
    if not required.issubset(seats_df.columns):
        return None

    years = pd.to_numeric(seats_df["year"], errors="coerce")
    target_rows = seats_df.loc[years == target_year, ["institute", "program"]].dropna()
    if target_rows.empty:
        return None

    return {
        (normalize_text(inst), normalize_text(prog))
        for inst, prog in target_rows.itertuples(index=False, name=None)
    }


def build_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build deterministic encoders saved with the model for inference/export."""
    return {
        "institute": {v: i for i, v in enumerate(_ordered_unique(df["institute"]))},
        "program": {v: i for i, v in enumerate(_ordered_unique(df["program"]))},
        "quota": {v: i for i, v in enumerate(_ordered_unique(df["quota"]))},
        "category": {
            v: i for i, v in enumerate(_ordered_unique(df["category"], PREFERRED_CATEGORY_ORDER))
        },
        "gender": {v: i for i, v in enumerate(_ordered_unique(df["gender"]))},
        "tier": dict(TIER_ENCODER),
    }


def _load_optional_seats(base_dir: str | None = None) -> pd.DataFrame:
    """Load normalized seat CSV files when available; otherwise return an empty frame."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seat_dir = os.path.join(base_dir, "seat_data")
    if not os.path.isdir(seat_dir):
        return pd.DataFrame()

    csv_files = sorted(
        os.path.join(seat_dir, name)
        for name in os.listdir(seat_dir)
        if name.endswith(".csv")
    )
    frames = []
    for path in csv_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  Skipping unreadable seat data {os.path.basename(path)}: {exc}")

    if not frames:
        return pd.DataFrame()

    seats = pd.concat(frames, ignore_index=True)
    required = set(IDENTITY_COLS + ["year", "seats"])
    if not required.issubset(seats.columns):
        missing = sorted(required - set(seats.columns))
        print(f"  Seat data ignored; missing columns: {missing}")
        return pd.DataFrame()

    for col in IDENTITY_COLS:
        seats[col] = seats[col].map(normalize_text)
    seats["category"] = seats["category"].map(normalize_category)
    seats["year"] = pd.to_numeric(seats["year"], errors="coerce")
    seats["seats"] = pd.to_numeric(seats["seats"], errors="coerce")
    seats = seats.dropna(subset=["year", "seats"])
    seats["year"] = seats["year"].astype(int)
    seats["seats"] = seats["seats"].clip(lower=0).astype(int)
    return seats[IDENTITY_COLS + ["year", "seats"]].drop_duplicates(
        subset=IDENTITY_COLS + ["year"], keep="last"
    )


def _append_prediction_rows(
    df: pd.DataFrame,
    target_year: int,
    allowed_branches: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Append target-year rows for recently active seat identities and rounds."""
    pred_rows: list[dict[str, Any]] = []
    skipped_discontinued = 0
    skipped_insufficient = 0
    skipped_not_in_seats = 0

    for key, group in df.groupby(GROUP_ROUND_COLS, observed=True):
        group = group.sort_values("year")
        n_years = group["year"].nunique()
        max_data_year = int(group["year"].max())

        if max_data_year < target_year - 2:
            skipped_discontinued += 1
            continue
        if n_years < 2:
            skipped_insufficient += 1
            continue

        inst, prog, quota, category, gender, round_num = key
        if allowed_branches is not None and (inst, prog) not in allowed_branches:
            skipped_not_in_seats += 1
            continue
        last = group.iloc[-1]
        pred_rows.append({
            "institute": inst,
            "program": prog,
            "quota": quota,
            "category": category,
            "gender": gender,
            "round": int(round_num),
            "type": str(last["type"]),
            "opening_rank": float(last["opening_rank"]),
            "closing_rank": np.nan,
            "year": target_year,
        })

    if skipped_discontinued or skipped_insufficient:
        print(
            f"  Skipped {skipped_discontinued} discontinued + "
            f"{skipped_insufficient} insufficient-data groups"
        )
    if skipped_not_in_seats:
        print(f"  Skipped {skipped_not_in_seats} groups not present in target-year seat data")

    if not pred_rows:
        return df
    return pd.concat([df, pd.DataFrame(pred_rows)], ignore_index=True)


def _merge_seats(df: pd.DataFrame, seats_df: pd.DataFrame | None) -> pd.DataFrame:
    if seats_df is None:
        seats_df = _load_optional_seats()

    if seats_df.empty:
        df["seats"] = 0
        df["seats_missing"] = True
        df["log_seats"] = 0.0
        return df

    seats = seats_df.copy()
    for col in IDENTITY_COLS:
        seats[col] = seats[col].map(normalize_text)
    seats["category"] = seats["category"].map(normalize_category)

    merged = pd.merge(df, seats, on=IDENTITY_COLS + ["year"], how="left")
    merged["seats_missing"] = merged["seats"].isna()
    merged["seats"] = merged["seats"].fillna(0).clip(lower=0).astype(int)
    merged["log_seats"] = np.log1p(merged["seats"])
    return merged


def _encode(df: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    df["institute_id"] = df["institute"].map(encoders["institute"]).astype(int)
    df["program_id"] = df["program"].map(encoders["program"]).astype(int)
    df["quota_id"] = df["quota"].map(encoders["quota"]).astype(int)
    df["category_id"] = df["category"].map(encoders["category"]).astype(int)
    df["gender_id"] = df["gender"].map(encoders["gender"]).astype(int)
    df["institute_tier"] = df["type"].map(encoders["tier"]).fillna(TIER_ENCODER["GFTI"]).astype(int)
    return df


def engineer_features(
    df: pd.DataFrame,
    target_year: int | None = None,
    seats_df: pd.DataFrame | None = None,
    encoders: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Engineer leakage-safe features for training and prediction.

    The frontend still consumes compact ranks.json separately; this frame is for
    model training/export and intentionally keeps human-readable labels.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    for col in ["institute", "program", "quota", "category", "gender", "type"]:
        df[col] = df[col].map(normalize_text)
    df["category"] = df["category"].map(normalize_category)

    df["opening_rank"] = pd.to_numeric(df.get("opening_rank", df["closing_rank"]), errors="coerce")
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype(int)

    if target_year is not None:
        allowed_branches = _target_year_seat_branches(seats_df, target_year)
        df = _append_prediction_rows(df, target_year, allowed_branches)

    df = _merge_seats(df, seats_df)
    min_year = int(df["year"].min())
    df["year_index"] = df["year"] - min_year
    df["round_num"] = df["round"]
    df["is_last_round"] = df["round"] == df.groupby(["year"] + IDENTITY_COLS)["round"].transform("max")

    df["log_opening_rank"] = np.log1p(df["opening_rank"].clip(lower=1))
    df["log_closing_rank"] = np.log1p(df["closing_rank"].clip(lower=1))

    df = df.sort_values(GROUP_ROUND_COLS + ["year"]).reset_index(drop=True)
    grp = df.groupby(GROUP_ROUND_COLS, observed=True)
    log_close = grp["log_closing_rank"]

    df["prev_year_closing"] = log_close.shift(1)
    df["rolling_mean_3yr"] = log_close.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["rolling_std_3yr"] = log_close.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=2).std()
    )
    df["yoy_change"] = log_close.diff()
    df["n_observations"] = grp.cumcount()
    df["is_new_program"] = df["n_observations"] < 4
    df["data_quality"] = pd.cut(
        df["n_observations"],
        bins=[-1, 3, 6, 1000],
        labels=["sparse", "medium", "rich"],
    ).astype(str)

    # Prediction and very early rows use the freshest available historical rank.
    df["prev_year_closing"] = df["prev_year_closing"].fillna(df["rolling_mean_3yr"])
    df["rolling_mean_3yr"] = df["rolling_mean_3yr"].fillna(df["prev_year_closing"])
    fallback_log = np.log1p(df["closing_rank"].fillna(df["opening_rank"]).clip(lower=1))
    df["prev_year_closing"] = df["prev_year_closing"].fillna(fallback_log)
    df["rolling_mean_3yr"] = df["rolling_mean_3yr"].fillna(df["prev_year_closing"])
    df["rolling_std_3yr"] = df["rolling_std_3yr"].fillna(0.15)
    df["yoy_change"] = df["yoy_change"].fillna(0.0)

    if encoders is None:
        encoders = build_encoders(df)
    df = _encode(df, encoders)
    df.attrs["encoders"] = encoders
    df.attrs["min_year"] = min_year
    return df


def get_model_feature_columns() -> list[str]:
    """Return numeric covariates used by the PyMC linear predictor."""
    return [
        "year_index",
        "log_opening_rank",
        "prev_year_closing",
        "rolling_mean_3yr",
        "rolling_std_3yr",
        "yoy_change",
        "n_observations",
        "log_seats",
        "seats_missing",
        "is_new_program",
    ]


def get_feature_columns() -> list[str]:
    """Compatibility alias for older imports."""
    return get_model_feature_columns()


def get_categorical_feature_indices() -> list[int]:
    """Compatibility alias; PyMC consumes encoded integer columns directly."""
    return []
