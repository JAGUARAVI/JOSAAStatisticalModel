"""Optional JoSAA seat-matrix parsing and normalization."""

from __future__ import annotations

import glob
import os
import re
from io import StringIO
from typing import Any

import pandas as pd

from ml.features import IDENTITY_COLS, normalize_category, normalize_text


SEAT_CATEGORY_COLUMNS = {
    "OPEN": "OPEN",
    "OPEN-PwD": "OPEN (PwD)",
    "GEN-EWS": "EWS",
    "GEN-EWS-PwD": "EWS (PwD)",
    "OBC-NCL": "OBC-NCL",
    "OBC-NCL-PwD": "OBC-NCL (PwD)",
    "SC": "SC",
    "SC-PwD": "SC (PwD)",
    "ST": "ST",
    "ST-PwD": "ST (PwD)",
}


def _extract_year(path: str, default_year: int | None = None) -> int:
    name = os.path.basename(path)
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    if default_year is not None:
        return default_year
    raise ValueError(f"Could not infer year from {name}")


def _flatten_columns(columns: Any) -> list[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [normalize_text(c) for c in col if normalize_text(c) and "Unnamed" not in normalize_text(c)]
            flat.append(" ".join(parts))
        else:
            flat.append(normalize_text(col))
    return flat


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    normalized = {normalize_text(col).lower(): col for col in df.columns}

    def find(*candidates: str) -> str | None:
        for candidate in candidates:
            cand = candidate.lower()
            for norm, original in normalized.items():
                if cand == norm or cand in norm:
                    return original
        return None

    mapping = {
        "institute": find("institute name", "institute"),
        "program": find("program name", "branch name", "program"),
        "quota": find("quota", "state", "seat type"),
        "gender": find("seat pool", "gender"),
    }
    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise ValueError(f"Seat matrix missing expected columns: {missing}")
    return mapping  # type: ignore[return-value]


def parse_seat_matrix_html(path: str, default_year: int | None = None) -> pd.DataFrame:
    """Parse one saved JoSAA seat-matrix HTML file into long normalized rows."""
    with open(path, "r") as f:
        html = f.read()

    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame(columns=IDENTITY_COLS + ["year", "seats"])

    year = _extract_year(path, default_year)
    df = max(tables, key=len).copy()
    df.columns = _flatten_columns(df.columns)
    lookup = _column_lookup(df)

    rows = []
    for _, row in df.iterrows():
        inst = normalize_text(row[lookup["institute"]])
        prog = normalize_text(row[lookup["program"]])
        quota = normalize_text(row[lookup["quota"]])
        gender = normalize_text(row[lookup["gender"]])

        for source_col, category in SEAT_CATEGORY_COLUMNS.items():
            matching_cols = [
                col for col in df.columns
                if normalize_text(col).replace(" ", "") == source_col.replace(" ", "")
            ]
            if not matching_cols:
                continue

            seats = pd.to_numeric(row[matching_cols[0]], errors="coerce")
            if pd.isna(seats):
                continue
            rows.append({
                "year": year,
                "institute": inst,
                "program": prog,
                "quota": quota,
                "category": normalize_category(category),
                "gender": gender,
                "seats": int(max(0, seats)),
            })

    if not rows:
        return pd.DataFrame(columns=IDENTITY_COLS + ["year", "seats"])

    return pd.DataFrame(rows).drop_duplicates(subset=IDENTITY_COLS + ["year"], keep="last")


def normalize_saved_seat_html(seat_dir: str | None = None) -> pd.DataFrame:
    """Parse all saved seat HTML files and write normalized CSV files."""
    if seat_dir is None:
        seat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seat_data")

    html_files = sorted(glob.glob(os.path.join(seat_dir, "*.html")))
    frames = []
    for html_path in html_files:
        try:
            parsed = parse_seat_matrix_html(html_path)
        except Exception as exc:
            print(f"Skipping {os.path.basename(html_path)}: {exc}")
            continue
        if parsed.empty:
            continue
        csv_path = html_path.replace(".html", ".csv")
        parsed.to_csv(csv_path, index=False)
        print(f"Saved {os.path.basename(csv_path)} ({len(parsed)} rows)")
        frames.append(parsed)

    if not frames:
        return pd.DataFrame(columns=IDENTITY_COLS + ["year", "seats"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    normalize_saved_seat_html()
