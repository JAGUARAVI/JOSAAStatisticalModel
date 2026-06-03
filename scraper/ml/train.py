"""PyMC hierarchical Bayesian training pipeline for JoSAA cutoff prediction.

Usage:
    python -m ml.train                 # Train and generate predictions.json
    python -m ml.train --quick          # Fast ADVI smoke-test fit
    python -m ml.train --validate       # Validate existing predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaner import load_all_data
from ml.features import engineer_features, get_model_feature_columns


CONFIDENCE_LEVEL = 0.90
RANDOM_SEED = 42


@dataclass
class BayesianFit:
    """Trained posterior and metadata needed for prediction."""

    idata: Any
    feature_cols: list[str]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    encoders: dict[str, dict[str, int]]
    fit_method: str
    validation_year: int | None = None
    validation_mae: float | None = None
    validation_rmse: float | None = None
    checkpoint_steps: int | None = None
    checkpoint_path: str | None = None
    checkpoint_count: int | None = None


def _rank_thresholds() -> list[int]:
    return sorted(set(
        list(range(100, 2000, 100)) +
        list(range(2000, 10000, 250)) +
        list(range(10000, 50000, 500)) +
        list(range(50000, 100001, 1000)) +
        list(range(100000, 200001, 5000))
    ))


def _compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        means[col] = mean
        stds[col] = std if std > 1e-9 else 1.0
    return means, stds


def _standardize(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    cols = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(means[col]).astype(float)
        cols.append(((values - means[col]) / stds[col]).to_numpy())
    return np.column_stack(cols).astype("float64")


def _training_frame(df_features: pd.DataFrame) -> pd.DataFrame:
    train_df = df_features.dropna(subset=["closing_rank", "log_closing_rank"]).copy()
    valid_rank_mask = (train_df["opening_rank"] > 0) & (train_df["closing_rank"] > 0)
    dropped_invalid = int((~valid_rank_mask).sum())
    if dropped_invalid:
        print(f"  Dropped {dropped_invalid} rows with non-positive ranks")
    train_df = train_df[valid_rank_mask].copy()
    if train_df.empty:
        raise ValueError("No training rows available; scrape OR/CR rank data before training.")
    return train_df


def _split_train_validation(
    train_df: pd.DataFrame,
    *,
    target_year: int,
    validation_year: int | None = None,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, str, int | None]:
    """Split observed rows into train/validation sets for checkpoint selection."""
    observed = train_df.copy()
    observed_years = sorted(observed["year"].dropna().astype(int).unique().tolist())
    candidate_years = [year for year in observed_years if year < target_year]

    if validation_year is None and len(candidate_years) >= 2:
        validation_year = candidate_years[-1]

    if validation_year is not None:
        val_df = observed[observed["year"] == validation_year].copy()
        train_split = observed[observed["year"] != validation_year].copy()
        if not train_split.empty and not val_df.empty:
            return train_split, val_df, f"year {validation_year}", validation_year

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(observed))
    val_size = max(1, int(round(len(observed) * validation_fraction)))
    if val_size >= len(observed):
        val_size = 1
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if len(train_idx) == 0:
        train_idx = perm[:-1]
        val_idx = perm[-1:]
    split_label = f"row-split {len(train_idx)}:{len(val_idx)}"
    return observed.iloc[train_idx].copy(), observed.iloc[val_idx].copy(), split_label, None


def _snapshot_approximation(approx: Any) -> list[np.ndarray]:
    """Capture the current variational parameters so we can restore a checkpoint."""
    return [np.array(param.get_value(), copy=True) for param in approx.params]


def _restore_approximation(approx: Any, snapshot: list[np.ndarray]) -> None:
    for param, value in zip(approx.params, snapshot):
        param.set_value(np.array(value, copy=True))


def _checkpoint_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "ml", "checkpoints")


def _validation_metrics(fit: BayesianFit, val_df: pd.DataFrame) -> dict[str, float | int]:
    if val_df.empty:
        return {"n": 0, "mae": float("inf"), "rmse": float("inf")}

    rank_samples = _predictive_rank_samples(fit, val_df, include_noise=False)
    predicted = np.median(rank_samples, axis=0)
    actual = val_df["closing_rank"].to_numpy(dtype="float64")
    errors = actual - predicted
    return {
        "n": int(len(val_df)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def _fit_checkpointed_advi(
    model: Any,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    encoders: dict[str, dict[str, int]],
    draws: int,
    advi_steps: int,
    checkpoint_steps: int,
    checkpoint_dir: str,
) -> BayesianFit:
    import pymc as pm

    os.makedirs(checkpoint_dir, exist_ok=True)
    advi = pm.ADVI(model=model, random_seed=RANDOM_SEED)
    checkpoint_steps = max(1, min(checkpoint_steps, advi_steps))
    checkpoint_eval_draws = max(50, min(250, draws // 4 if draws > 1 else 1))
    n_checkpoints = int(np.ceil(advi_steps / checkpoint_steps))
    manifest: dict[str, Any] = {
        "fit_method": "advi-checkpointed",
        "checkpoint_steps": checkpoint_steps,
        "advi_steps": advi_steps,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "candidates": [],
    }

    best_snapshot: list[np.ndarray] | None = None
    best_idata: Any | None = None
    best_metrics: dict[str, float | int] | None = None
    best_step = 0
    best_path: str | None = None
    completed = 0

    for checkpoint_idx in range(n_checkpoints):
        chunk = min(checkpoint_steps, advi_steps - completed)
        label = "Fitting" if checkpoint_idx == 0 else "Refining"
        print(f"  {label} checkpoint {checkpoint_idx + 1}/{n_checkpoints} ({chunk} steps)...")
        if checkpoint_idx == 0:
            advi.fit(
                chunk,
                progressbar=True,
                obj_optimizer=pm.adam(learning_rate=1e-3),
            )
        else:
            advi.refine(chunk, progressbar=True)
        completed += chunk

        snapshot = _snapshot_approximation(advi.approx)
        with model:
            checkpoint_idata = advi.approx.sample(draws=checkpoint_eval_draws, random_seed=RANDOM_SEED)
        checkpoint_fit = BayesianFit(
            idata=checkpoint_idata,
            feature_cols=feature_cols,
            feature_means=feature_means,
            feature_stds=feature_stds,
            encoders=encoders,
            fit_method="advi-checkpointed",
            checkpoint_steps=checkpoint_steps,
        )
        metrics = _validation_metrics(checkpoint_fit, val_df)
        checkpoint_path = os.path.join(checkpoint_dir, f"advi_step_{completed:05d}.nc")
        checkpoint_path = _save_inferencedata(checkpoint_path, checkpoint_idata)

        candidate = {
            "step": completed,
            "path": checkpoint_path,
            "validation_n": metrics["n"],
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
        }
        manifest["candidates"].append(candidate)
        print(
            f"    val MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
            f"saved {os.path.basename(checkpoint_path)}"
        )

        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_snapshot = snapshot
            best_idata = checkpoint_idata
            best_metrics = metrics
            best_step = completed
            best_path = checkpoint_path

    if best_snapshot is not None:
        _restore_approximation(advi.approx, best_snapshot)
        with model:
            final_idata = advi.approx.sample(draws=draws, random_seed=RANDOM_SEED)
    elif best_idata is not None:
        final_idata = best_idata
    else:
        with model:
            final_idata = advi.approx.sample(draws=draws, random_seed=RANDOM_SEED)

    manifest["best"] = {
        "step": best_step,
        "path": best_path,
        "validation_mae": None if best_metrics is None else best_metrics["mae"],
        "validation_rmse": None if best_metrics is None else best_metrics["rmse"],
    }
    _write_json(os.path.join(checkpoint_dir, "advi_manifest.json"), manifest)

    return BayesianFit(
        idata=final_idata,
        feature_cols=feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        encoders=encoders,
        fit_method="advi-checkpointed",
        validation_year=None,
        validation_mae=None if best_metrics is None else float(best_metrics["mae"]),
        validation_rmse=None if best_metrics is None else float(best_metrics["rmse"]),
        checkpoint_steps=checkpoint_steps,
        checkpoint_path=best_path,
        checkpoint_count=n_checkpoints,
    )


def _fit_simple_advi(
    model: Any,
    *,
    feature_cols: list[str],
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    encoders: dict[str, dict[str, int]],
    draws: int,
    advi_steps: int,
    fit_method: str,
) -> BayesianFit:
    import pymc as pm

    with model:
        print(f"  Fitting {fit_method} posterior ({advi_steps} steps)...")
        approx = pm.fit(
            n=advi_steps,
            method="advi",
            random_seed=RANDOM_SEED,
            progressbar=True,
            obj_optimizer=pm.adam(learning_rate=1e-3),
        )
        idata = approx.sample(draws=draws, random_seed=RANDOM_SEED)

    return BayesianFit(
        idata=idata,
        feature_cols=feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        encoders=encoders,
        fit_method=fit_method,
    )


def build_josaa_model(
    train_df: pd.DataFrame,
    x_train: np.ndarray,
    coords: dict[str, list[Any]],
    *,
    use_minibatch: bool = False,
    minibatch_size: int = 4096,
):
    """Build the hierarchical log-rank model."""
    import pymc as pm
    import pytensor.tensor as pt

    inst_idx = train_df["institute_id"].to_numpy(dtype="int64")
    prog_idx = train_df["program_id"].to_numpy(dtype="int64")
    quota_idx = train_df["quota_id"].to_numpy(dtype="int64")
    cat_idx = train_df["category_id"].to_numpy(dtype="int64")
    gender_idx = train_df["gender_id"].to_numpy(dtype="int64")
    tier_idx = train_df["institute_tier"].to_numpy(dtype="int64")
    round_idx = train_df["round_num"].to_numpy(dtype="int64") - 1
    year = train_df["year_index"].to_numpy(dtype="float64")
    n_obs_scaled = 1.0 - np.clip(train_df["n_observations"].to_numpy(dtype="float64"), 0, 9) / 9.0
    is_new = train_df["is_new_program"].astype(float).to_numpy()
    seats_missing = train_df["seats_missing"].astype(float).to_numpy()
    observed = train_df["log_closing_rank"].to_numpy(dtype="float64")

    if use_minibatch:
        batch_size = min(minibatch_size, len(train_df))
        x_data = pm.Minibatch(x_train, batch_size=batch_size)
        inst_idx = pm.Minibatch(inst_idx, batch_size=batch_size)
        prog_idx = pm.Minibatch(prog_idx, batch_size=batch_size)
        quota_idx = pm.Minibatch(quota_idx, batch_size=batch_size)
        cat_idx = pm.Minibatch(cat_idx, batch_size=batch_size)
        gender_idx = pm.Minibatch(gender_idx, batch_size=batch_size)
        tier_idx = pm.Minibatch(tier_idx, batch_size=batch_size)
        round_idx = pm.Minibatch(round_idx, batch_size=batch_size)
        year = pm.Minibatch(year, batch_size=batch_size)
        n_obs_scaled = pm.Minibatch(n_obs_scaled, batch_size=batch_size)
        is_new = pm.Minibatch(is_new, batch_size=batch_size)
        seats_missing = pm.Minibatch(seats_missing, batch_size=batch_size)
        observed = pm.Minibatch(observed, batch_size=batch_size)

    with pm.Model(coords=coords) as model:
        if not use_minibatch:
            x_data = pm.Data("x_data", x_train, dims=("row", "covariate"))

        global_mu = pm.Normal("global_mu", mu=8.0, sigma=1.0)
        beta_year = pm.Normal("beta_year", mu=-0.02, sigma=0.05)
        beta_cov = pm.Normal("beta_cov", mu=0.0, sigma=0.25, dims="covariate")

        tier_offset = pm.Normal("tier_offset", mu=0.0, sigma=0.5, dims="tier")
        alpha_institute = pm.Normal("alpha_institute", mu=0.0, sigma=0.55, dims="institute")
        alpha_program = pm.Normal("alpha_program", mu=0.0, sigma=0.45, dims="program")

        quota_offset = pm.Normal("quota_offset", mu=0.0, sigma=0.3, dims="quota")
        category_offset = pm.Normal("category_offset", mu=0.0, sigma=0.9, dims="category")
        gender_offset = pm.Normal("gender_offset", mu=0.0, sigma=0.2, dims="gender")

        round_effect = pm.Normal("round_effect", mu=0.0, sigma=0.08, dims="round")

        base_sigma = pm.HalfNormal("base_sigma", sigma=0.25)
        sigma_sparsity = pm.HalfNormal("sigma_sparsity", sigma=0.20)
        sigma_new_program = pm.HalfNormal("sigma_new_program", sigma=0.15)
        sigma_missing_seat = pm.HalfNormal("sigma_missing_seat", sigma=0.08)

        mu = (
            global_mu
            + beta_year * year
            + pm.math.dot(x_data, beta_cov)
            + tier_offset[tier_idx]
            + alpha_institute[inst_idx]
            + alpha_program[prog_idx]
            + quota_offset[quota_idx]
            + category_offset[cat_idx]
            + gender_offset[gender_idx]
            + round_effect[round_idx]
        )

        sigma = (
            base_sigma
            + sigma_sparsity * n_obs_scaled
            + sigma_new_program * is_new
            + sigma_missing_seat * seats_missing
        )
        sigma = pt.clip(sigma, 0.05, np.inf)

        if use_minibatch:
            pm.Normal("closing_rank_obs", mu=mu, sigma=sigma, observed=observed, total_size=len(train_df))
        else:
            pm.Normal("closing_rank_obs", mu=mu, sigma=sigma, observed=observed, dims="row")

    return model


def _coords_from_encoders(encoders: dict[str, dict[str, int]], n_obs: int, feature_cols: list[str]) -> dict[str, list[Any]]:
    def ordered(mapping: dict[str, int]) -> list[str]:
        return [item[0] for item in sorted(mapping.items(), key=lambda kv: kv[1])]

    return {
        "institute": ordered(encoders["institute"]),
        "program": ordered(encoders["program"]),
        "quota": ordered(encoders["quota"]),
        "category": ordered(encoders["category"]),
        "gender": ordered(encoders["gender"]),
        "tier": ordered(encoders["tier"]),
        "round": [1, 2, 3, 4, 5, 6],
        "covariate": feature_cols,
        "row": list(range(n_obs)),
    }


def train_model(
    df_features: pd.DataFrame,
    *,
    quick: bool = False,
    checkpoint_advi: bool = False,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    advi_steps: int = 8000,
    checkpoint_steps: int = 2000,
    validation_year: int | None = None,
    base_dir: str | None = None,
) -> BayesianFit:
    """Fit the PyMC model and return posterior artifacts."""
    import pymc as pm

    observed_df = _training_frame(df_features)
    feature_cols = get_model_feature_columns()
    encoders = df_features.attrs.get("encoders")
    if not encoders:
        raise ValueError("Feature frame is missing deterministic encoders.")

    train_df = observed_df
    val_df = pd.DataFrame()
    validation_label = None
    validation_year_used: int | None = None
    if checkpoint_advi or quick:
        train_df, val_df, validation_label, validation_year_used = _split_train_validation(
            observed_df,
            target_year=int(df_features["year"].max()),
            validation_year=validation_year,
        )

    feature_means, feature_stds = _compute_feature_stats(train_df, feature_cols)
    x_train = _standardize(train_df, feature_cols, feature_means, feature_stds)
    coords = _coords_from_encoders(encoders, len(train_df), feature_cols)

    print(f"  Training samples: {len(train_df)}")
    if validation_label is not None:
        print(f"  Validation split: {validation_label} ({len(val_df)} rows)")
    print(f"  Institutes: {len(coords['institute'])}, Programs: {len(coords['program'])}")
    print(f"  Covariates: {len(feature_cols)}")
    print("  Target: log1p(closing_rank)")

    model = build_josaa_model(
        train_df,
        x_train,
        coords,
        use_minibatch=quick or checkpoint_advi,
        minibatch_size=4096,
    )
    if quick or checkpoint_advi:
        checkpoint_root = _checkpoint_dir(base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fit = _fit_checkpointed_advi(
            model,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            feature_means=feature_means,
            feature_stds=feature_stds,
            encoders=encoders,
            draws=draws,
            advi_steps=advi_steps,
            checkpoint_steps=checkpoint_steps,
            checkpoint_dir=checkpoint_root,
        )
    else:
        print(f"  Sampling NUTS posterior ({chains} chains, {draws} draws, {tune} tune)...")
        try:
            with model:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    init="advi+adapt_diag",
                    target_accept=0.95,
                    random_seed=RANDOM_SEED,
                    return_inferencedata=True,
                )
            fit_method = "nuts"
        except Exception as exc:
            print(f"  NUTS failed: {exc}")
            print("  Falling back to minibatch ADVI so the pipeline can finish.")
            fallback_model = build_josaa_model(
                train_df,
                x_train,
                coords,
                use_minibatch=True,
                minibatch_size=4096,
            )
            fit = _fit_simple_advi(
                fallback_model,
                feature_cols=feature_cols,
                feature_means=feature_means,
                feature_stds=feature_stds,
                encoders=encoders,
                draws=draws,
                advi_steps=advi_steps,
                fit_method="advi-fallback",
            )
            validation_year_used = None

        if not isinstance(fit, BayesianFit):
            fit = BayesianFit(
                idata=idata,
                feature_cols=feature_cols,
                feature_means=feature_means,
                feature_stds=feature_stds,
                encoders=encoders,
                fit_method=fit_method,
            )

    if validation_label is not None and fit.validation_year is None:
        fit.validation_year = validation_year_used

    return fit


def _posterior_samples(idata: Any, name: str) -> np.ndarray:
    arr = idata.posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...)
    return np.asarray(arr.values)


def _predictive_rank_samples(
    fit: BayesianFit,
    pred_df: pd.DataFrame,
    *,
    include_noise: bool = True,
) -> np.ndarray:
    x_pred = _standardize(pred_df, fit.feature_cols, fit.feature_means, fit.feature_stds)

    global_mu = _posterior_samples(fit.idata, "global_mu")
    beta_year = _posterior_samples(fit.idata, "beta_year")
    beta_cov = _posterior_samples(fit.idata, "beta_cov")
    tier_offset = _posterior_samples(fit.idata, "tier_offset")
    alpha_institute = _posterior_samples(fit.idata, "alpha_institute")
    alpha_program = _posterior_samples(fit.idata, "alpha_program")
    quota_offset = _posterior_samples(fit.idata, "quota_offset")
    category_offset = _posterior_samples(fit.idata, "category_offset")
    gender_offset = _posterior_samples(fit.idata, "gender_offset")
    round_effect = _posterior_samples(fit.idata, "round_effect")

    inst_idx = pred_df["institute_id"].to_numpy(dtype="int64")
    prog_idx = pred_df["program_id"].to_numpy(dtype="int64")
    quota_idx = pred_df["quota_id"].to_numpy(dtype="int64")
    cat_idx = pred_df["category_id"].to_numpy(dtype="int64")
    gender_idx = pred_df["gender_id"].to_numpy(dtype="int64")
    tier_idx = pred_df["institute_tier"].to_numpy(dtype="int64")
    round_idx = pred_df["round_num"].to_numpy(dtype="int64") - 1
    year = pred_df["year_index"].to_numpy(dtype="float64")

    mu = (
        global_mu[:, None]
        + beta_year[:, None] * year[None, :]
        + (x_pred @ beta_cov.T).T
        + tier_offset[:, tier_idx]
        + alpha_institute[:, inst_idx]
        + alpha_program[:, prog_idx]
        + quota_offset[:, quota_idx]
        + category_offset[:, cat_idx]
        + gender_offset[:, gender_idx]
        + round_effect[:, round_idx]
    )

    if include_noise:
        base_sigma = _posterior_samples(fit.idata, "base_sigma")
        sigma_sparsity = _posterior_samples(fit.idata, "sigma_sparsity")
        sigma_new_program = _posterior_samples(fit.idata, "sigma_new_program")
        sigma_missing_seat = _posterior_samples(fit.idata, "sigma_missing_seat")
        n_obs_scaled = 1.0 - np.clip(pred_df["n_observations"].to_numpy(dtype="float64"), 0, 9) / 9.0
        is_new = pred_df["is_new_program"].astype(float).to_numpy()
        seats_missing = pred_df["seats_missing"].astype(float).to_numpy()
        sigma = (
            base_sigma[:, None]
            + sigma_sparsity[:, None] * n_obs_scaled[None, :]
            + sigma_new_program[:, None] * is_new[None, :]
            + sigma_missing_seat[:, None] * seats_missing[None, :]
        )
        rng = np.random.default_rng(RANDOM_SEED)
        mu = rng.normal(mu, sigma)

    ranks = np.expm1(mu)
    return np.clip(ranks, 1, 1_000_000)


def generate_predictions(fit: BayesianFit, df_features: pd.DataFrame, target_year: int) -> list[dict[str, Any]]:
    """Generate frontend-compatible posterior predictions."""
    pred_df = df_features[
        (df_features["year"] == target_year) &
        (df_features["closing_rank"].isna())
    ].copy()

    if pred_df.empty:
        print("  No prediction rows found for target year")
        return []

    print(f"  Generating posterior predictions for {len(pred_df)} seat groups...")
    rank_samples = _predictive_rank_samples(fit, pred_df, include_noise=True)

    med = np.median(rank_samples, axis=0)
    mean = np.mean(rank_samples, axis=0)
    std = np.std(rank_samples, axis=0)
    q_low = np.quantile(rank_samples, (1 - CONFIDENCE_LEVEL) / 2, axis=0)
    q_high = np.quantile(rank_samples, 1 - (1 - CONFIDENCE_LEVEL) / 2, axis=0)

    predictions: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(pred_df.iterrows()):
        pred = int(round(max(1, med[idx])))
        ci_low = int(round(max(1, q_low[idx])))
        ci_high = int(round(max(ci_low, q_high[idx])))
        predictions.append({
            "i": str(row["institute"]),
            "p": str(row["program"]),
            "q": str(row["quota"]),
            "c": str(row["category"]),
            "g": str(row["gender"]),
            "r": int(row["round_num"]),
            "t": str(row["type"]),
            "pred": pred,
            "ci_low": min(ci_low, pred),
            "ci_high": max(ci_high, pred),
            "mu": round(float(mean[idx] - pred), 2),
            "sigma": round(float(max(std[idx], 1.0)), 2),
        })
    return predictions


def predict_frame_for_backtest(fit: BayesianFit, df_features: pd.DataFrame, test_year: int) -> pd.DataFrame:
    """Return posterior predictions merged-ready for a historical test year."""
    pred_df = df_features[
        (df_features["year"] == test_year) &
        (df_features["closing_rank"].isna())
    ].copy()
    if pred_df.empty:
        return pd.DataFrame()

    rank_samples = _predictive_rank_samples(fit, pred_df, include_noise=True)
    pred_df["predicted"] = np.median(rank_samples, axis=0)
    pred_df["ci_low"] = np.quantile(rank_samples, (1 - CONFIDENCE_LEVEL) / 2, axis=0)
    pred_df["ci_high"] = np.quantile(rank_samples, 1 - (1 - CONFIDENCE_LEVEL) / 2, axis=0)
    return pred_df


def compute_confidence_interval(predicted_cutoff: float, residual_mu: float, residual_sigma: float, level: float = 0.90):
    """Compatibility helper used by the WLR baseline code."""
    z = stats.norm.ppf(1 - (1 - level) / 2)
    center = predicted_cutoff + residual_mu
    low = max(1, center - z * residual_sigma)
    high = center + z * residual_sigma
    return int(round(low)), int(round(high))


def validate_predictions(predictions_path: str) -> bool:
    """Validate predictions.json schema and integrity."""
    with open(predictions_path) as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    thresholds = data.get("rank_thresholds", [])
    errors: list[str] = []

    if not isinstance(preds, list):
        errors.append("predictions is not a list")
    if not isinstance(thresholds, list):
        errors.append("rank_thresholds is not a list")

    for i, p in enumerate(preds[:100]):
        for key in ["i", "p", "q", "c", "g", "r", "t", "pred", "ci_low", "ci_high", "mu", "sigma"]:
            if key not in p:
                errors.append(f"Prediction {i}: missing {key}")
        if p.get("pred", 0) < 1:
            errors.append(f"Prediction {i}: pred < 1")
        if p.get("ci_low", 0) > p.get("pred", 0):
            errors.append(f"Prediction {i}: ci_low > pred")
        if p.get("ci_high", 0) < p.get("pred", 0):
            errors.append(f"Prediction {i}: ci_high < pred")
        if not np.isfinite(p.get("sigma", np.nan)) or p.get("sigma", -1) < 0:
            errors.append(f"Prediction {i}: invalid sigma")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"Validation passed: {len(preds)} predictions, {len(thresholds)} rank thresholds")
    return True


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _save_inferencedata(path: str, idata: Any) -> str:
    """Persist an InferenceData object, falling back to pickle when needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        idata.to_netcdf(path)
        return path
    except Exception as exc:
        fallback_path = os.path.splitext(path)[0] + ".pkl"
        with open(fallback_path, "wb") as f:
            pickle.dump(idata, f)
        print(f"  NetCDF save failed; wrote pickle checkpoint instead: {exc}")
        return fallback_path


def save_artifacts(
    fit: BayesianFit,
    predictions: list[dict[str, Any]],
    *,
    base_dir: str,
    target_year: int,
) -> str:
    data_dir = os.path.join(base_dir, "..", "app", "public", "data")
    predictions_path = os.path.join(data_dir, "predictions.json")
    encoders_path = os.path.join(data_dir, "encoders.json")
    posterior_path = os.path.join(base_dir, "ml", "josaa_pymc_trace.nc")

    output = {
        "target_year": target_year,
        "model": "pymc-hierarchical-lognormal",
        "fit_method": fit.fit_method,
        "confidence_level": CONFIDENCE_LEVEL,
        "validation_year": fit.validation_year,
        "validation_mae": fit.validation_mae,
        "validation_rmse": fit.validation_rmse,
        "checkpoint_steps": fit.checkpoint_steps,
        "checkpoint_path": fit.checkpoint_path,
        "checkpoint_count": fit.checkpoint_count,
        "rank_thresholds": _rank_thresholds(),
        "predictions": predictions,
    }
    _write_json(predictions_path, output)
    _write_json(encoders_path, {
        "encoders": fit.encoders,
        "feature_cols": fit.feature_cols,
        "feature_means": fit.feature_means,
        "feature_stds": fit.feature_stds,
        "model": output["model"],
    })

    try:
        saved_path = _save_inferencedata(posterior_path, fit.idata)
        print(f"  Saved posterior trace: {saved_path}")
    except Exception as exc:
        print(f"  Could not save posterior trace: {exc}")

    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyMC hierarchical JoSAA prediction model")
    parser.add_argument("--validate", action="store_true", help="Validate existing predictions.json")
    parser.add_argument("--max-year", type=int, default=None, help="Max training year")
    parser.add_argument("--target-year", type=int, default=2026, help="Year to predict")
    parser.add_argument(
        "--validation-year",
        type=int,
        default=None,
        help="Hold out this historical year for checkpoint selection",
    )
    parser.add_argument(
        "--checkpoint-advi",
        action="store_true",
        help="Use checkpointed ADVI instead of NUTS so the best checkpoint can be selected",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=2000,
        help="ADVI steps between saved checkpoints",
    )
    parser.add_argument("--quick", action="store_true", help="Use ADVI for a fast smoke-test posterior")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws")
    parser.add_argument("--tune", type=int, default=1000, help="NUTS tuning steps")
    parser.add_argument("--chains", type=int, default=2, help="NUTS chains")
    parser.add_argument("--advi-steps", type=int, default=8000, help="ADVI optimization steps for --quick")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictions_path = os.path.join(base_dir, "..", "app", "public", "data", "predictions.json")

    if args.validate:
        validate_predictions(predictions_path)
        return

    max_year = args.max_year or (args.target_year - 1)

    print("=== PyMC Hierarchical Bayesian Training Pipeline ===")
    print(f"  Training data: <= {max_year}")
    print(f"  Target year: {args.target_year}")
    print(f"  Fit mode: {'ADVI quick' if args.quick else 'NUTS'}")

    print("\nLoading rank data...")
    df = load_all_data(max_year=max_year)
    print(f"  Loaded {len(df)} records")

    print("\nEngineering features...")
    df_features = engineer_features(df, target_year=args.target_year)
    print(f"  Feature matrix: {df_features.shape}")

    print("\nTraining model...")
    fit = train_model(
        df_features,
        quick=args.quick,
        checkpoint_advi=args.checkpoint_advi,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        advi_steps=args.advi_steps,
        checkpoint_steps=args.checkpoint_steps,
        validation_year=args.validation_year,
        base_dir=base_dir,
    )

    print(f"\nGenerating predictions for {args.target_year}...")
    predictions = generate_predictions(fit, df_features, args.target_year)
    print(f"  Generated {len(predictions)} predictions")

    print("\nSaving artifacts...")
    predictions_path = save_artifacts(fit, predictions, base_dir=base_dir, target_year=args.target_year)
    size_mb = os.path.getsize(predictions_path) / (1024 * 1024)
    print(f"  Saved predictions.json ({size_mb:.1f} MB)")

    print("\nValidating...")
    validate_predictions(predictions_path)


if __name__ == "__main__":
    main()
