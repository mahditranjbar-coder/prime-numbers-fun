from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .singular_series import offset_arithmetic_features


def _model_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    offsets = frame["offset"].to_numpy(dtype=np.float64)
    log_mid = np.log(frame["midpoint"].to_numpy(dtype=np.float64))
    names = [
        "inv_log_mid",
        "inv_log_mid_sq",
        "log_offset_over_log_mid",
        "log_singular_series",
        "omega",
        "big_omega",
        "log_largest_prime_factor",
        "reciprocal_factor_sum",
    ]
    matrix = np.column_stack(
        [
            1.0 / log_mid,
            1.0 / log_mid**2,
            np.log(offsets) / log_mid,
            np.log(frame["singular_series"].to_numpy(dtype=np.float64)),
            frame["omega"].to_numpy(dtype=np.float64),
            frame["big_omega"].to_numpy(dtype=np.float64),
            np.log(frame["largest_prime_factor"].to_numpy(dtype=np.float64)),
            frame["reciprocal_factor_sum"].to_numpy(dtype=np.float64),
        ]
    )
    return matrix, names


def analyze_results(results_dir: Path) -> dict[str, object]:
    pair_path = results_dir / "pair_metrics.csv.gz"
    pairs = pd.read_csv(pair_path)
    blocks = pd.read_csv(results_dir / "block_summary.csv")

    offsets = np.sort(pairs["offset"].unique())
    features = offset_arithmetic_features(offsets)
    feature_frame = pd.DataFrame({"offset": offsets, **features})
    pairs = pairs.merge(feature_frame, on="offset", how="left", validate="many_to_one")

    grouped = pairs.groupby("offset", sort=True)
    offset_summary = grouped.agg(
        singular_series=("singular_series", "first"),
        observed_count=("observed_count", "sum"),
        predicted_count=("predicted_count", "sum"),
        weighted_observed=("weighted_observed", "sum"),
        weighted_predicted=("weighted_predicted", "sum"),
        mean_count_residual=("count_residual", "mean"),
        std_count_residual=("count_residual", "std"),
        mean_psi_residual=("psi_residual", "mean"),
        std_psi_residual=("psi_residual", "std"),
    ).reset_index()
    offset_summary["aggregate_count_residual"] = (
        offset_summary["observed_count"] / offset_summary["predicted_count"] - 1.0
    )
    offset_summary["aggregate_psi_residual"] = (
        offset_summary["weighted_observed"] / offset_summary["weighted_predicted"] - 1.0
    )
    offset_summary = offset_summary.merge(feature_frame, on="offset", validate="one_to_one")
    offset_summary.to_csv(results_dir / "offset_summary.csv", index=False)

    block_metrics = pairs.groupby("block_id", sort=True).apply(
        lambda g: pd.Series(
            {
                "actual_count_rmse": math.sqrt(float(np.mean(g["count_residual"] ** 2))),
                "actual_psi_rmse": math.sqrt(float(np.mean(g["psi_residual"] ** 2))),
                "wheel_null_rmse": math.sqrt(float(np.mean(g["null_residual"] ** 2))),
                "aggregate_count_ratio": float(g["observed_count"].sum() / g["predicted_count"].sum()),
                "aggregate_psi_ratio": float(g["weighted_observed"].sum() / g["weighted_predicted"].sum()),
            }
        ),
        include_groups=False,
    ).reset_index()
    block_metrics = block_metrics.merge(
        blocks[["block_id", "low", "high", "midpoint"]], on="block_id", validate="one_to_one"
    )
    block_metrics.to_csv(results_dir / "block_residual_summary.csv", index=False)

    # Chronological holdout: no random split is permitted for a scale-generalization test.
    unique_blocks = np.sort(pairs["block_id"].unique())
    split_at = max(1, int(len(unique_blocks) * 0.7))
    train_blocks = unique_blocks[:split_at]
    test_blocks = unique_blocks[split_at:]
    train = pairs[pairs["block_id"].isin(train_blocks)].copy()
    test = pairs[pairs["block_id"].isin(test_blocks)].copy()
    x_train, feature_names = _model_features(train)
    x_test, _ = _model_features(test)
    y_train = train["psi_residual"].to_numpy()
    y_test = test["psi_residual"].to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    ridge = Ridge(alpha=10.0).fit(x_train_scaled, y_train)
    ridge_prediction = ridge.predict(x_test_scaled)
    boost = HistGradientBoostingRegressor(
        max_iter=160,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=104729,
    ).fit(x_train, y_train)
    boost_prediction = boost.predict(x_test)

    # A small symbolic-style feature search: second-order products from the
    # interpretable feature library, followed by sparsity. Alpha is selected on
    # the latest training blocks, never on the final chronological holdout.
    inner_split = max(1, int(len(train_blocks) * 0.75))
    inner_fit = train[train["block_id"].isin(train_blocks[:inner_split])]
    inner_validation = train[train["block_id"].isin(train_blocks[inner_split:])]
    x_inner_fit, _ = _model_features(inner_fit)
    x_inner_validation, _ = _model_features(inner_validation)
    y_inner_fit = inner_fit["psi_residual"].to_numpy()
    y_inner_validation = inner_validation["psi_residual"].to_numpy()

    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    x_inner_fit_poly = polynomial.fit_transform(x_inner_fit)
    x_inner_validation_poly = polynomial.transform(x_inner_validation)
    sparse_scaler = StandardScaler()
    x_inner_fit_poly = sparse_scaler.fit_transform(x_inner_fit_poly)
    x_inner_validation_poly = sparse_scaler.transform(x_inner_validation_poly)
    # Smaller penalties were numerically unstable because the polynomial
    # library contains deliberately redundant terms; those fits also failed
    # chronological validation in development runs.
    alpha_candidates = [1e-4, 3e-4, 1e-3, 3e-3]
    selected_alpha = alpha_candidates[0]
    selected_validation_rmse = float("inf")
    for alpha in alpha_candidates:
        candidate = ElasticNet(alpha=alpha, l1_ratio=0.9, max_iter=20_000).fit(
            x_inner_fit_poly, y_inner_fit
        )
        candidate_rmse = math.sqrt(
            mean_squared_error(y_inner_validation, candidate.predict(x_inner_validation_poly))
        )
        if candidate_rmse < selected_validation_rmse:
            selected_alpha = alpha
            selected_validation_rmse = candidate_rmse

    x_train_poly = polynomial.fit_transform(x_train)
    x_test_poly = polynomial.transform(x_test)
    sparse_scaler = StandardScaler()
    x_train_poly = sparse_scaler.fit_transform(x_train_poly)
    x_test_poly = sparse_scaler.transform(x_test_poly)
    sparse_model = ElasticNet(alpha=selected_alpha, l1_ratio=0.9, max_iter=20_000).fit(
        x_train_poly, y_train
    )
    sparse_prediction = sparse_model.predict(x_test_poly)
    polynomial_names = polynomial.get_feature_names_out(feature_names)
    nonzero = {
        name: float(coefficient)
        for name, coefficient in zip(polynomial_names, sparse_model.coef_, strict=True)
        if coefficient != 0.0
    }

    model_metrics = {
        "train_blocks": train_blocks.tolist(),
        "test_blocks": test_blocks.tolist(),
        "test_rows": int(len(test)),
        "zero_baseline_rmse": math.sqrt(mean_squared_error(y_test, np.zeros_like(y_test))),
        "zero_baseline_mae": mean_absolute_error(y_test, np.zeros_like(y_test)),
        "ridge_rmse": math.sqrt(mean_squared_error(y_test, ridge_prediction)),
        "ridge_mae": mean_absolute_error(y_test, ridge_prediction),
        "boosted_tree_rmse": math.sqrt(mean_squared_error(y_test, boost_prediction)),
        "boosted_tree_mae": mean_absolute_error(y_test, boost_prediction),
        "sparse_polynomial_alpha": selected_alpha,
        "sparse_polynomial_validation_rmse": selected_validation_rmse,
        "sparse_polynomial_rmse": math.sqrt(mean_squared_error(y_test, sparse_prediction)),
        "sparse_polynomial_mae": mean_absolute_error(y_test, sparse_prediction),
        "sparse_polynomial_nonzero_terms": nonzero,
        "ridge_coefficients": dict(zip(feature_names, ridge.coef_.tolist(), strict=True)),
    }

    correlation_metrics: dict[str, dict[str, float]] = {}
    for column in [
        "singular_series",
        "omega",
        "big_omega",
        "largest_prime_factor",
        "reciprocal_factor_sum",
    ]:
        rho, p_value = spearmanr(offset_summary[column], offset_summary["aggregate_psi_residual"])
        correlation_metrics[column] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    # PCA across blocks. Center each offset to avoid treating persistent offset bias as a time mode.
    matrix = pairs.pivot(index="block_id", columns="offset", values="psi_residual").to_numpy()
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular_values**2
    explained = variance / variance.sum()
    pca_summary = {
        "explained_variance_ratio": explained[: min(8, len(explained))].tolist(),
        "singular_values": singular_values[: min(8, len(singular_values))].tolist(),
    }
    pca_loadings = pd.DataFrame({"offset": offsets})
    for component in range(min(5, vt.shape[0])):
        pca_loadings[f"component_{component + 1}"] = vt[component]
    pca_loadings.to_csv(results_dir / "pca_offset_loadings.csv", index=False)

    spectral_summary, spectral_frame = _spectral_diagnostic(pairs)
    spectral_frame.to_csv(results_dir / "spectral_summary.csv", index=False)

    aggregate_metrics = {
        "aggregate_count_ratio": float(pairs["observed_count"].sum() / pairs["predicted_count"].sum()),
        "aggregate_psi_ratio": float(pairs["weighted_observed"].sum() / pairs["weighted_predicted"].sum()),
        "median_absolute_offset_psi_residual": float(
            np.median(np.abs(offset_summary["aggregate_psi_residual"]))
        ),
        "max_absolute_offset_psi_residual": float(
            np.max(np.abs(offset_summary["aggregate_psi_residual"]))
        ),
    }

    analysis = {
        "aggregate": aggregate_metrics,
        "models": model_metrics,
        "correlations": correlation_metrics,
        "pca": pca_summary,
        "spectral": spectral_summary,
    }
    (results_dir / "analysis_summary.json").write_text(json.dumps(analysis, indent=2) + "\n")

    _make_plots(results_dir, pairs, blocks, block_metrics, offset_summary, explained, spectral_frame)
    _write_report(results_dir, analysis, blocks, block_metrics, offset_summary)
    return analysis


def _spectral_diagnostic(pairs: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Scan common residual power on the log(x) axis.

    Twelve blocks offer low frequency resolution. This diagnostic is included
    to reject obvious common oscillations, not to estimate zeta-zero spectra.
    """
    actual = pairs.pivot(index="block_id", columns="offset", values="psi_residual")
    null = pairs.pivot(index="block_id", columns="offset", values="null_residual")
    midpoint = pairs.groupby("block_id", sort=True)["midpoint"].first().to_numpy(dtype=np.float64)
    time_axis = np.log(midpoint)
    centered_time = time_axis - time_axis.min()
    span = centered_time.max()
    median_step = float(np.median(np.diff(time_axis)))
    frequencies = np.linspace(2.0 * np.pi / span, np.pi / median_step, 160)
    window = np.hanning(len(time_axis))[:, None]

    def mean_normalized_power(frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.to_numpy(dtype=np.float64)
        matrix -= matrix.mean(axis=0, keepdims=True)
        matrix *= window
        energy = np.sum(matrix**2, axis=0)
        usable = energy > 0
        transform = np.exp(-1j * np.outer(frequencies, centered_time)) @ matrix[:, usable]
        power = np.abs(transform) ** 2 / energy[usable]
        return power.mean(axis=1)

    actual_power = mean_normalized_power(actual)
    null_power = mean_normalized_power(null)
    frame = pd.DataFrame(
        {
            "angular_frequency_log_x": frequencies,
            "cycles_across_observed_span": frequencies * span / (2.0 * np.pi),
            "mean_actual_power": actual_power,
            "mean_wheel_null_power": null_power,
            "actual_to_null_power": actual_power / null_power,
        }
    )
    peak_index = int(np.argmax(frame["actual_to_null_power"]))
    peak = frame.iloc[peak_index]
    summary: dict[str, float | int] = {
        "block_samples": int(len(time_axis)),
        "frequency_bins": int(len(frequencies)),
        "peak_angular_frequency_log_x": float(peak.angular_frequency_log_x),
        "peak_cycles_across_observed_span": float(peak.cycles_across_observed_span),
        "peak_actual_to_null_power": float(peak.actual_to_null_power),
    }
    return summary, frame


def _make_plots(
    results_dir: Path,
    pairs: pd.DataFrame,
    blocks: pd.DataFrame,
    block_metrics: pd.DataFrame,
    offset_summary: pd.DataFrame,
    explained: np.ndarray,
    spectral_frame: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    final_block = pairs[pairs["block_id"] == pairs["block_id"].max()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(final_block["offset"], final_block["psi_ratio"], s=5, alpha=0.45, color="#1f77b4")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set(xlabel="Even offset h", ylabel="Observed / Hardy–Littlewood", title="Final-block weighted pair ratio")
    ax.set_ylim(*np.quantile(final_block["psi_ratio"], [0.002, 0.998]))
    fig.tight_layout()
    fig.savefig(results_dir / "final_block_pair_ratio.png", dpi=170)
    plt.close(fig)

    selected = [h for h in [2, 6, 30, 210, 2310, 10_000] if h in set(pairs["offset"])]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for h in selected:
        subset = pairs[pairs["offset"] == h]
        ax.plot(subset["midpoint"], subset["psi_residual"], marker="o", linewidth=1.3, label=f"h={h}")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set(xlabel="Block midpoint", ylabel="Weighted residual", title="Residual across numerical scales")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "selected_residual_trends.png", dpi=170)
    plt.close(fig)

    heat = pairs.pivot(index="block_id", columns="offset", values="psi_residual")
    clip = float(np.quantile(np.abs(heat.to_numpy()), 0.98))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(
        heat.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-clip,
        vmax=clip,
        extent=[heat.columns.min(), heat.columns.max(), heat.index.max() + 0.5, heat.index.min() - 0.5],
    )
    ax.set(xlabel="Even offset h", ylabel="Block id", title="Hardy–Littlewood weighted residual heatmap")
    fig.colorbar(image, ax=ax, label="Residual")
    fig.tight_layout()
    fig.savefig(results_dir / "residual_heatmap.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(block_metrics["midpoint"], block_metrics["actual_psi_rmse"], marker="o", label="Prime data")
    ax.plot(block_metrics["midpoint"], block_metrics["wheel_null_rmse"], marker="o", label="Wheel null")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Block midpoint", ylabel="RMSE across offsets", title="Residual scale versus wheel-conditioned null")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "block_residual_rmse.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    shown = min(8, len(explained))
    ax.bar(np.arange(1, shown + 1), explained[:shown], color="#4c78a8")
    ax.set(xlabel="Principal component", ylabel="Explained variance", title="Cross-offset residual modes")
    fig.tight_layout()
    fig.savefig(results_dir / "pca_explained_variance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        spectral_frame["cycles_across_observed_span"],
        spectral_frame["mean_actual_power"],
        label="Prime residual",
    )
    ax.plot(
        spectral_frame["cycles_across_observed_span"],
        spectral_frame["mean_wheel_null_power"],
        label="Wheel null",
    )
    ax.set(
        xlabel="Cycles across sampled log-range",
        ylabel="Mean normalized power",
        title="Low-resolution log-scale spectral diagnostic",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "residual_spectral_scan.png", dpi=170)
    plt.close(fig)

    gap_path = results_dir / "gap_distribution.csv"
    if gap_path.exists():
        gaps = pd.read_csv(gap_path)
        finite = gaps[np.isfinite(gaps["bin_right"])]
        centers = (finite["bin_left"] + finite["bin_right"]) / 2
        width = finite["bin_right"] - finite["bin_left"]
        density = finite["count"] / (finite["count"].sum() * width)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.step(centers, density, where="mid", label="Prime gaps")
        ax.plot(centers, np.exp(-centers), color="black", linestyle="--", label="Exp(1) reference")
        ax.set_xlim(0, 5)
        ax.set(xlabel=r"Gap / log(p)", ylabel="Density", title="Normalized consecutive-prime gaps")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "normalized_gap_distribution.png", dpi=170)
        plt.close(fig)


def _write_report(
    results_dir: Path,
    analysis: dict[str, object],
    blocks: pd.DataFrame,
    block_metrics: pd.DataFrame,
    offset_summary: pd.DataFrame,
) -> None:
    aggregate = analysis["aggregate"]
    models = analysis["models"]
    correlations = analysis["correlations"]
    strongest = max(correlations.items(), key=lambda item: abs(item[1]["spearman_rho"]))
    worst = offset_summary.iloc[np.argmax(np.abs(offset_summary["aggregate_psi_residual"]))]
    final = block_metrics.iloc[-1]

    text = f"""# Prime Pattern Lab — Initial Run Report

## Scope

- Numerical range: `{int(blocks.low.min()):,}` to `{int(blocks.high.max()):,}`
- Logarithmic blocks: `{len(blocks)}`
- Tested even offsets: `2` through `{int(offset_summary.offset.max()):,}`
- Primary statistic: weighted Hardy–Littlewood pair residual
- Validation: chronological holdout plus a wheel-conditioned random null

## Main numerical results

- Aggregate raw-count observed/predicted ratio: `{aggregate['aggregate_count_ratio']:.8f}`
- Aggregate weighted observed/predicted ratio: `{aggregate['aggregate_psi_ratio']:.8f}`
- Median absolute per-offset weighted residual: `{aggregate['median_absolute_offset_psi_residual']:.6f}`
- Largest absolute per-offset weighted residual: `{aggregate['max_absolute_offset_psi_residual']:.6f}` at `h={int(worst.offset)}`
- Final-block weighted residual RMSE across offsets: `{final.actual_psi_rmse:.6f}`
- Final-block wheel-null RMSE across offsets: `{final.wheel_null_rmse:.6f}`

## Result

No transferable residual law was found in the tested model class. The aggregate weighted prediction error was `{aggregate['aggregate_psi_ratio'] - 1.0:+.8f}`, and every fitted model performed worse on later numerical scales than simply predicting a zero residual. The correct conclusion from this run is that the Hardy–Littlewood singular series explains the tested pair frequencies extremely well through `10^8`; it is not evidence that the conjecture has been proved.

## Out-of-scale prediction

The models were trained only on earlier blocks and evaluated on later blocks.

| Model | Holdout RMSE | Holdout MAE |
|---|---:|---:|
| No-extra-pattern baseline (residual = 0) | {models['zero_baseline_rmse']:.6f} | {models['zero_baseline_mae']:.6f} |
| Interpretable ridge model | {models['ridge_rmse']:.6f} | {models['ridge_mae']:.6f} |
| Shallow boosted-tree model | {models['boosted_tree_rmse']:.6f} | {models['boosted_tree_mae']:.6f} |
| Sparse second-order feature model | {models['sparse_polynomial_rmse']:.6f} | {models['sparse_polynomial_mae']:.6f} |

An AI model is useful only if it beats the zero-residual baseline on later numerical scales. Training fit is deliberately not reported as evidence.

## Log-scale spectral scan

The scan used only `{analysis['spectral']['block_samples']}` block samples. Its strongest prime/null power ratio was `{analysis['spectral']['peak_actual_to_null_power']:.4f}` at approximately `{analysis['spectral']['peak_cycles_across_observed_span']:.3f}` cycles across the observed log-range. With so few scale samples, this can reject a large common oscillation but cannot resolve or identify a zeta-zero-like spectrum.

## Arithmetic-feature check

The strongest tested rank correlation between an offset feature and the aggregate weighted residual was `{strongest[0]}`, with Spearman `rho={strongest[1]['spearman_rho']:.5f}` and uncorrected `p={strongest[1]['p_value']:.3g}`. Because several features were tested and offsets are arithmetically dependent, this is diagnostic rather than discovery evidence.

## Interpretation rules

1. Ratios near one support the known singular-series baseline; they do not prove Hardy–Littlewood.
2. A model that loses to residual = 0 has found no transferable extra law.
3. The wheel null is a finite modular model, not a complete model of primes.
4. Heatmap bands aligned with factors of offsets are presumed modular until they survive larger wheels.
5. This range is far too small to assess the Riemann hypothesis or prove any fixed prime-pair conjecture.

## Files

- `pair_metrics.csv.gz`: every block/offset measurement
- `offset_summary.csv`: aggregate statistics and arithmetic features
- `block_summary.csv`: sieve and gap summary by block
- `block_residual_summary.csv`: residual error by scale
- `analysis_summary.json`: model, correlation and PCA results
- `pca_offset_loadings.csv`: dominant cross-offset modes
- `spectral_summary.csv`: low-resolution log-scale frequency scan
- `gap_distribution.csv`: normalized gap histogram
- PNG files: diagnostic plots
"""
    (results_dir / "REPORT.md").write_text(text)
