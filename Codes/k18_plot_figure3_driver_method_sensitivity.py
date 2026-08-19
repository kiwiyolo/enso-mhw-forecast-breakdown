#!/usr/bin/env python
"""Build a publication-ready SI audit of the Figure 3 driver-regime framework."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oafm-matplotlib")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from scipy import stats


PAPER_DIR = Path(__file__).resolve().parents[1]
RAW = PAPER_DIR / "Data/Nature_real_rebuild/raw"
DEFAULT_OUTPUT = PAPER_DIR / "Figures/Figure3"
BASELINE_START = "1985-01"
BASELINE_END = "2014-12"
HISTORY_END = "2020-12"
PERSISTENCE_TRAIN_END = "2002-12"
MEMORY_VALIDATION_START = "2003-01"

BLACK = "#222222"
GREY = "#969696"
LIGHT_GREY = "#E3E3E3"
RED = "#D94B3A"
BLUE = "#3976B9"
ORANGE = "#E5A022"

SOURCE_CASES = (
    ("Current broad", (-10.0, 10.0, 120.0, 280.0)),
    ("Equatorial", (-5.0, 5.0, 120.0, 280.0)),
    ("Central-east", (-10.0, 10.0, 160.0, 280.0)),
    ("Nino3.4", (-5.0, 5.0, 190.0, 240.0)),
)
MAXIMUM_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
MARGIN_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot the archived diagnostics in --output-dir without reading raw SST inputs.",
    )
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "axes.titleweight": "normal",
            "axes.titlepad": 5.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def panel(axis: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        clip_on=False,
    )


def clean(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#ECECEC", linewidth=0.5, zorder=0)


def parse_cpc_nino(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        values = line.split()
        if len(values) < 10:
            continue
        rows.append(
            {
                "time": pd.Timestamp(int(values[0]), int(values[1]), 1),
                "nino34": float(values[9]),
            }
        )
    return pd.DataFrame(rows).set_index("time").sort_index()


def correlation_map(field: xr.DataArray, index: xr.DataArray) -> xr.DataArray:
    index = index.broadcast_like(field)
    valid = np.isfinite(field) & np.isfinite(index)
    x = field.where(valid)
    y = index.where(valid)
    x = x - x.mean("time", skipna=True)
    y = y - y.mean("time", skipna=True)
    denominator = np.sqrt(((x * x).sum("time")) * ((y * y).sum("time")))
    return (x * y).sum("time") / denominator.where(denominator > 0)


def source_mask(lat: xr.DataArray, lon: xr.DataArray, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    south, north, west, east = bounds
    return (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)


def driver_shares(
    enso_score: xr.DataArray,
    local_score: xr.DataArray,
    bounds: tuple[float, float, float, float],
) -> xr.Dataset:
    residual_score = (1.0 - enso_score - local_score).clip(0.05, 1.0)
    mask = source_mask(enso_score.lat, enso_score.lon, bounds)
    direct = enso_score.where(mask, 0.0)
    remote = enso_score.where(~mask, 0.0)
    total = direct + remote + local_score + residual_score
    return xr.Dataset(
        {
            "direct_share": direct / total,
            "remote_share": remote / total,
            "local_share": local_score / total,
            "residual_share": residual_score / total,
        }
    )


def classify_regime(
    shares: xr.Dataset,
    valid: np.ndarray,
    maximum_threshold: float,
    margin_threshold: float,
) -> np.ndarray:
    stack = np.stack(
        (
            shares.direct_share.values,
            shares.remote_share.values,
            shares.local_share.values,
        ),
        axis=0,
    )
    ordered = np.sort(stack, axis=0)
    maximum = np.nanmax(stack, axis=0)
    regime = np.nanargmax(np.nan_to_num(stack, nan=-1.0), axis=0).astype(np.int8)
    mixed = (maximum < maximum_threshold) | ((ordered[-1] - ordered[-2]) < margin_threshold)
    regime[mixed] = 3
    regime[~valid] = -1
    return regime


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(probabilities, cumulative, values)


def persistence_relation(
    local_score: xr.DataArray,
    continuation: xr.DataArray,
    event_predecessors: xr.DataArray,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    lon_grid, lat_grid = np.meshgrid(local_score.lon.values, local_score.lat.values)
    weights = np.cos(np.deg2rad(lat_grid)).clip(min=0.0)
    x = local_score.values.ravel()
    y = continuation.values.ravel()
    predecessors = event_predecessors.values.ravel()
    area = weights.ravel()
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(area) & (predecessors >= 6)
    frame = pd.DataFrame(
        {
            "local_r2": x[valid],
            "continuation": y[valid],
            "weight": area[valid],
            "lat": lat[valid],
            "lon": lon[valid],
        }
    )
    frame["block"] = (
        np.floor((frame.lat + 60.0) / 10.0).astype(int) * 18
        + np.floor(frame.lon / 20.0).astype(int)
    )
    boundaries = weighted_quantiles(
        frame.local_r2.to_numpy(), frame.weight.to_numpy(), np.linspace(0.0, 1.0, 9)
    )
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    frame["bin"] = np.digitize(frame.local_r2, boundaries[1:-1], right=True)

    block_ids = np.sort(frame.block.unique())
    block_lookup = {block: index for index, block in enumerate(block_ids)}
    bins = np.arange(8)
    sum_weight = np.zeros((len(block_ids), len(bins)), dtype=np.float64)
    sum_value = np.zeros_like(sum_weight)
    for row in frame.itertuples():
        block_index = block_lookup[row.block]
        sum_weight[block_index, row.bin] += row.weight
        sum_value[block_index, row.bin] += row.weight * row.continuation

    rng = np.random.default_rng(seed)
    boot = np.full((bootstrap, len(bins)), np.nan, dtype=np.float64)
    for iteration in range(bootstrap):
        sample = rng.integers(0, len(block_ids), len(block_ids))
        denominator = sum_weight[sample].sum(axis=0)
        numerator = sum_value[sample].sum(axis=0)
        boot[iteration] = np.divide(
            numerator,
            denominator,
            out=np.full(len(bins), np.nan),
            where=denominator > 0,
        )

    rows = []
    for bin_index in bins:
        subset = frame[frame.bin == bin_index]
        rows.append(
            {
                "bin": int(bin_index + 1),
                "mean_local_r2": weighted_average(
                    subset.local_r2.to_numpy(), subset.weight.to_numpy()
                ),
                "mean_mhw_continuation": weighted_average(
                    subset.continuation.to_numpy(), subset.weight.to_numpy()
                ),
                "ci_low": float(np.nanpercentile(boot[:, bin_index], 2.5)),
                "ci_high": float(np.nanpercentile(boot[:, bin_index], 97.5)),
                "grid_cells": int(len(subset)),
            }
        )
    binned = pd.DataFrame(rows)

    block_rows = []
    for _, subset in frame.groupby("block"):
        if len(subset) < 10:
            continue
        block_rows.append(
            {
                "local_r2": weighted_average(subset.local_r2.to_numpy(), subset.weight.to_numpy()),
                "continuation": weighted_average(
                    subset.continuation.to_numpy(), subset.weight.to_numpy()
                ),
            }
        )
    block_frame = pd.DataFrame(block_rows).dropna()
    association = stats.spearmanr(block_frame.local_r2, block_frame.continuation)
    summary = {
        "spearman_rho_across_10x20_degree_blocks": float(association.statistic),
        "spearman_p": float(association.pvalue),
        "spatial_blocks": int(len(block_frame)),
        "valid_grid_cells": int(len(frame)),
        "bootstrap_iterations": int(bootstrap),
    }
    return binned, summary


def calculate_diagnostics(args: argparse.Namespace) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nino = parse_cpc_nino(RAW / "cpc_ersst5_nino_monthly_1991_2020_base.txt")
    with xr.open_dataset(RAW / "noaa_ersstv5_sst_monthly.nc") as dataset:
        sst = dataset.sst.sel(time=slice(BASELINE_START, "2024-12"), lat=slice(60, -60)).load()
    sst = sst.sortby("lat")
    baseline = sst.sel(time=slice(BASELINE_START, BASELINE_END))
    anomaly = sst.groupby("time.month") - baseline.groupby("time.month").mean("time")
    history = anomaly.sel(time=slice(BASELINE_START, HISTORY_END))
    nino_series = xr.DataArray(
        nino.reindex(pd.DatetimeIndex(history.time.values)).nino34.to_numpy(),
        coords={"time": history.time},
        dims="time",
    )
    enso_r = correlation_map(history, nino_series)
    centered_nino = nino_series - nino_series.mean()
    beta = ((history - history.mean("time")) * centered_nino).sum("time") / (
        (centered_nino * centered_nino).sum("time")
    )
    residual = history - beta * nino_series
    local_r = correlation_map(
        residual.isel(time=slice(1, None)),
        residual.shift(time=1).isel(time=slice(1, None)),
    )
    enso_score = (enso_r * enso_r).clip(0.0, 1.0).fillna(0.0)
    local_score = (local_r * local_r).clip(0.0, 1.0).fillna(0.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        threshold = baseline.groupby("time.month").quantile(0.9, dim="time")
    validation_sst = sst.sel(time=slice(MEMORY_VALIDATION_START, HISTORY_END))
    validation_threshold = threshold.sel(
        month=xr.DataArray(validation_sst.time.dt.month, dims="time")
    ).drop_vars("month")
    indicator = (validation_sst > validation_threshold).where(np.isfinite(validation_sst))
    previous = indicator.shift(time=1)
    valid_pair = np.isfinite(indicator) & np.isfinite(previous)
    event_predecessors = ((previous == 1) & valid_pair).sum("time")
    continuation = (((indicator == 1) & (previous == 1) & valid_pair).sum("time")) / event_predecessors.where(
        event_predecessors > 0
    )

    target = sst.sel(time=slice("2023-05", "2024-08"))
    target_threshold = threshold.sel(
        month=xr.DataArray(target.time.dt.month, dims="time")
    ).drop_vars("month")
    intensity = (target - target_threshold).where(target > target_threshold, 0.0).mean(
        "time", skipna=True
    )
    valid_ocean = np.isfinite(sst.isel(time=-1).values)
    lat_weights = np.cos(np.deg2rad(sst.lat.values))[:, None]
    area_weights = np.broadcast_to(lat_weights, valid_ocean.shape)
    intensity_weights = np.nan_to_num(intensity.values, nan=0.0) * area_weights
    if np.sum(intensity_weights) <= 0:
        intensity_weights = area_weights * valid_ocean

    source_rows = []
    source_shares: dict[str, xr.Dataset] = {}
    current_shares: xr.Dataset | None = None
    for case, bounds in SOURCE_CASES:
        shares = driver_shares(enso_score, local_score, bounds)
        source_shares[case] = shares
        if case == "Current broad":
            current_shares = shares
        row: dict[str, object] = {
            "source_case": case,
            "south": bounds[0],
            "north": bounds[1],
            "west": bounds[2],
            "east": bounds[3],
        }
        for component in ("direct", "remote", "local", "residual"):
            row[f"{component}_share"] = weighted_average(
                shares[f"{component}_share"].values,
                intensity_weights,
            )
        source_rows.append(row)
    if current_shares is None:
        raise RuntimeError("Current source-box case was not calculated")

    baseline_regime = classify_regime(current_shares, valid_ocean, 0.50, 0.15)
    source_denominator = float(np.sum(area_weights[valid_ocean]))
    for row in source_rows:
        regime = classify_regime(source_shares[str(row["source_case"])], valid_ocean, 0.50, 0.15)
        row["area_weighted_regime_agreement"] = float(
            np.sum(area_weights[valid_ocean] * (regime[valid_ocean] == baseline_regime[valid_ocean]))
            / source_denominator
        )
    source_sensitivity = pd.DataFrame(source_rows)
    threshold_rows = []
    denominator = float(np.sum(area_weights[valid_ocean]))
    for maximum_threshold in MAXIMUM_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            regime = classify_regime(
                current_shares,
                valid_ocean,
                maximum_threshold,
                margin_threshold,
            )
            row = {
                "maximum_threshold": maximum_threshold,
                "margin_threshold": margin_threshold,
                "area_weighted_agreement": float(
                    np.sum(area_weights[valid_ocean] * (regime[valid_ocean] == baseline_regime[valid_ocean]))
                    / denominator
                ),
            }
            for code, name in enumerate(("direct", "remote", "local", "mixed")):
                row[f"{name}_area_fraction"] = float(
                    np.sum(area_weights[valid_ocean] * (regime[valid_ocean] == code)) / denominator
                )
            threshold_rows.append(row)
    threshold_sensitivity = pd.DataFrame(threshold_rows)

    persistence_history = history.sel(time=slice(BASELINE_START, PERSISTENCE_TRAIN_END))
    persistence_nino = nino_series.sel(time=slice(BASELINE_START, PERSISTENCE_TRAIN_END))
    centered_persistence_nino = persistence_nino - persistence_nino.mean()
    persistence_beta = (
        (persistence_history - persistence_history.mean("time")) * centered_persistence_nino
    ).sum("time") / (centered_persistence_nino * centered_persistence_nino).sum("time")
    persistence_residual = persistence_history - persistence_beta * persistence_nino
    validation_local_r = correlation_map(
        persistence_residual.isel(time=slice(1, None)),
        persistence_residual.shift(time=1).isel(time=slice(1, None)),
    )
    validation_local_score = (validation_local_r * validation_local_r).clip(0.0, 1.0).fillna(0.0)
    binned, persistence_summary = persistence_relation(
        validation_local_score,
        continuation,
        event_predecessors,
        args.bootstrap,
        args.seed,
    )
    diagnostics = xr.Dataset(
        {
            "enso_correlation": enso_r,
            "enso_explained_variance": enso_score,
            "enso_removed_lag1_correlation": local_r,
            "enso_removed_persistence_score": local_score,
            "held_out_validation_persistence_score": validation_local_score,
            "observed_mhw_continuation_probability": continuation,
            "observed_mhw_predecessor_months": event_predecessors,
            "baseline_driver_regime": (("lat", "lon"), baseline_regime),
            "mhw_intensity_2023_24": intensity,
        }
    )
    diagnostics.attrs.update(
        {
            "history_period": f"{BASELINE_START} to {HISTORY_END}",
            "persistence_training_period": f"{BASELINE_START} to {PERSISTENCE_TRAIN_END}",
            "mhw_memory_validation_period": f"{MEMORY_VALIDATION_START} to {HISTORY_END}",
            "mhw_climatology": f"{BASELINE_START} to {BASELINE_END}, calendar-month 90th percentile",
            "baseline_source_box": "10S-10N, 120E-80W",
            "baseline_regime_rule": "maximum share >= 0.50 and lead over runner-up >= 0.15",
            "interpretation": "association and persistence proxies; not a causal mixed-layer heat-budget attribution",
        }
    )
    summary: dict[str, object] = {
        **persistence_summary,
        "baseline_source_case": "Current broad",
        "baseline_maximum_threshold": 0.50,
        "baseline_margin_threshold": 0.15,
        "source_cases": [row[0] for row in SOURCE_CASES],
        "threshold_combinations": int(len(threshold_sensitivity)),
        "minimum_threshold_agreement": float(threshold_sensitivity.area_weighted_agreement.min()),
        "median_threshold_agreement": float(threshold_sensitivity.area_weighted_agreement.median()),
        "minimum_source_box_regime_agreement": float(
            source_sensitivity.area_weighted_regime_agreement.min()
        ),
    }
    return diagnostics, binned, source_sensitivity, threshold_sensitivity, summary


def draw_map(axis: plt.Axes, data: xr.DataArray, title: str, norm: Normalize) -> object:
    mesh = axis.pcolormesh(
        data.lon,
        data.lat,
        data,
        cmap="YlGnBu",
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        rasterized=True,
    )
    axis.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="#F3F3F3",
        edgecolor="#666666",
        linewidth=0.3,
        zorder=4,
    )
    axis.coastlines(resolution="110m", linewidth=0.3, color="#555555")
    axis.set_extent((-180, 180, -60, 60), crs=ccrs.PlateCarree())
    axis.set_title(title)
    return mesh


def plot_figure(
    diagnostics: xr.Dataset,
    binned: pd.DataFrame,
    source_sensitivity: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    summary: dict[str, object],
    output: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.9))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.05, 1.0),
        hspace=0.32,
        left=0.07,
        right=0.985,
        top=0.97,
        bottom=0.10,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.08)
    axis_a = fig.add_subplot(top[0], projection=ccrs.Robinson(central_longitude=180))
    axis_b = fig.add_subplot(top[1], projection=ccrs.Robinson(central_longitude=180))
    norm = Normalize(0.0, 0.8)
    mesh = draw_map(
        axis_a,
        diagnostics.enso_explained_variance,
        r"ENSO-associated SST variance ($r^2$)",
        norm,
    )
    bounds = SOURCE_CASES[0][1]
    south, north, west, east = bounds
    axis_a.plot(
        [west, east, east, west, west],
        [south, south, north, north, south],
        color=RED,
        linewidth=0.9,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    panel(axis_a, "a", -0.04, 1.05)
    draw_map(
        axis_b,
        diagnostics.enso_removed_persistence_score,
        r"ENSO-removed SST persistence (lag-1 $r^2$)",
        norm,
    )
    panel(axis_b, "b", -0.04, 1.05)
    colorbar = fig.colorbar(
        mesh,
        ax=[axis_a, axis_b],
        orientation="horizontal",
        fraction=0.055,
        pad=0.06,
        aspect=34,
    )
    colorbar.set_label(r"Squared correlation ($r^2$)")

    lower = outer[1].subgridspec(1, 3, width_ratios=(1.10, 1.02, 1.0), wspace=0.48)
    axis_c = fig.add_subplot(lower[0])
    axis_c.fill_between(
        binned.mean_local_r2,
        binned.ci_low,
        binned.ci_high,
        color=LIGHT_GREY,
        linewidth=0,
        zorder=1,
    )
    axis_c.plot(
        binned.mean_local_r2,
        binned.mean_mhw_continuation,
        color=BLACK,
        linewidth=1.15,
        marker="o",
        markersize=3.8,
        markerfacecolor=BLUE,
        markeredgecolor=BLACK,
        markeredgewidth=0.4,
        zorder=3,
    )
    p_value = float(summary["spearman_p"])
    p_text = "$P<0.001$" if p_value < 0.001 else f"$P={p_value:.3f}$"
    axis_c.text(
        0.04,
        0.96,
        rf"$\rho={float(summary['spearman_rho_across_10x20_degree_blocks']):.2f}$, {p_text}",
        transform=axis_c.transAxes,
        va="top",
        fontsize=6.4,
    )
    axis_c.set_xlabel(r"1985-2002 residual persistence (lag-1 $r^2$)")
    axis_c.set_ylabel("2003-2020 MHW continuation")
    axis_c.set_title("Held-out MHW memory")
    axis_c.set_ylim(0.0, min(1.0, float(binned.ci_high.max()) + 0.08))
    clean(axis_c)
    panel(axis_c, "c", -0.18, 1.08)

    axis_d = fig.add_subplot(lower[1])
    y = np.arange(len(source_sensitivity))
    bottoms = np.zeros(len(source_sensitivity))
    pieces = (
        ("direct_share", "Direct ENSO", RED),
        ("remote_share", "Remote ENSO", ORANGE),
        ("local_share", "Local persistence", BLUE),
        ("residual_share", "Residual", GREY),
    )
    for column, label, color in pieces:
        values = source_sensitivity[column].to_numpy()
        axis_d.barh(
            y,
            values,
            left=bottoms,
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=label,
        )
        bottoms += values
    axis_d.set_yticks(y, source_sensitivity.source_case)
    axis_d.invert_yaxis()
    axis_d.set_xlim(0, 1)
    axis_d.set_xlabel("Intensity-weighted share")
    axis_d.set_title("Source-box sensitivity")
    axis_d.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in pieces],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.43),
        columnspacing=0.7,
        handlelength=1.0,
    )
    clean(axis_d)
    panel(axis_d, "d", -0.22, 1.08)

    axis_e = fig.add_subplot(lower[2])
    pivot = threshold_sensitivity.pivot(
        index="maximum_threshold",
        columns="margin_threshold",
        values="area_weighted_agreement",
    ).loc[list(MAXIMUM_THRESHOLDS), list(MARGIN_THRESHOLDS)]
    image = axis_e.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="Blues",
        norm=Normalize(max(0.5, float(pivot.values.min()) - 0.02), 1.0),
    )
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            axis_e.text(
                column,
                row,
                f"{100 * value:.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if value > 0.86 else BLACK,
            )
    baseline_row = list(MAXIMUM_THRESHOLDS).index(0.50)
    baseline_column = list(MARGIN_THRESHOLDS).index(0.15)
    axis_e.add_patch(
        Rectangle(
            (baseline_column - 0.48, baseline_row - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor=RED,
            linewidth=1.1,
        )
    )
    axis_e.set_xticks(range(len(MARGIN_THRESHOLDS)), [f"{value:.2f}" for value in MARGIN_THRESHOLDS])
    axis_e.set_yticks(range(len(MAXIMUM_THRESHOLDS)), [f"{value:.2f}" for value in MAXIMUM_THRESHOLDS])
    axis_e.set_xlabel("Required lead over runner-up")
    axis_e.set_ylabel("Required maximum share")
    axis_e.set_title("Regime agreement (%)")
    axis_e.spines[:].set_visible(False)
    panel(axis_e, "e", -0.24, 1.08)
    cbar = fig.colorbar(image, ax=axis_e, orientation="horizontal", fraction=0.07, pad=0.20, aspect=20)
    cbar.set_label("Agreement with 0.50 / 0.15 rule")

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "Figure3_driver_method_sensitivity"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_document(
    output: Path,
    source_sensitivity: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    baseline = source_sensitivity[source_sensitivity.source_case == "Current broad"].iloc[0]
    minimum = float(threshold_sensitivity.area_weighted_agreement.min())
    median = float(threshold_sensitivity.area_weighted_agreement.median())
    source_minimum = float(source_sensitivity.area_weighted_regime_agreement.min())
    text = f"""# Supplementary Figure: robustness of the Figure 3 statistical driver framework

## Caption

**Observational diagnostics and sensitivity tests for the statistical driver-regime framework used in Figure 3.** **a**, Fraction of local monthly SST-anomaly variance linearly associated with the contemporaneous CPC ERSSTv5 Nino3.4 index during 1985-2020, expressed as squared Pearson correlation. The red outline marks the baseline broad tropical-Pacific source box (10S-10N, 120E-80W). **b**, Squared lag-1 correlation of monthly SST anomalies after linearly removing the contemporaneous Nino3.4 component. This quantity is interpreted as an ENSO-removed SST-persistence proxy, not as a resolved mixed-layer heat-budget term. **c**, Temporally held-out validation of that proxy: residual persistence is estimated during 1985-2002 and related to observed MHW continuation during 2003-2020. Points show eight equal-area bins and shading gives 95% confidence intervals from resampling 10-degree latitude by 20-degree longitude spatial blocks. The annotation reports the Spearman association across block means. **d**, Sensitivity of 2023-24 MHW-intensity-weighted global association shares to four explicit source-box definitions. Changing the box redistributes the ENSO-associated component between direct and remote labels while leaving local persistence and residual shares unchanged. **e**, Area-weighted agreement of alternative categorical regime maps with the baseline rule requiring a maximum share of at least 0.50 and a lead over the runner-up of at least 0.15. The red outline identifies the baseline rule; cell labels are percentages.

## Main results

- ENSO-associated variance and ENSO-removed persistence have distinct spatial structures, supporting their use as separate statistical predictors rather than interchangeable fields.
- Across {summary['spatial_blocks']} spatial blocks, 1985-2002 residual SST persistence is positively associated with 2003-2020 observed MHW continuation (Spearman rho={summary['spearman_rho_across_10x20_degree_blocks']:.3f}, P={summary['spearman_p']:.4g}). The non-overlapping periods reduce same-sample circularity and support the interpretation of lag-1 residual SST correlation as an MHW-memory proxy, but not as direct identification of a local physical process.
- Under the baseline source box, the global 2023-24 intensity-weighted shares are direct ENSO {baseline.direct_share:.3f}, remote ENSO {baseline.remote_share:.3f}, local persistence {baseline.local_share:.3f}, and residual {baseline.residual_share:.3f}.
- Alternative source boxes retain at least {source_minimum:.1%} area-weighted agreement with the baseline categorical regime map. The direct-versus-remote split is boundary-dependent and should not be interpreted as an independently observed physical partition.
- Across the 25 tested categorical thresholds, agreement with the baseline regime map ranges from {minimum:.1%} to 100%, with median agreement {median:.1%}. Complete regime fractions for every threshold pair are stored in `Figure3_driver_threshold_sensitivity.csv`.

## Calculation

For each ocean grid cell, the ENSO association score is

    S_ENSO = corr[SST'(t), Nino3.4(t)]^2.

The contemporaneous linear Nino3.4 component is removed grid-cell by grid-cell:

    SST'_res(t) = SST'(t) - beta Nino3.4(t),

after which the persistence score is

    S_local = corr[SST'_res(t), SST'_res(t-1)]^2.

For a specified source box, `S_ENSO` is labelled direct inside the box and remote outside it. The remaining score is `clip(1 - S_ENSO - S_local, 0.05, 1)`, after which all four scores are normalized to unit sum. A categorical regime is assigned only when the largest of direct, remote and local shares meets both the maximum-share and runner-up-margin requirements; otherwise it is labelled mixed.

Observed monthly MHW occurrence is defined when ERSST exceeds the local calendar-month 90th percentile for 1985-2014. For the held-out validation, the persistence predictor is estimated over 1985-2002 and MHW continuation is `P(MHW_t = 1 | MHW_t-1 = 1)` over 2003-2020. At least six predecessor MHW months are required at a grid cell.

## Interpretation boundary

This SI figure supports the internal statistical behaviour of the framework; it does not convert the framework into causal attribution. In simple one-predictor regression, squared Pearson correlation equals explained variance, but serial correlation and omitted climate modes limit causal interpretation. Linear removal of contemporaneous Nino3.4 does not fully remove lagged, nonlinear or seasonally varying ENSO teleconnections. Likewise, residual lag-1 persistence can reflect mixed-layer heat capacity, re-emergence, advection, other climate modes or unresolved forcing. Accordingly, the preferred terminology is `ENSO-associated source-region variance`, `remote ENSO association`, and `ENSO-removed local SST persistence`. Identifying physical basin-local drivers requires mixed-layer heat-budget terms or controlled forecast perturbation experiments.

## References

- Holbrook, N. J. et al. A global assessment of marine heatwaves and their drivers. *Nature Communications* 10, 2624 (2019). https://doi.org/10.1038/s41467-019-10206-z
- Jacox, M. G. et al. Global seasonal forecasts of marine heatwaves. *Nature* 604, 486-490 (2022). https://doi.org/10.1038/s41586-022-04573-9
- Compo, G. P. & Sardeshmukh, P. D. Removing ENSO-related variations from the climate record. *Journal of Climate* 23, 1957-1978 (2010). https://doi.org/10.1175/2009JCLI2735.1
- Gunnarson, J. L. et al. Removing ENSO's influence from global SST variability, with insights into the record-setting marine heat waves of 2023-24. *Bulletin of the American Meteorological Society* 106 (2025). https://doi.org/10.1175/BAMS-D-24-0023.1
- Shi, H. et al. Global decline in ocean memory over the 21st century. *Science Advances* 8, eabm3468 (2022). https://doi.org/10.1126/sciadv.abm3468
- Hobday, A. J. et al. A hierarchical approach to defining marine heatwaves. *Progress in Oceanography* 141, 227-238 (2016). https://doi.org/10.1016/j.pocean.2015.12.014

## Reproducibility

- Calculation and plotting script: `{Path(__file__).resolve()}`
- SST: `{(RAW / 'noaa_ersstv5_sst_monthly.nc').resolve()}`
- Nino3.4: `{(RAW / 'cpc_ersst5_nino_monthly_1991_2020_base.txt').resolve()}`
- Gridded diagnostics: `Figure3_driver_method_diagnostics.nc`
- Binned persistence diagnostic: `Figure3_driver_persistence_validation.csv`
- Source-box sensitivity: `Figure3_driver_source_box_sensitivity.csv`
- Regime-threshold sensitivity: `Figure3_driver_threshold_sensitivity.csv`
- Machine-readable summary: `Figure3_driver_method_sensitivity.json`
"""
    (output / "Figure3_driver_method_sensitivity.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with xr.open_dataset(args.output_dir / "Figure3_driver_method_diagnostics.nc") as dataset:
            diagnostics = dataset.load()
        binned = pd.read_csv(args.output_dir / "Figure3_driver_persistence_validation.csv")
        source_sensitivity = pd.read_csv(args.output_dir / "Figure3_driver_source_box_sensitivity.csv")
        threshold_sensitivity = pd.read_csv(args.output_dir / "Figure3_driver_threshold_sensitivity.csv")
        summary = json.loads(
            (args.output_dir / "Figure3_driver_method_sensitivity.json").read_text(encoding="utf-8")
        )
    else:
        diagnostics, binned, source_sensitivity, threshold_sensitivity, summary = calculate_diagnostics(args)
        diagnostics.to_netcdf(args.output_dir / "Figure3_driver_method_diagnostics.nc")
        binned.to_csv(args.output_dir / "Figure3_driver_persistence_validation.csv", index=False)
        source_sensitivity.to_csv(args.output_dir / "Figure3_driver_source_box_sensitivity.csv", index=False)
        threshold_sensitivity.to_csv(args.output_dir / "Figure3_driver_threshold_sensitivity.csv", index=False)
        (args.output_dir / "Figure3_driver_method_sensitivity.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    plot_figure(
        diagnostics,
        binned,
        source_sensitivity,
        threshold_sensitivity,
        summary,
        args.output_dir,
        args.dpi,
    )
    write_document(args.output_dir, source_sensitivity, threshold_sensitivity, summary)
    print(f"[Saved] {args.output_dir / 'Figure3_driver_method_sensitivity.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
