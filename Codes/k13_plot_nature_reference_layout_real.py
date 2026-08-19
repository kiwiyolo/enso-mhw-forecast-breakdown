#!/usr/bin/env python
"""Plot real-data Figures 1-4 using the supplied reference-panel layouts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oafm-matplotlib")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Rectangle
from scipy import stats


PAPER_DIR = Path(__file__).resolve().parents[1]
DERIVED = PAPER_DIR / "Data/Nature_real_rebuild/derived"
DEFAULT_ROOT = PAPER_DIR / "Figures"
FIGURE1_METRICS = PAPER_DIR / "output/Figure1/candidate09_metrics/Figure1_candidate09_metrics_by_init_lead.csv"

BLACK = "#222222"
GREY = "#969696"
LIGHT_GREY = "#E3E3E3"
RED = "#D94B3A"
BLUE = "#3976B9"
ORANGE = "#E5A022"
PURPLE = "#8B75B6"
TEAL = "#2A8D80"

VISUAL_DESIGN_NOTE = """## Visual design

The main figure uses a restrained Nature-style information hierarchy: no in-figure headline, short panel titles, a shared mechanism palette, direct labelling only where it carries the central result, and detailed statistical or methodological annotation in this companion document. The visual benchmark was the main-figure treatment in England et al., *Nature* (2025), https://www.nature.com/articles/s41586-025-08903-5, and Peng et al., *Nature Geoscience* (2025), https://www.nature.com/articles/s41561-025-01700-9. This is a design reference only; all plotted values are generated from the data and methods documented here.
"""

STRONG_EVENTS = ("1997/98", "2015/16")
TARGET_EVENT = "2023/24"
HISTORICAL_GROUP = "Comparable events"

BASIN_ORDER = (
    "North Pacific", "South Pacific", "Indian Ocean", "North Atlantic",
    "Tropical Atlantic", "South Atlantic", "Global 60S-60N",
)

BASIN_SHORT = {
    "North Pacific": "North\nPacific", "South Pacific": "South\nPacific",
    "Indian Ocean": "Indian\nOcean", "North Atlantic": "North\nAtlantic",
    "Tropical Atlantic": "Tropical\nAtlantic", "South Atlantic": "South\nAtlantic",
    "Global 60S-60N": "Global\n60S-60N",
}

BASIN_BOUNDS = {
    "North Pacific": (0.0, 60.0, 120.0, 280.0),
    "South Pacific": (-60.0, 0.0, 120.0, 290.0),
    "Indian Ocean": (-60.0, 30.0, 20.0, 120.0),
    "North Atlantic": (20.0, 60.0, 280.0, 360.0),
    "Tropical Atlantic": (-20.0, 20.0, 280.0, 360.0),
    "South Atlantic": (-60.0, 0.0, 290.0, 360.0),
    "Global 60S-60N": (-60.0, 60.0, 0.0, 360.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--figure1-sedi-only",
        action="store_true",
        help="Generate the SEDI version of Figure 1 without regenerating Figures 1-4.",
    )
    parser.add_argument(
        "--figure1-rmse-only",
        action="store_true",
        help="Generate the MHW-intensity RMSE version of Figure 1 without regenerating Figures 1-4.",
    )
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
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


def panel(axis: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    axis.text(x, y, label, transform=axis.transAxes, fontsize=8, fontweight="bold", fontstyle="normal", va="bottom", clip_on=False)


def clean(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#ECECEC", linewidth=0.5, zorder=0)


def save(figure: plt.Figure, directory: Path, stem: str, dpi: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def prediction_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    fit = stats.linregress(x, y)
    fitted = fit.intercept + fit.slope * grid
    residual = y - (fit.intercept + fit.slope * x)
    degrees = max(len(x) - 2, 1)
    standard = np.sqrt(np.sum(residual * residual) / degrees)
    spread = np.sqrt(1.0 + 1.0 / len(x) + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    half = stats.t.ppf(0.975, degrees) * standard * spread
    return fitted, fitted - half, fitted + half, fit


def predictive_p(x: np.ndarray, y: np.ndarray, challenge_x: float, challenge_y: float) -> tuple[float, float, float]:
    fit = stats.linregress(x, y)
    expected = fit.intercept + fit.slope * challenge_x
    residuals = y - (fit.intercept + fit.slope * x)
    degrees = max(len(x) - 2, 1)
    standard = np.sqrt(np.sum(residuals**2) / degrees)
    leverage = 1 + 1 / len(x) + (challenge_x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    statistic = (challenge_y - expected) / (standard * np.sqrt(leverage))
    return float(expected), float(challenge_y - expected), float(2 * stats.t.sf(abs(statistic), degrees))


def phase_subset(frame: pd.DataFrame, phase: str, metric: str = "AUC") -> pd.DataFrame:
    return (
        frame[frame.Nino34 >= 0.5].dropna(subset=["Nino34", metric])
        if phase == "El Nino"
        else frame[frame.Nino34 <= -0.5].dropna(subset=["Nino34", metric])
    )


def bootstrap_phase_slopes(
    frame: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
    metric: str = "AUC",
) -> list[dict[str, float | int | str]]:
    phases = ("El Nino", "La Nina")
    point = {}
    for phase in phases:
        selected = phase_subset(frame, phase, metric)
        point[phase] = stats.linregress(selected.Nino34.abs(), selected[metric])
    years = np.sort(frame.time.dt.year.unique())
    samples = {phase: [] for phase in phases}
    correlation_samples = {phase: [] for phase in phases}
    contrast = []
    correlation_contrast = []
    for _ in range(iterations):
        sampled_years = rng.choice(years, len(years), replace=True)
        sampled = pd.concat(
            [frame[frame.time.dt.year == year] for year in sampled_years], ignore_index=True
        )
        slopes = {}
        correlations = {}
        for phase in phases:
            selected = phase_subset(sampled, phase, metric)
            x = selected.Nino34.abs().to_numpy()
            if len(selected) < 3 or np.std(x) <= 1e-8:
                break
            fit = stats.linregress(x, selected[metric])
            slopes[phase] = fit.slope
            correlations[phase] = fit.rvalue
        if len(slopes) == 2:
            for phase in phases:
                samples[phase].append(slopes[phase])
                correlation_samples[phase].append(correlations[phase])
            contrast.append(slopes["El Nino"] - slopes["La Nina"])
            correlation_contrast.append(correlations["El Nino"] - correlations["La Nina"])
    rows = []
    for phase in phases:
        low, high = np.percentile(samples[phase], (2.5, 97.5))
        correlation_low, correlation_high = np.percentile(
            correlation_samples[phase], (2.5, 97.5)
        )
        rows.append(
            {
                "Phase": phase,
                "Slope": point[phase].slope,
                "CI_low": low,
                "CI_high": high,
                "Correlation": point[phase].rvalue,
                "Correlation_CI_low": correlation_low,
                "Correlation_CI_high": correlation_high,
                "P": point[phase].pvalue,
                "n": len(phase_subset(frame, phase, metric)),
            }
        )
    contrast = np.asarray(contrast)
    contrast_point = point["El Nino"].slope - point["La Nina"].slope
    low, high = np.percentile(contrast, (2.5, 97.5))
    p_value = 2.0 * min(
        (1 + np.sum(contrast <= 0)) / (len(contrast) + 1),
        (1 + np.sum(contrast >= 0)) / (len(contrast) + 1),
    )
    rows.append(
        {
            "Phase": "El Nino minus La Nina",
            "Slope": contrast_point,
            "CI_low": low,
            "CI_high": high,
            "Correlation": point["El Nino"].rvalue - point["La Nina"].rvalue,
            "Correlation_CI_low": float(np.percentile(correlation_contrast, 2.5)),
            "Correlation_CI_high": float(np.percentile(correlation_contrast, 97.5)),
            "P": min(float(p_value), 1.0),
            "n": len(contrast),
        }
    )
    return rows


def figure1(output: Path, dpi: int, bootstrap: int, seed: int) -> dict[str, float]:
    events = pd.read_csv(DERIVED / "figure1_event_relation.csv")
    monthly = pd.read_csv(DERIVED / "figure1_monthly_t1_t9.csv", parse_dates=["time"])
    historical = events[events.Event != "2023/24"]
    target = events[events.Event == "2023/24"].iloc[0]
    x, y = historical.Peak_Nino34.to_numpy(), historical.Mean_AUC.to_numpy()
    x_grid = np.linspace(max(0.4, x.min() - 0.1), max(x.max(), target.Peak_Nino34) + 0.15, 250)
    fitted, lower, upper, fit = prediction_band(x, y, x_grid)
    expected, residual, residual_p = predictive_p(x, y, target.Peak_Nino34, target.Mean_AUC)

    fig = plt.figure(figsize=(7.2, 5.35))
    outer = fig.add_gridspec(
        2,
        5,
        height_ratios=(1.48, 1.0),
        width_ratios=(1, 1, 1, 0.8, 0.8),
        hspace=0.34,
        wspace=0.72,
    )
    ax_a = fig.add_subplot(outer[0, :])
    left = outer[1, :3].subgridspec(2, 1, height_ratios=(4.2, 0.58), hspace=0.08)
    ax_b = fig.add_subplot(left[0])
    ax_strip = fig.add_subplot(left[1], sharex=ax_b)
    ax_c = fig.add_subplot(outer[1, 3:])
    ax_a.fill_between(x_grid, lower, upper, color=LIGHT_GREY, alpha=0.72, linewidth=0)
    ax_a.plot(x_grid, fitted, color=BLACK, linewidth=1.25)
    for row in historical.itertuples():
        highlight = row.Event in {"1997/98", "2015/16"}
        ax_a.scatter(
            row.Peak_Nino34,
            row.Mean_AUC,
            s=38 if highlight else 25,
            facecolor=BLACK if highlight else "#BEBEBE",
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=4,
        )
        if highlight:
            ax_a.annotate(
                row.Event,
                (row.Peak_Nino34, row.Mean_AUC),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=6.2,
            )
    ax_a.scatter(target.Peak_Nino34, target.Mean_AUC, s=50, facecolor=RED, edgecolor=BLACK, linewidth=0.65, zorder=6)
    ax_a.annotate("2023/24", (target.Peak_Nino34, target.Mean_AUC), xytext=(8, -2), textcoords="offset points", color=RED, fontsize=7.0, fontweight="bold")
    ax_a.hlines(expected, target.Peak_Nino34 - 0.38, target.Peak_Nino34, color=BLACK, linestyle=(0, (4, 3)), linewidth=0.9)
    ax_a.annotate("", xy=(target.Peak_Nino34, target.Mean_AUC), xytext=(target.Peak_Nino34, expected), arrowprops={"arrowstyle": "<->", "color": RED, "lw": 1.2})
    ax_a.text(
        target.Peak_Nino34 + 0.05,
        (expected + target.Mean_AUC) / 2,
        rf"$\Delta$AUC {residual:+.2f}",
        color=RED,
        va="center",
        fontsize=6.7,
    )
    p_text = "$P<0.001$" if fit.pvalue < 0.001 else f"$P={fit.pvalue:.3f}$"
    ax_a.text(
        0.02,
        0.96,
        f"Historical: $r={fit.rvalue:.2f}$, {p_text}",
        transform=ax_a.transAxes,
        va="top",
        fontsize=6.5,
    )
    ax_a.set(xlabel=r"El Nino amplitude (3-month Nino3.4, $^\circ$C)", ylabel="Global MHW forecast skill (AUC)", ylim=(0.45, 0.85))
    clean(ax_a)
    panel(ax_a, "a", -0.075, 1.025)

    ax_b.plot(monthly.time, monthly.AUC, color=BLACK, linewidth=0.9)
    for label in (*STRONG_EVENTS, TARGET_EVENT):
        row = events[events.Event == label].iloc[0]
        start, end = pd.Timestamp(row.Start_Month), pd.Timestamp(row.End_Month) + pd.offsets.MonthEnd(1)
        ax_b.axvspan(start, end, color=RED if label == "2023/24" else GREY, alpha=0.13, linewidth=0)
        ax_b.text(start + (end - start) / 2, 0.875, label, ha="center", va="top", fontsize=5.9, color=RED if label == "2023/24" else BLACK)
    ax_b.set(ylabel="Forecast skill (AUC)", ylim=(0.35, 0.90))
    ax_b.tick_params(labelbottom=False)
    clean(ax_b)
    panel(ax_b, "b", -0.10, 1.03)
    values = monthly.Nino34.to_numpy()[None]
    extent = (mdates.date2num(monthly.time.iloc[0]), mdates.date2num(monthly.time.iloc[-1] + pd.offsets.MonthEnd()), 0, 1)
    ax_strip.imshow(values, aspect="auto", extent=extent, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("Nino3.4", rotation=0, ha="right", va="center", labelpad=5)
    ax_strip.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_strip.spines[["top", "right", "left"]].set_visible(False)

    rng = np.random.default_rng(seed)
    slopes = bootstrap_phase_slopes(monthly, bootstrap, rng)
    slope_lookup = {row["Phase"]: row for row in slopes}
    display = ("El Nino", "La Nina")
    colors = (RED, BLUE)
    values = np.asarray([slope_lookup[phase]["Correlation"] for phase in display])
    lower_error = np.asarray(
        [
            slope_lookup[phase]["Correlation"]
            - slope_lookup[phase]["Correlation_CI_low"]
            for phase in display
        ]
    )
    upper_error = np.asarray(
        [
            slope_lookup[phase]["Correlation_CI_high"]
            - slope_lookup[phase]["Correlation"]
            for phase in display
        ]
    )
    positions = np.asarray((-0.32, 0.32))
    ax_c.bar(
        positions,
        values,
        color=colors,
        edgecolor=BLACK,
        linewidth=0.55,
        width=0.25,
        zorder=2,
    )
    ax_c.errorbar(
        positions,
        values,
        yerr=np.vstack((lower_error, upper_error)),
        fmt="none",
        ecolor=BLACK,
        capsize=3,
        linewidth=0.9,
        zorder=3,
    )
    for color_index, (xpos, phase, value) in enumerate(
        zip(positions, display, values, strict=True)
    ):
        row = slope_lookup[phase]
        p_text = "$P<0.001$" if row["P"] < 0.001 else f"$P={row['P']:.3f}$"
        ax_c.text(
            xpos,
            max(value, row["Correlation_CI_high"]) + 0.055,
            f"$r={value:.2f}$\n{p_text}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=colors[color_index],
        )
    ax_c.axhline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
    ax_c.set_xticks(positions, display)
    ax_c.set_ylabel("Skill-intensity correlation, $r$")
    ax_c.set_xlim(-0.68, 0.68)
    ax_c.set_ylim(-0.10, 0.90)
    clean(ax_c)
    panel(ax_c, "c", -0.10, 1.03)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.10)
    save(fig, output, "Figure1", dpi)
    events.to_csv(output / "Figure1_source_events.csv", index=False)
    pd.DataFrame(slopes).to_csv(output / "Figure1_phase_slopes.csv", index=False)
    result = {
        "r": fit.rvalue,
        "p": fit.pvalue,
        "expected": expected,
        "observed": target.Mean_AUC,
        "residual": residual,
        "predictive_p": residual_p,
        "el_nino_slope": slope_lookup["El Nino"]["Slope"],
        "la_nina_slope": slope_lookup["La Nina"]["Slope"],
        "el_nino_correlation": slope_lookup["El Nino"]["Correlation"],
        "el_nino_correlation_p": slope_lookup["El Nino"]["P"],
        "la_nina_correlation": slope_lookup["La Nina"]["Correlation"],
        "la_nina_correlation_p": slope_lookup["La Nina"]["P"],
        "slope_contrast": slope_lookup["El Nino minus La Nina"]["Slope"],
        "slope_contrast_p": slope_lookup["El Nino minus La Nina"]["P"],
    }
    (output / "Figure1_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_figure1_doc(output / "Figure1.md", result, len(historical))
    return result


def figure1_detailed_metrics() -> pd.DataFrame:
    return pd.read_csv(FIGURE1_METRICS)


def figure1_sedi_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = figure1_detailed_metrics()
    selected = detailed[
        (detailed.Method == "NMME ensemble mean") & detailed.Lead_Month.between(1, 9)
    ].copy()
    selected["time"] = pd.to_datetime(selected.Target_Month)
    monthly = selected.groupby("time", as_index=False).agg(
        SEDI=("SEDI", "mean"), leads=("Lead_Month", "nunique")
    )
    monthly = monthly[monthly.leads == 9].copy()
    nino = pd.read_csv(DERIVED / "figure1_monthly_t1_t9.csv", parse_dates=["time"])
    monthly = monthly.merge(nino[["time", "Nino34"]], on="time", how="left", validate="one_to_one")
    if monthly[["SEDI", "Nino34"]].isna().any().any():
        raise RuntimeError("Figure 1 SEDI input contains missing target-month values")

    events = pd.read_csv(DERIVED / "figure1_event_relation.csv")
    means = []
    counts = []
    for event in events.itertuples():
        window = monthly[
            monthly.time.between(pd.Timestamp(event.Start_Month), pd.Timestamp(event.End_Month))
        ]
        means.append(float(window.SEDI.mean()))
        counts.append(int(len(window)))
    events = events.drop(columns=["Mean_AUC", "Valid_Months"], errors="ignore")
    events["Mean_SEDI"] = means
    events["Valid_Months"] = counts
    if (events.Valid_Months != 12).any():
        raise RuntimeError("Every Figure 1 event must contain 12 complete target months")
    return events, monthly


def plot_figure1a_robustness_panel(
    events: pd.DataFrame,
    value_column: str,
    ylabel: str,
    delta_label: str,
    output: Path,
    stem: str,
    dpi: int,
    lower_bound: float | None = None,
    unit: str = "",
) -> tuple[pd.DataFrame, pd.Series, object, float, float, float]:
    """Plot one SI panel using the visual and statistical contract of Figure 1a."""
    historical = events[events.Event != TARGET_EVENT]
    target = events[events.Event == TARGET_EVENT].iloc[0]
    x = historical.Peak_Nino34.to_numpy()
    y = historical[value_column].to_numpy()
    target_value = float(target[value_column])
    x_grid = np.linspace(max(0.4, x.min() - 0.1), max(x.max(), target.Peak_Nino34) + 0.15, 250)
    fitted, lower, upper, fit = prediction_band(x, y, x_grid)
    expected, residual, residual_p = predictive_p(x, y, target.Peak_Nino34, target_value)

    fig, axis = plt.subplots(figsize=(7.2, 3.0))
    axis.fill_between(x_grid, lower, upper, color=LIGHT_GREY, alpha=0.72, linewidth=0)
    axis.plot(x_grid, fitted, color=BLACK, linewidth=1.25)
    for row in historical.itertuples():
        highlight = row.Event in STRONG_EVENTS
        axis.scatter(
            row.Peak_Nino34,
            getattr(row, value_column),
            s=38 if highlight else 25,
            facecolor=BLACK if highlight else "#BEBEBE",
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=4,
        )
        if highlight:
            axis.annotate(
                row.Event,
                (row.Peak_Nino34, getattr(row, value_column)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=6.2,
            )
    axis.scatter(
        target.Peak_Nino34,
        target_value,
        s=50,
        facecolor=RED,
        edgecolor=BLACK,
        linewidth=0.65,
        zorder=6,
    )
    axis.annotate(
        TARGET_EVENT,
        (target.Peak_Nino34, target_value),
        xytext=(8, -2),
        textcoords="offset points",
        color=RED,
        fontsize=7.0,
        fontweight="bold",
    )
    axis.hlines(
        expected,
        target.Peak_Nino34 - 0.38,
        target.Peak_Nino34,
        color=BLACK,
        linestyle=(0, (4, 3)),
        linewidth=0.9,
    )
    axis.annotate(
        "",
        xy=(target.Peak_Nino34, target_value),
        xytext=(target.Peak_Nino34, expected),
        arrowprops={"arrowstyle": "<->", "color": RED, "lw": 1.2},
    )
    axis.text(
        target.Peak_Nino34 + 0.05,
        (expected + target_value) / 2,
        f"{delta_label} {residual:+.2f}{unit}",
        color=RED,
        va="center",
        fontsize=6.7,
    )
    p_text = "$P<0.001$" if fit.pvalue < 0.001 else f"$P={fit.pvalue:.3f}$"
    axis.text(
        0.02,
        0.96,
        f"Historical: $r={fit.rvalue:.2f}$, {p_text}",
        transform=axis.transAxes,
        va="top",
        fontsize=6.5,
    )
    event_min = min(float(events[value_column].min()), float(lower.min()))
    event_max = max(float(events[value_column].max()), float(upper.max()))
    event_pad = max(0.035, 0.08 * (event_max - event_min))
    ymin = event_min - event_pad
    if lower_bound is not None:
        ymin = max(lower_bound, ymin)
    axis.set(
        xlabel=r"El Nino amplitude (3-month Nino3.4, $^\circ$C)",
        ylabel=ylabel,
        ylim=(ymin, event_max + event_pad),
    )
    clean(axis)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.95, bottom=0.19)
    save(fig, output, stem, dpi)
    return historical, target, fit, expected, residual, residual_p


def figure1_sedi(output: Path, dpi: int, bootstrap: int, seed: int) -> dict[str, float]:
    events, monthly = figure1_sedi_data()
    historical, target, fit, expected, residual, residual_p = plot_figure1a_robustness_panel(
        events,
        "Mean_SEDI",
        "Global MHW forecast skill (SEDI)",
        r"$\Delta$SEDI",
        output,
        "Figure1_SEDI",
        dpi,
    )
    rng = np.random.default_rng(seed)
    slopes = bootstrap_phase_slopes(monthly, bootstrap, rng, metric="SEDI")
    slope_lookup = {row["Phase"]: row for row in slopes}

    events.to_csv(output / "Figure1_SEDI_source_events.csv", index=False)
    monthly.to_csv(output / "Figure1_SEDI_monthly_t1_t9.csv", index=False)
    pd.DataFrame(slopes).to_csv(output / "Figure1_SEDI_phase_slopes.csv", index=False)
    result = {
        "metric": "SEDI",
        "r": fit.rvalue,
        "p": fit.pvalue,
        "expected": expected,
        "observed": float(target.Mean_SEDI),
        "residual": residual,
        "predictive_p": residual_p,
        "el_nino_slope": slope_lookup["El Nino"]["Slope"],
        "la_nina_slope": slope_lookup["La Nina"]["Slope"],
        "slope_contrast": slope_lookup["El Nino minus La Nina"]["Slope"],
        "slope_contrast_p": slope_lookup["El Nino minus La Nina"]["P"],
        "historical_events": int(len(historical)),
        "monthly_samples": int(len(monthly)),
        "lead_months": "1-9",
        "mhw_climatology": "1985-2014 calendar-month 90th percentile",
    }
    (output / "Figure1_SEDI_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_figure1_sedi_doc(output / "Figure1_SEDI.md", result, len(historical))
    return result


def figure1_rmse_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = figure1_detailed_metrics()
    selected = detailed[
        (detailed.Method == "NMME ensemble mean") & detailed.Lead_Month.between(1, 9)
    ].copy()
    selected["time"] = pd.to_datetime(selected.Target_Month)
    monthly = (
        selected.groupby("time")
        .agg(
            RMSE=("RMSE", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            leads=("Lead_Month", "nunique"),
        )
        .reset_index()
    )
    monthly = monthly[monthly.leads == 9].copy()
    nino = pd.read_csv(DERIVED / "figure1_monthly_t1_t9.csv", parse_dates=["time"])
    monthly = monthly.merge(nino[["time", "Nino34"]], on="time", how="left", validate="one_to_one")
    if monthly[["RMSE", "Nino34"]].isna().any().any():
        raise RuntimeError("Figure 1 RMSE input contains missing target-month values")

    events = pd.read_csv(DERIVED / "figure1_event_relation.csv")
    means = []
    counts = []
    for event in events.itertuples():
        window = monthly[
            monthly.time.between(pd.Timestamp(event.Start_Month), pd.Timestamp(event.End_Month))
        ]
        means.append(float(np.sqrt(np.mean(np.square(window.RMSE)))))
        counts.append(int(len(window)))
    events = events.drop(columns=["Mean_AUC", "Valid_Months"], errors="ignore")
    events["Mean_RMSE"] = means
    events["Valid_Months"] = counts
    if (events.Valid_Months != 12).any():
        raise RuntimeError("Every Figure 1 event must contain 12 complete target months")
    return events, monthly


def figure1_rmse(output: Path, dpi: int, bootstrap: int, seed: int) -> dict[str, float]:
    events, monthly = figure1_rmse_data()
    historical, target, fit, expected, residual, residual_p = plot_figure1a_robustness_panel(
        events,
        "Mean_RMSE",
        r"Global MHW intensity RMSE ($^\circ$C)",
        r"$\Delta$RMSE",
        output,
        "Figure1_RMSE",
        dpi,
        lower_bound=0.0,
        unit=r" $^\circ$C",
    )
    rng = np.random.default_rng(seed)
    slopes = bootstrap_phase_slopes(monthly, bootstrap, rng, metric="RMSE")
    slope_lookup = {row["Phase"]: row for row in slopes}

    events.to_csv(output / "Figure1_RMSE_source_events.csv", index=False)
    monthly.to_csv(output / "Figure1_RMSE_monthly_t1_t9.csv", index=False)
    pd.DataFrame(slopes).to_csv(output / "Figure1_RMSE_phase_slopes.csv", index=False)
    result = {
        "metric": "MHW intensity RMSE",
        "r": fit.rvalue,
        "p": fit.pvalue,
        "expected": expected,
        "observed": float(target.Mean_RMSE),
        "residual": residual,
        "predictive_p": residual_p,
        "el_nino_slope": slope_lookup["El Nino"]["Slope"],
        "la_nina_slope": slope_lookup["La Nina"]["Slope"],
        "slope_contrast": slope_lookup["El Nino minus La Nina"]["Slope"],
        "slope_contrast_p": slope_lookup["El Nino minus La Nina"]["P"],
        "historical_events": int(len(historical)),
        "monthly_samples": int(len(monthly)),
        "lead_months": "1-9",
        "mhw_climatology": "1985-2014 calendar-month 90th percentile",
        "lower_is_better": True,
    }
    (output / "Figure1_RMSE_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_figure1_rmse_doc(output / "Figure1_RMSE.md", result, len(historical))
    return result


def lon_label(value: float) -> str:
    if value == 180:
        return "180"
    return f"{int(value)}E" if value < 180 else f"{int(360 - value)}W"


def hov_panel(axis: plt.Axes, data: xr.DataArray, limit: float, show_x: bool, show_y: bool) -> object:
    mesh = axis.pcolormesh(data.lon, data.relative_month, data, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), shading="auto", rasterized=True)
    axis.axhline(0, color=BLACK, linestyle=(0, (3, 2)), linewidth=0.65)
    axis.axhline(-6, color="#777777", linestyle=(0, (3, 2)), linewidth=0.5)
    axis.axhline(6, color="#777777", linestyle=(0, (3, 2)), linewidth=0.5)
    axis.set_ylim(-18, 18)
    axis.set_yticks((-18, -12, -6, 0, 6, 12, 18))
    axis.set_xticks((120, 150, 180, 210, 240, 270), [lon_label(v) for v in (120, 150, 180, 210, 240, 270)])
    axis.tick_params(labelbottom=show_x, labelleft=show_y)
    return mesh


def figure2(output: Path, dpi: int) -> dict[str, float]:
    process_skill = pd.read_csv(DERIVED / "figure2_process_forecast_skill.csv")
    fidelity_audit = pd.read_csv(DERIVED / "figure2_pattern_fidelity_lead_audit.csv")
    errors = pd.read_csv(DERIVED / "figure2_source_signal_errors.csv")
    skill = pd.read_csv(DERIVED / "figure2_source_mhw_skill.csv")
    relative_error_scales = pd.read_csv(DERIVED / "figure2_relative_error_scales.csv")
    source_maps = xr.open_dataset(DERIVED / "figure2_source_process_maps_t1_t9.nc")
    metadata = json.loads((DERIVED / "figure2_process_forecast_skill_metadata.json").read_text())
    fig = plt.figure(figsize=(7.2, 5.65))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.35, 1.05),
        hspace=0.37,
        left=0.085,
        right=0.985,
        top=0.98,
        bottom=0.075,
    )
    top = outer[0].subgridspec(
        3,
        3,
        width_ratios=(1.36, 1.36, 0.86),
        wspace=0.17,
        hspace=0.24,
    )
    map_specs = (
        ("SST", "sst_comparable_relative_error", "sst_target_relative_error", "SST"),
        ("Zonal wind stress", "stress_comparable_relative_error", "stress_target_relative_error", "Zonal wind stress"),
        ("Convection proxy", "precipitation_comparable_relative_error", "precipitation_target_relative_error", "Precipitation"),
    )
    relative_error_limit = 200.0
    map_axes: list[plt.Axes] = []
    skill_axes: list[plt.Axes] = []
    map_mesh = None
    for row, (process_name, comparable_name, target_name, row_label) in enumerate(map_specs):
        for column, (field_name, heading) in enumerate(
            ((comparable_name, "Comparable events"), (target_name, "2023/24"))
        ):
            ax_map = fig.add_subplot(top[row, column], projection=ccrs.PlateCarree(central_longitude=180))
            field = source_maps[field_name]
            mesh = ax_map.pcolormesh(
                field.lon,
                field.lat,
                field,
                cmap="RdBu_r",
                norm=TwoSlopeNorm(
                    vmin=-relative_error_limit,
                    vcenter=0,
                    vmax=relative_error_limit,
                ),
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            ax_map.add_feature(
                cfeature.LAND.with_scale("110m"),
                facecolor="#F2F2F2",
                edgecolor="#777777",
                linewidth=0.3,
            )
            ax_map.coastlines(resolution="110m", linewidth=0.3, color="#666666")
            ax_map.set_extent((140, 260, -25, 25), crs=ccrs.PlateCarree())
            grid = ax_map.gridlines(
                draw_labels=True,
                xlocs=(140, 180, -140, -100),
                ylocs=(-20, -10, 0, 10, 20),
                linewidth=0.25,
                color="#A8A8A8",
                linestyle=":",
            )
            grid.top_labels = False
            grid.right_labels = False
            grid.bottom_labels = row == 2
            grid.left_labels = column == 0
            grid.xlabel_style = {"size": 5.2}
            grid.ylabel_style = {"size": 5.2}
            if row == 0:
                ax_map.set_title(heading, fontsize=6.5)
            if column == 0:
                ax_map.text(
                    0.02,
                    0.95,
                    row_label,
                    transform=ax_map.transAxes,
                    ha="left",
                    va="top",
                    color=BLUE,
                    fontsize=5.8,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
                )
            map_axes.append(ax_map)
            if map_mesh is None:
                map_mesh = mesh

        selected = process_skill[process_skill.Process == process_name]
        historical = selected[selected.Event != TARGET_EVENT]
        target_process = selected[selected.Event == TARGET_EVENT]
        historical_summary = historical.groupby("Lead").Pattern_correlation.agg(["mean", "min", "max"]).reset_index()
        ax_skill = fig.add_subplot(top[row, 2])
        ax_skill.fill_between(
            historical_summary.Lead,
            historical_summary["min"],
            historical_summary["max"],
            color=LIGHT_GREY,
            alpha=0.75,
            linewidth=0,
            label="Comparable-event range",
        )
        ax_skill.plot(
            historical_summary.Lead,
            historical_summary["mean"],
            color=BLACK,
            marker="o",
            markersize=2.8,
            linewidth=1.05,
            label="Comparable-event mean",
        )
        ax_skill.plot(
            target_process.Lead,
            target_process.Pattern_correlation,
            color=RED,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label="2023/24",
        )
        ax_skill.axhline(0, color="#999999", linewidth=0.55)
        ax_skill.set_xlim(0.7, 9.3)
        ax_skill.set_ylim(-0.10, 1.0)
        ax_skill.set_xticks((1, 3, 5, 7, 9))
        ax_skill.tick_params(labelbottom=row == 2)
        if row == 0:
            ax_skill.set_title("Pattern fidelity", fontsize=7.0)
            ax_skill.legend(frameon=False, fontsize=5.0, loc="lower left", handlelength=1.6)
        if row == 2:
            ax_skill.set_xlabel("Lead month")
        clean(ax_skill)
        skill_axes.append(ax_skill)
    fig.canvas.draw()
    left_maps = min(axis.get_position().x0 for axis in map_axes)
    right_maps = max(axis.get_position().x1 for axis in map_axes)
    map_bottom = min(axis.get_position().y0 for axis in map_axes)
    colorbar_axis = fig.add_axes(
        [
            left_maps,
            map_bottom - 0.055,
            right_maps - left_maps,
            0.010,
        ]
    )
    colorbar = fig.colorbar(
        map_mesh,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=(-200, -100, 0, 100, 200),
    )
    colorbar.set_label(
        "Relative forecast error (% of observed pattern RMS)",
        fontsize=5.2,
        labelpad=2.0,
    )
    colorbar.ax.tick_params(labelsize=5.0, length=2, pad=1)
    fig.text(
        0.016,
        min(0.992, map_axes[0].get_position().y1 + 0.010),
        "a",
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        va="bottom",
    )

    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.58, 0.82), wspace=0.22)
    ax_b = fig.add_subplot(bottom[0])
    metrics = errors.Metric.drop_duplicates().tolist()
    groups = (HISTORICAL_GROUP, TARGET_EVENT)
    score = (
        errors.pivot(
            index="Metric",
            columns="Group",
            values="Relative_to_overall_mean_percent",
        )
        .reindex(metrics)[list(groups)]
        .T
    )
    score_limit = max(25.0, 25.0 * math.ceil(float(np.nanmax(np.abs(score))) / 25.0))
    ax_b.imshow(
        score,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-score_limit, vcenter=0, vmax=score_limit),
        aspect="auto",
    )
    for row in range(2):
        for col in range(len(metrics)):
            value = float(score.iloc[row, col])
            ax_b.text(
                col,
                row,
                f"{value:+.0f}%",
                ha="center",
                va="center",
                fontsize=5.3,
                color="white" if abs(value) >= 0.42 * score_limit else BLACK,
            )
    short_metrics = (
        "Peak\nintensity",
        "Peak\ntiming",
        "SST\npattern",
        "Wind-stress\npattern",
        "Convection*\npattern",
    )
    ax_b.set_xticks(range(len(metrics)), short_metrics, fontsize=5.1)
    ax_b.set_yticks((0, 1), ("Comparable\nevents", "2023–24"), fontsize=5.5)
    ax_b.set_title("Error relative to overall mean (%)")
    ax_b.spines[:].set_visible(False)

    ax_c = fig.add_subplot(bottom[1])
    historical_events = skill[
        (skill.Group == HISTORICAL_GROUP) & (skill.Record == "Event")
    ]
    hist = skill[
        (skill.Group == HISTORICAL_GROUP) & (skill.Record == "Summary")
    ].iloc[0]
    target = skill[(skill.Group == "2023/24") & (skill.Record == "Event")].iloc[0]
    historical_min = float(historical_events.AUC.min())
    historical_max = float(historical_events.AUC.max())
    ax_c.add_patch(
        Rectangle(
            (-0.24, historical_min),
            0.48,
            historical_max - historical_min,
            facecolor=LIGHT_GREY,
            edgecolor="#999999",
            linewidth=0.6,
            zorder=1,
        )
    )
    ax_c.scatter(0, hist.AUC, s=28, color=BLACK, zorder=4)
    ax_c.scatter(1, target.AUC, s=38, color=RED, edgecolor=BLACK, linewidth=0.5, zorder=4)
    ax_c.hlines(hist.AUC, 0.25, 1.15, color=BLACK, linestyle=(0, (4, 3)), linewidth=0.85)
    ax_c.annotate("", xy=(1, target.AUC), xytext=(1, hist.AUC), arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.1})
    decline = float(hist.AUC - target.AUC)
    decline_percent = 100.0 * decline / float(hist.AUC)
    ax_c.text(0, hist.AUC + 0.012, f"{hist.AUC:.2f}", ha="center", fontsize=5.8)
    ax_c.text(1, target.AUC - 0.022, f"{target.AUC:.2f}", ha="center", va="top", color=RED, fontsize=5.8)
    ax_c.text(1.04, (hist.AUC + target.AUC) / 2, rf"$\Delta$AUC -{decline:.2f}", va="center", color=RED, fontsize=6.0)
    ax_c.set_xticks((0, 1), ("Comparable\nevents", "2023-24"))
    ax_c.set_ylabel("Source-region MHW AUC")
    ax_c.set_xlim(-0.55, 1.45)
    ax_c.set_ylim(0.55, 0.90)
    ax_c.set_title("Source-region MHW skill")
    clean(ax_c)
    fig.canvas.draw()
    lower_label_y = max(ax_b.get_position().y1, ax_c.get_position().y1) + 0.010
    fig.text(0.016, lower_label_y, "b", fontsize=8, fontweight="bold", fontstyle="normal", va="bottom")
    fig.text(
        ax_c.get_position().x0 - 0.025,
        lower_label_y,
        "c",
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        va="bottom",
    )
    save(fig, output, "Figure2", dpi)
    process_skill.to_csv(output / "Figure2_process_forecast_skill.csv", index=False)
    fidelity_audit.to_csv(output / "Figure2_pattern_fidelity_lead_audit.csv", index=False)
    pd.read_csv(DERIVED / "figure2_nmme_sst_start_coordinate_audit.csv").to_csv(
        output / "Figure2_nmme_sst_start_coordinate_audit.csv", index=False
    )
    relative_error_scales.to_csv(
        output / "Figure2_relative_error_scales.csv", index=False
    )
    errors.to_csv(output / "Figure2_source_signal_errors.csv", index=False)
    pd.read_csv(DERIVED / "figure2_source_signal_errors_by_event.csv").to_csv(
        output / "Figure2_source_signal_errors_by_event.csv", index=False
    )
    skill.to_csv(output / "Figure2_source_mhw_skill.csv", index=False)
    deprecated_profile = output / "Figure2_source_latitude_error_profiles.csv"
    if deprecated_profile.exists():
        deprecated_profile.unlink()
    source_maps.to_netcdf(output / "Figure2_source_process_maps_t1_t9.nc")
    result = {
        "historical_source_auc": hist.AUC,
        "target_source_auc": target.AUC,
        "difference": target.AUC - hist.AUC,
        "decline_percent": decline_percent,
        "sst_comparable_pattern_r": float(process_skill[(process_skill.Process == "SST") & (process_skill.Event != TARGET_EVENT)].Pattern_correlation.mean()),
        "sst_target_pattern_r": float(process_skill[(process_skill.Process == "SST") & (process_skill.Event == TARGET_EVENT)].Pattern_correlation.mean()),
        "stress_comparable_pattern_r": float(process_skill[(process_skill.Process == "Zonal wind stress") & (process_skill.Event != TARGET_EVENT)].Pattern_correlation.mean()),
        "stress_target_pattern_r": float(process_skill[(process_skill.Process == "Zonal wind stress") & (process_skill.Event == TARGET_EVENT)].Pattern_correlation.mean()),
        "convection_comparable_pattern_r": float(process_skill[(process_skill.Process == "Convection proxy") & (process_skill.Event != TARGET_EVENT)].Pattern_correlation.mean()),
        "convection_target_pattern_r": float(process_skill[(process_skill.Process == "Convection proxy") & (process_skill.Event == TARGET_EVENT)].Pattern_correlation.mean()),
        "fidelity_series_decreasing": int(fidelity_audit.Decreases_with_lead.sum()),
        "fidelity_series_total": int(len(fidelity_audit)),
        "comparable_relative_error_min": float(
            errors.loc[
                errors.Group == HISTORICAL_GROUP,
                "Relative_to_overall_mean_percent",
            ].min()
        ),
        "comparable_relative_error_max": float(
            errors.loc[
                errors.Group == HISTORICAL_GROUP,
                "Relative_to_overall_mean_percent",
            ].max()
        ),
        "target_relative_error_min": float(
            errors.loc[
                errors.Group == TARGET_EVENT,
                "Relative_to_overall_mean_percent",
            ].min()
        ),
        "target_relative_error_max": float(
            errors.loc[
                errors.Group == TARGET_EVENT,
                "Relative_to_overall_mean_percent",
            ].max()
        ),
    }
    write_figure2_doc(output / "Figure2.md", result, metadata)
    source_maps.close()
    return result


def figure3(output: Path, dpi: int) -> dict[str, float]:
    fields = xr.open_dataset(DERIVED / "figure3_driver_regime_and_mhw_intensity.nc")
    matrix = pd.read_csv(DERIVED / "figure3_teleconnection_efficiency.csv")
    shares = pd.read_csv(DERIVED / "figure3_basin_driver_contributions.csv").set_index("Basin").loc[list(BASIN_ORDER)].reset_index()
    activity = pd.read_csv(DERIVED / "figure3_local_mhw_activity.csv")
    activity_stats = json.loads(
        (DERIVED / "figure3_local_mhw_activity_statistics.json").read_text()
    )
    basin_skill = pd.read_csv(PAPER_DIR / "Figures/Figure3/Figure3_basin_mhw_skill.csv").set_index("basin")
    basin_loss = basin_skill.Canonical - basin_skill["2023/24"]
    fig = plt.figure(figsize=(7.2, 5.30))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.52, 1.0),
        hspace=0.25,
        left=0.065,
        right=0.985,
        top=0.98,
        bottom=0.105,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(6.0, 1.15), wspace=0.02)
    map_central_longitude = 200.0
    map_latitude_limit = 65.0
    map_west_edge = map_central_longitude - 180.0 + 0.001
    map_east_edge = map_central_longitude + 180.0 - 0.001
    ax_a = fig.add_subplot(
        top[0], projection=ccrs.Robinson(central_longitude=map_central_longitude)
    )
    ax_legend = fig.add_subplot(top[1])
    display_latitude = (
        (fields.lat >= -map_latitude_limit)
        & (fields.lat <= map_latitude_limit)
    )
    regime = fields.driver_regime.where(
        (fields.driver_regime >= 0) & display_latitude,
        drop=True,
    )
    colors = (RED, ORANGE, BLUE, PURPLE)
    ax_a.pcolormesh(regime.lon, regime.lat, regime, cmap=ListedColormap(colors), norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), 4), transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    intensity = fields.mhw_intensity_2023_24.where(display_latitude, drop=True)
    contours = ax_a.contour(intensity.lon, intensity.lat, intensity, levels=(0.25, 0.50, 0.90), colors=BLACK, linewidths=(0.45, 0.75, 1.15), transform=ccrs.PlateCarree())
    ax_a.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F3F3F3", edgecolor="#666666", linewidth=0.35, zorder=5)
    ax_a.coastlines(resolution="110m", linewidth=0.35, color="#555555")

    # Show a small amount of context beyond the 60S-60N analysis domain and
    # follow the projected parallels to preserve the map's natural aspect.
    edge_points = 241
    top_lon = np.linspace(map_west_edge, map_east_edge, edge_points)
    side_lat = np.linspace(map_latitude_limit, -map_latitude_limit, edge_points)
    boundary_lon = np.concatenate(
        (
            top_lon,
            np.full(edge_points, map_east_edge),
            top_lon[::-1],
            np.full(edge_points, map_west_edge),
        )
    )
    boundary_lat = np.concatenate(
        (
            np.full(edge_points, map_latitude_limit),
            side_lat,
            np.full(edge_points, -map_latitude_limit),
            side_lat[::-1],
        )
    )
    boundary_xy = ax_a.projection.transform_points(
        ccrs.PlateCarree(), boundary_lon, boundary_lat
    )[:, :2]
    ax_a.set_boundary(MplPath(boundary_xy, closed=True), transform=ax_a.transData)
    ax_a.set_xlim(boundary_xy[:, 0].min(), boundary_xy[:, 0].max())
    ax_a.set_ylim(boundary_xy[:, 1].min(), boundary_xy[:, 1].max())
    grid = ax_a.gridlines(
        draw_labels=False,
        xlocs=(80, -160, -40),
        ylocs=(-60, -40, -20, 0, 20, 40, 60),
        linewidth=0.25,
        color="#A8A8A8",
        linestyle=":",
        zorder=4,
    )
    for latitude in (-60, -40, -20, 0, 20, 40, 60):
        x_label, y_label = ax_a.projection.transform_point(
            map_west_edge, latitude, ccrs.PlateCarree()
        )
        latitude_label = (
            "0\N{DEGREE SIGN}"
            if latitude == 0
            else f"{abs(latitude)}\N{DEGREE SIGN}{'N' if latitude > 0 else 'S'}"
        )
        ax_a.annotate(
            latitude_label,
            xy=(x_label, y_label),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=5.0,
            annotation_clip=False,
            zorder=10,
        )
    for longitude, longitude_label in (
        (80.0, "80\N{DEGREE SIGN}E"),
        (200.0, "160\N{DEGREE SIGN}W"),
        (320.0, "40\N{DEGREE SIGN}W"),
    ):
        x_label, y_label = ax_a.projection.transform_point(
            longitude, -map_latitude_limit, ccrs.PlateCarree()
        )
        ax_a.annotate(
            longitude_label,
            xy=(x_label, y_label),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=5.0,
            annotation_clip=False,
            zorder=10,
        )
    lon_grid, lat_grid = np.meshgrid(regime.lon.values, regime.lat.values)
    valid_ocean = np.isfinite(regime.values)
    for basin, bounds in BASIN_BOUNDS.items():
        if basin == "Global 60S-60N":
            continue
        loss = float(basin_loss.loc[basin])
        if loss < 0.05:
            continue
        south, north, west, east = bounds
        basin_mask = (
            valid_ocean
            & (lat_grid >= south)
            & (lat_grid < north)
            & (lon_grid >= west)
            & (lon_grid < east)
        )
        sampled = basin_mask[::6, ::6]
        ax_a.scatter(
            lon_grid[::6, ::6][sampled],
            lat_grid[::6, ::6][sampled],
            s=0.7,
            color=BLACK,
            alpha=0.42,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    labels = (("North Pacific", 200, 34), ("South Pacific", 210, -30), ("Indian Ocean", 80, -24), ("North Atlantic", 318, 36), ("Tropical Atlantic", 315, 5), ("South Atlantic", 318, -32))
    for text_value, lon, lat in labels:
        ax_a.text(lon, lat, text_value, transform=ccrs.PlateCarree(), ha="center", va="center", fontsize=5.8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.82}, zorder=8)
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, ("Direct ENSO", "Remote ENSO", "Basin-local", "Mixed"), strict=True)]
    handles.extend(
        (
            Line2D([0], [0], color=BLACK, lw=0.45, label=r"MHW intensity 0.25 $^\circ$C"),
            Line2D([0], [0], color=BLACK, lw=0.75, label=r"MHW intensity 0.50 $^\circ$C"),
            Line2D([0], [0], color=BLACK, lw=1.15, label=r"MHW intensity 0.90 $^\circ$C"),
            Line2D([0], [0], marker=".", color=BLACK, lw=0, label="AUC loss > 0.05"),
        )
    )
    ax_legend.legend(handles=handles, loc="center left", frameon=False, ncol=1, fontsize=5.1, handlelength=1.7, labelspacing=0.53)
    ax_legend.axis("off")
    ax_a.set_title("MHW driver regimes and 2023/24 intensity", pad=4)

    lower = outer[1].subgridspec(
        1,
        3,
        width_ratios=(1.34, 0.92, 0.84),
        wspace=0.28,
    )

    # First summarize the observed driver-association balance, then diagnose
    # forecast teleconnection fidelity and the changing local MHW background.
    ax_b = fig.add_subplot(lower[0])
    x = np.arange(len(shares))
    bottoms = np.zeros(len(shares))
    categories = (
        ("direct_share", "Direct ENSO", RED),
        ("remote_share", "Remote ENSO", ORANGE),
        ("local_share", "Basin-local", BLUE),
        ("residual_share", "Residual", GREY),
    )
    for column, label, color in categories:
        value = shares[column].to_numpy() * 100
        ax_b.bar(
            x,
            value,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            width=0.72,
            label=label,
        )
        bottoms += value
    compact = ("N Pac", "S Pac", "Indian", "N Atl", "Trop Atl", "S Atl", "Global")
    ax_b.set_xticks(x, compact, rotation=34, ha="right")
    ax_b.set_ylabel("Association share (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_title("MHW-weighted driver associations")
    clean(ax_b)

    ax_c = fig.add_subplot(lower[1])
    values = matrix.set_index("Bridge")[["Historical", "2023/24", "Difference"]]
    ax_c.imshow(
        values,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.2),
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax_c.text(
                col,
                row,
                f"{values.iloc[row, col]:+.2f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if abs(values.iloc[row, col]) > 1.1 else BLACK,
            )
    ax_c.set_xticks(
        (0, 1, 2),
        ("Comparable", "2023–24", r"$\Delta$"),
        rotation=25,
        ha="right",
    )
    ax_c.set_yticks(range(len(values)), ("PNA", "PSA", "Indian", "Atlantic"))
    ax_c.set_title("Teleconnection-pattern fidelity")
    ax_c.spines[:].set_visible(False)

    ax_d = fig.add_subplot(lower[2])
    basin_activity = activity[activity.Series != "Local-dominated basin mean"]
    mean_activity = activity[activity.Series == "Local-dominated basin mean"].sort_values("Year")
    for _, basin_frame in basin_activity.groupby("Series"):
        basin_frame = basin_frame.sort_values("Year")
        ax_d.plot(
            basin_frame.Year,
            basin_frame.MHW_activity_C_days,
            color="#BDBDBD",
            linewidth=0.45,
            alpha=0.48,
            zorder=1,
        )
    trend = stats.linregress(mean_activity.Year, mean_activity.MHW_activity_C_days)
    ax_d.plot(
        mean_activity.Year,
        mean_activity.MHW_activity_C_days,
        color=BLUE,
        linewidth=1.3,
        label="Local-dominated basin mean",
        zorder=3,
    )
    ax_d.plot(
        mean_activity.Year,
        trend.intercept + trend.slope * mean_activity.Year,
        color=BLACK,
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        label="Linear trend",
        zorder=2,
    )
    recent = mean_activity[mean_activity.Year >= 2023]
    ax_d.scatter(
        recent.Year,
        recent.MHW_activity_C_days,
        s=25,
        color=RED,
        edgecolor=BLACK,
        linewidth=0.45,
        zorder=4,
        label="2023-24",
    )
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel(r"MHW activity ($^\circ$C days yr$^{-1}$)", labelpad=2)
    ax_d.set_title("Basin-local MHW activity")
    ax_d.legend(frameon=False, fontsize=5.0, loc="upper left", labelspacing=0.35)
    clean(ax_d)
    fig.canvas.draw()
    top_label_y = min(0.992, ax_a.get_position().y1 + 0.008)
    lower_label_y = max(ax_b.get_position().y1, ax_c.get_position().y1, ax_d.get_position().y1) + 0.010
    fig.text(0.016, top_label_y, "a", fontsize=8, fontweight="bold", fontstyle="normal", va="bottom")
    fig.text(0.016, lower_label_y, "b", fontsize=8, fontweight="bold", fontstyle="normal", va="bottom")
    fig.text(ax_c.get_position().x0 - 0.038, lower_label_y, "c", fontsize=8, fontweight="bold", fontstyle="normal", va="bottom")
    fig.text(ax_d.get_position().x0 - 0.028, lower_label_y, "d", fontsize=8, fontweight="bold", fontstyle="normal", va="bottom")
    save(fig, output, "Figure3", dpi)
    matrix.to_csv(output / "Figure3_teleconnection_efficiency.csv", index=False)
    shares.to_csv(output / "Figure3_driver_contributions.csv", index=False)
    activity.to_csv(output / "Figure3_local_mhw_activity.csv", index=False)
    (output / "Figure3_local_mhw_activity_statistics.json").write_text(
        json.dumps(activity_stats, indent=2), encoding="utf-8"
    )
    result = {
        "activity_trend_per_decade": activity_stats["theil_sen_slope_C_days_per_decade"],
        "activity_kendall_tau": activity_stats["kendall_tau"],
        "activity_kendall_p": activity_stats["kendall_p"],
        "activity_years": activity_stats["year_count"],
        "activity_basin_count": len(activity_stats["local_dominated_basins"]),
        "global_mhw_intensity_C": float(shares.loc[shares.Basin == "Global 60S-60N", "MHW_intensity_C"].iloc[0]),
    }
    write_figure3_doc(output / "Figure3.md", result)
    fields.close()
    return result


def draw_box(axis: plt.Axes, center: tuple[float, float], text: str, facecolor: str, width: float = 0.20) -> None:
    x, y = center
    patch = FancyBboxPatch((x - width / 2, y - 0.08), width, 0.16, boxstyle="round,pad=0.018", facecolor=facecolor, edgecolor="#555555", linewidth=0.7, transform=axis.transAxes)
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", transform=axis.transAxes, fontsize=6.1)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = BLACK, linestyle: str = "-") -> None:
    axis.annotate("", xy=end, xytext=start, xycoords=axis.transAxes, textcoords=axis.transAxes, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "linestyle": linestyle})


def sankey_band(
    axis: plt.Axes,
    source: tuple[float, float],
    target: tuple[float, float],
    color: str,
) -> None:
    x0, x1 = 0.25, 0.76
    middle = (x0 + x1) / 2
    low0, high0 = source
    low1, high1 = target
    vertices = [
        (x0, low0),
        (middle, low0),
        (middle, low1),
        (x1, low1),
        (x1, high1),
        (middle, high1),
        (middle, high0),
        (x0, high0),
        (x0, low0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    axis.add_patch(
        PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=0.72)
    )


def draw_attribution_sankey(axis: plt.Axes, global_row: pd.Series) -> None:
    pieces = (
        ("ENSO_source_error", "ENSO source", RED),
        ("Teleconnection_error", "Teleconnection", BLUE),
        ("Basin_local_error", "Basin-local", ORANGE),
        ("Irreducible", "Unresolved", GREY),
    )
    values = np.asarray([float(global_row[column]) for column, _, _ in pieces])
    total = float(values.sum())
    proportions = values / max(total, 1e-12)
    flow_height = 0.72
    source_bottom = 0.10
    target_bottom = 0.12
    gap = 0.025
    target_cursor = target_bottom
    source_cursor = source_bottom
    targets = []
    for proportion in proportions:
        height = proportion * flow_height
        targets.append((target_cursor, target_cursor + height))
        target_cursor += height
    for (column, label, color), value, proportion, target_interval in zip(
        pieces, values, proportions, targets, strict=True
    ):
        height = proportion * flow_height
        source_interval = (source_cursor, source_cursor + height)
        sankey_band(axis, source_interval, target_interval, color)
        axis.add_patch(
            Rectangle(
                (0.04, source_interval[0]),
                0.21,
                height,
                facecolor=color,
                edgecolor=BLACK,
                linewidth=0.45,
            )
        )
        if height >= 0.065:
            axis.text(
                0.145,
                np.mean(source_interval),
                f"{label}\n{100 * proportion:.0f}%",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white" if column in {"ENSO_source_error", "Teleconnection_error"} else BLACK,
            )
        else:
            axis.annotate(
                f"{label} {100 * proportion:.0f}%",
                xy=(0.145, np.mean(source_interval)),
                xytext=(0.145, min(0.96, source_interval[1] + 0.055)),
                ha="center",
                va="bottom",
                fontsize=5.0,
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.55},
            )
        source_cursor += height + gap
    axis.add_patch(
        Rectangle(
            (0.76, target_bottom),
            0.22,
            flow_height,
            facecolor="#F1F1F1",
            edgecolor=BLACK,
            linewidth=0.65,
        )
    )
    axis.text(
        0.87,
        target_bottom + flow_height / 2,
        f"Figure 1 residual\n$\\Delta$AUC = {total:.3f}",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Residual closure")


def figure4_validation_products(seed: int = 20260801) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe basin-event-lead validation products for Figure 4."""

    path = pd.read_csv(DERIVED / "figure4_path_samples.csv")
    local = pd.read_csv(DERIVED / "figure3_local_process_event_lead_basin.csv")
    source = path.groupby(["Event", "Lead"], as_index=False).Source_skill.first()
    teleconnection = path[["Event", "Lead", "Bridge", "Pattern_skill"]].copy()
    bridge_by_basin = {
        "North Pacific": "PNA (N. Pacific)",
        "South Pacific": "PSA (S. Pacific)",
        "Indian Ocean": "Indian bridge",
        "North Atlantic": "Atlantic bridge",
        "Tropical Atlantic": "Atlantic bridge",
        "South Atlantic": "Atlantic bridge",
    }
    local["Bridge"] = local.Basin.map(bridge_by_basin)
    samples = local.merge(source, on=["Event", "Lead"], validate="many_to_one").merge(
        teleconnection,
        on=["Event", "Lead", "Bridge"],
        validate="many_to_one",
    )
    samples["Source_error"] = 1.0 - samples.Source_skill
    samples["Teleconnection_error"] = 1.0 - samples.Pattern_skill
    samples = samples.sort_values(["Event", "Basin", "Lead"]).reset_index(drop=True)

    controls = pd.get_dummies(
        samples[["Basin", "Lead"]].astype({"Lead": str}),
        drop_first=True,
        dtype=float,
    )

    def leave_one_event_out(predictors: tuple[str, ...]) -> np.ndarray:
        design = controls.copy()
        for predictor in predictors:
            design[predictor] = samples[predictor]
        values = design.to_numpy(dtype=float)
        target = samples.AUC.to_numpy(dtype=float)
        prediction = np.full(len(samples), np.nan, dtype=float)
        for event in samples.Event.unique():
            train = samples.Event.ne(event).to_numpy()
            test = ~train
            mean = values[train].mean(axis=0)
            scale = values[train].std(axis=0)
            scale[scale < 1e-12] = 1.0
            train_design = np.column_stack(
                (np.ones(train.sum()), (values[train] - mean) / scale)
            )
            test_design = np.column_stack(
                (np.ones(test.sum()), (values[test] - mean) / scale)
            )
            coefficients = np.linalg.lstsq(
                train_design,
                target[train],
                rcond=None,
            )[0]
            prediction[test] = test_design @ coefficients
        return prediction

    model_specs = (
        ("Location + lead", ()),
        ("Source", ("Source_error",)),
        ("Source + teleconnection", ("Source_error", "Teleconnection_error")),
        (
            "Source + teleconnection + regional process",
            ("Source_error", "Teleconnection_error", "Local_ocean_pattern_error"),
        ),
    )
    target = samples.AUC.to_numpy(dtype=float)
    denominator = np.sum((target - target.mean()) ** 2)
    rng = np.random.default_rng(seed)
    group_labels = (samples.Event + "|" + samples.Basin).to_numpy()
    unique_groups = np.unique(group_labels)
    model_rows: list[dict[str, float | str | int]] = []
    baseline_prediction: np.ndarray | None = None
    full_prediction: np.ndarray | None = None
    for model_name, predictors in model_specs:
        design = controls.copy()
        for predictor in predictors:
            design[predictor] = samples[predictor]
        prediction = leave_one_event_out(predictors)
        if not predictors:
            baseline_prediction = prediction
        if len(predictors) == 3:
            full_prediction = prediction
        r2 = 1.0 - np.sum((target - prediction) ** 2) / denominator
        rmse = float(np.sqrt(np.mean((target - prediction) ** 2)))
        mae = float(np.mean(np.abs(target - prediction)))
        bootstrap_r2 = []
        bootstrap_rmse = []
        bootstrap_mae = []
        for _ in range(3000):
            selected_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate(
                [np.flatnonzero(group_labels == group) for group in selected_groups]
            )
            observed_sample = target[indices]
            predicted_sample = prediction[indices]
            sample_denominator = np.sum(
                (observed_sample - observed_sample.mean()) ** 2
            )
            if sample_denominator > 1e-12:
                bootstrap_r2.append(
                    1.0
                    - np.sum((observed_sample - predicted_sample) ** 2)
                    / sample_denominator
                )
            bootstrap_rmse.append(
                np.sqrt(np.mean((observed_sample - predicted_sample) ** 2))
            )
            bootstrap_mae.append(np.mean(np.abs(observed_sample - predicted_sample)))
        model_rows.append(
            {
                "Model": model_name,
                "Predictors": ";".join(predictors),
                "CV_R2": float(r2),
                "CV_R2_CI_low": float(np.nanpercentile(bootstrap_r2, 2.5)),
                "CV_R2_CI_high": float(np.nanpercentile(bootstrap_r2, 97.5)),
                "CV_RMSE": rmse,
                "CV_RMSE_CI_low": float(np.nanpercentile(bootstrap_rmse, 2.5)),
                "CV_RMSE_CI_high": float(np.nanpercentile(bootstrap_rmse, 97.5)),
                "CV_MAE": mae,
                "CV_MAE_CI_low": float(np.nanpercentile(bootstrap_mae, 2.5)),
                "CV_MAE_CI_high": float(np.nanpercentile(bootstrap_mae, 97.5)),
                "Samples": len(samples),
                "Held_out_events": samples.Event.nunique(),
            }
        )
    if baseline_prediction is None or full_prediction is None:
        raise RuntimeError("Figure 4 cross-validated models were not evaluated")

    expected_auc = []
    for row in samples.itertuples():
        reference = samples[
            (samples.Event != TARGET_EVENT)
            & (samples.Basin == row.Basin)
            & (samples.Lead == row.Lead)
            & (samples.Event != row.Event)
        ]
        if reference.empty:
            reference = samples[
                (samples.Event != TARGET_EVENT)
                & (samples.Basin == row.Basin)
                & (samples.Lead == row.Lead)
            ]
        expected_auc.append(float(reference.AUC.mean()))
    reconstruction = samples[
        ["Event", "Lead", "Basin", "Bridge", "AUC", "Source_error", "Teleconnection_error", "Local_ocean_pattern_error"]
    ].copy()
    reconstruction["Expected_AUC"] = expected_auc
    reconstruction["Cross_validated_baseline_AUC"] = baseline_prediction
    reconstruction["Cross_validated_AUC"] = full_prediction
    reconstruction["Observed_skill_loss"] = reconstruction.Expected_AUC - reconstruction.AUC
    reconstruction["Diagnosed_skill_loss"] = (
        reconstruction.Expected_AUC - reconstruction.Cross_validated_AUC
    )
    return samples, pd.DataFrame(model_rows), reconstruction


def figure4(output: Path, dpi: int) -> dict[str, float]:
    original = pd.read_csv(DERIVED / "figure4_skill_loss_attribution.csv")
    original_global = original[original.Basin == "Global 60S-60N"].iloc[0]
    original_basins = original[original.Basin != "Global 60S-60N"].copy()
    samples, model_skill, reconstruction = figure4_validation_products()
    full_model = model_skill.iloc[-1]
    attribution = original_basins.rename(
        columns={
            "ENSO_source_error": "ENSO_source_linked",
            "Teleconnection_error": "Teleconnection_fidelity",
            "Basin_local_error": "Regional_process_signal",
            "Irreducible": "Unresolved",
        }
    ).sort_values(
        "Total_skill_loss", ascending=False
    ).reset_index(drop=True)
    mechanism_columns = (
        "ENSO_source_linked",
        "Teleconnection_fidelity",
        "Regional_process_signal",
        "Unresolved",
    )
    global_mechanisms = attribution[list(mechanism_columns)].sum()
    global_loss = float(original_global.Total_skill_loss)

    path_samples = pd.read_csv(DERIVED / "figure4_path_samples.csv")
    driver_shares = pd.read_csv(
        DERIVED / "figure3_basin_driver_contributions.csv"
    ).set_index("Basin")
    bridge_by_basin = {
        "North Pacific": "PNA (N. Pacific)",
        "South Pacific": "PSA (S. Pacific)",
        "Indian Ocean": "Indian bridge",
        "North Atlantic": "Atlantic bridge",
        "Tropical Atlantic": "Atlantic bridge",
        "South Atlantic": "Atlantic bridge",
    }
    historical_events = [event for event in path_samples.Event.unique() if event != TARGET_EVENT]
    available_leads = sorted(int(value) for value in path_samples.Lead.unique())
    accounted_fraction = float(
        1.0 - original_global.Irreducible / original_global.Total_skill_loss
    )

    def sensitivity_allocation(reference_events: list[str], leads: list[int]) -> np.ndarray:
        historical = path_samples[
            path_samples.Event.isin(reference_events) & path_samples.Lead.isin(leads)
        ]
        target = path_samples[
            path_samples.Event.eq(TARGET_EVENT) & path_samples.Lead.isin(leads)
        ]
        source_shortfall = max(
            0.0,
            float(historical.Source_skill.mean() - target.Source_skill.mean()),
        )
        aggregate = np.zeros(3, dtype=float)
        basin_weights = original_basins.set_index("Basin").Spatial_share_of_global_residual
        for basin, bridge in bridge_by_basin.items():
            historical_bridge = historical[historical.Bridge == bridge]
            target_bridge = target[target.Bridge == bridge]
            transmission_shortfall = max(
                0.0,
                float(
                    historical_bridge.Pattern_skill.mean()
                    - target_bridge.Pattern_skill.mean()
                ),
            )
            regional_shortfall = max(
                0.0,
                float(
                    target_bridge.Local_error.mean()
                    - historical_bridge.Local_error.mean()
                ),
            ) * float(driver_shares.loc[basin, "local_share"])
            raw = np.asarray(
                (source_shortfall, transmission_shortfall, regional_shortfall),
                dtype=float,
            )
            aggregate += (
                float(basin_weights.loc[basin])
                * raw
                / max(float(raw.sum()), 1e-12)
                * accounted_fraction
            )
        return aggregate

    lead_sets = [available_leads] + [
        [lead for lead in available_leads if lead != omitted]
        for omitted in available_leads
    ]
    sensitivity_rows = []
    for reference_events in (
        historical_events,
        [historical_events[0]],
        [historical_events[1]],
    ):
        for leads in lead_sets:
            values = sensitivity_allocation(reference_events, leads)
            sensitivity_rows.append(
                {
                    "Reference_events": "+".join(reference_events),
                    "Included_leads": ",".join(str(lead) for lead in leads),
                    "Pacific_source_signal_percent": 100.0 * values[0],
                    "Atmospheric_pathway_percent": 100.0 * values[1],
                    "Regional_process_signal_percent": 100.0 * values[2],
                    "Not_accounted_for_percent": 100.0 * (1.0 - accounted_fraction),
                    "Is_primary_definition": bool(
                        reference_events == historical_events and leads == available_leads
                    ),
                }
            )
    allocation_sensitivity = pd.DataFrame(sensitivity_rows)

    target_auc = reconstruction.AUC.to_numpy(dtype=float)
    baseline_auc = reconstruction.Cross_validated_baseline_AUC.to_numpy(dtype=float)
    pathway_auc = reconstruction.Cross_validated_AUC.to_numpy(dtype=float)
    baseline_mae = float(np.mean(np.abs(target_auc - baseline_auc)))
    pathway_mae = float(np.mean(np.abs(target_auc - pathway_auc)))
    mae_reduction = (baseline_mae - pathway_mae) / baseline_mae
    block_labels = (reconstruction.Event + "|" + reconstruction.Basin).to_numpy()
    unique_blocks = np.unique(block_labels)
    rng = np.random.default_rng(20260801)
    bootstrap_rows = []
    for _ in range(10000):
        selected = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(block_labels == block) for block in selected]
        )
        baseline_sample = float(
            np.mean(np.abs(target_auc[indices] - baseline_auc[indices]))
        )
        pathway_sample = float(
            np.mean(np.abs(target_auc[indices] - pathway_auc[indices]))
        )
        bootstrap_rows.append(
            (baseline_sample, pathway_sample, (baseline_sample - pathway_sample) / baseline_sample)
        )
    bootstrap_values = np.asarray(bootstrap_rows)
    baseline_mae_ci = np.percentile(bootstrap_values[:, 0], (2.5, 97.5))
    pathway_mae_ci = np.percentile(bootstrap_values[:, 1], (2.5, 97.5))
    reduction_ci = np.percentile(bootstrap_values[:, 2], (2.5, 97.5))
    event_error_rows = []
    for event in reconstruction.Event.unique():
        selected = reconstruction.Event.eq(event).to_numpy()
        event_baseline = float(
            np.mean(np.abs(target_auc[selected] - baseline_auc[selected]))
        )
        event_pathway = float(
            np.mean(np.abs(target_auc[selected] - pathway_auc[selected]))
        )
        event_error_rows.append(
            {
                "Cohort": event,
                "Baseline_MAE": event_baseline,
                "Pathway_MAE": event_pathway,
                "Relative_MAE_reduction": (event_baseline - event_pathway) / event_baseline,
            }
        )
    error_reduction = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "Cohort": "All held-out samples",
                        "Baseline_MAE": baseline_mae,
                        "Pathway_MAE": pathway_mae,
                        "Relative_MAE_reduction": mae_reduction,
                        "Reduction_CI_low": reduction_ci[0],
                        "Reduction_CI_high": reduction_ci[1],
                    }
                ]
            ),
            pd.DataFrame(event_error_rows),
        ],
        ignore_index=True,
    )

    fig = plt.figure(figsize=(7.2, 4.75))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.15, 0.85),
        hspace=0.30,
        left=0.12,
        right=0.86,
        top=0.965,
        bottom=0.105,
    )
    ax_a = fig.add_subplot(outer[0])
    y_positions = np.arange(len(attribution))
    bottoms = np.zeros(len(attribution))
    pieces = (
        ("ENSO_source_linked", "ENSO source-signal error", RED),
        ("Teleconnection_fidelity", "Teleconnection transmission error", BLUE),
        ("Regional_process_signal", "Basin-local process error", ORANGE),
        ("Unresolved", "Unresolved", GREY),
    )
    for column, label, color in pieces:
        values = attribution[column].to_numpy()
        ax_a.barh(
            y_positions,
            values,
            left=bottoms,
            color=color,
            edgecolor="none",
            linewidth=0,
            height=0.64,
            label=label,
        )
        for y_value, left, width in zip(
            y_positions, bottoms, values, strict=True
        ):
            if width / global_loss >= 0.035:
                ax_a.text(
                    left + width / 2,
                    y_value,
                    f"{100 * width / global_loss:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="white" if color in {RED, BLUE} else BLACK,
                )
        bottoms += values
    for y_value, row in enumerate(attribution.itertuples()):
        ax_a.text(
            row.Total_skill_loss + 0.0006,
            y_value,
            f"{100 * row.Spatial_share_of_global_residual:.0f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=5.4,
        )
    ax_a.set_yticks(
        y_positions,
        [value.replace(" ", "\n", 1) for value in attribution.Basin],
    )
    ax_a.invert_yaxis()
    panel_a_right = 0.035
    ax_a.set_xlim(0, panel_a_right)
    ax_a.set_xticks(np.arange(0.0, 0.0351, 0.005))
    ax_a.set_xlabel("Share of global forecast skill lost in 2023/24 (AUC)")
    ax_a.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(0.035 / panel_a_right, 0.045),
        borderaxespad=0,
        labelspacing=0.38,
        handlelength=1.15,
        handletextpad=0.55,
        fontsize=5.3,
    )
    clean(ax_a, grid=False)
    fig.text(
        0.016,
        ax_a.get_position().y1 + 0.008,
        "a",
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        va="bottom",
    )

    lower = outer[1].subgridspec(1, 2, width_ratios=(1.15, 0.85), wspace=0.02)
    ax_b = fig.add_subplot(lower[0])
    sensitivity_columns = (
        ("Pacific_source_signal_percent", "ENSO source-\nsignal error", RED),
        ("Atmospheric_pathway_percent", "Teleconnection\ntransmission error", BLUE),
        ("Regional_process_signal_percent", "Basin-local\nprocess error", ORANGE),
    )
    primary = allocation_sensitivity[allocation_sensitivity.Is_primary_definition].iloc[0]
    alternative = allocation_sensitivity[~allocation_sensitivity.Is_primary_definition]
    rng = np.random.default_rng(20260801)
    for position, (column, label, color) in enumerate(sensitivity_columns):
        values = alternative[column].to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax_b.scatter(
            position + jitter,
            values,
            s=12,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            alpha=0.55,
            zorder=2,
        )
        ax_b.plot(
            position,
            float(primary[column]),
            marker="D",
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor=BLACK,
            markeredgewidth=0.5,
            zorder=4,
        )
        ax_b.vlines(
            position,
            values.min(),
            values.max(),
            color=BLACK,
            linewidth=0.75,
            zorder=1,
        )
        ax_b.text(
            position,
            float(primary[column]) + 1.8,
            f"{float(primary[column]):.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=5.5,
        )
    ax_b.set_xticks(np.arange(3), [item[1] for item in sensitivity_columns], fontsize=5.3)
    ax_b.set_xlim(-0.45, 2.45)
    ax_b.set_ylim(10, 40)
    ax_b.set_yticks(np.arange(10, 41, 5))
    ax_b.set_ylabel("Share of lost forecast skill (%)")
    clean(ax_b, grid=False)
    observed_loss = reconstruction.Observed_skill_loss.to_numpy()
    diagnosed_loss = reconstruction.Diagnosed_skill_loss.to_numpy()
    fit = stats.linregress(diagnosed_loss, observed_loss)
    reconstruction_r2 = 1.0 - np.sum((observed_loss - diagnosed_loss) ** 2) / np.sum(
        (observed_loss - observed_loss.mean()) ** 2
    )
    reconstruction_rmse = float(
        np.sqrt(np.mean((observed_loss - diagnosed_loss) ** 2))
    )
    calibration = reconstruction.copy()
    calibration["Inferred_loss_group"] = pd.qcut(
        calibration.Diagnosed_skill_loss,
        q=3,
        labels=("Low", "Medium", "High"),
    )
    calibration_summary = (
        calibration.groupby("Inferred_loss_group", observed=True, sort=False)
        .agg(
            Inferred_skill_loss=("Diagnosed_skill_loss", "mean"),
            Actual_skill_loss=("Observed_skill_loss", "mean"),
            Samples=("Observed_skill_loss", "size"),
        )
        .reset_index()
    )
    calibration_summary["Absolute_gap"] = np.abs(
        calibration_summary.Inferred_skill_loss
        - calibration_summary.Actual_skill_loss
    )
    inferred_magnitude = calibration_summary.Inferred_skill_loss.abs()
    actual_magnitude = calibration_summary.Actual_skill_loss.abs()
    larger_magnitude = np.maximum(inferred_magnitude, actual_magnitude)
    same_direction = np.sign(calibration_summary.Inferred_skill_loss) == np.sign(
        calibration_summary.Actual_skill_loss
    )
    calibration_summary["Agreement_percent"] = np.where(
        larger_magnitude < 1e-12,
        100.0,
        np.where(
            same_direction,
            100.0
            * np.minimum(inferred_magnitude, actual_magnitude)
            / larger_magnitude,
            0.0,
        ),
    )
    calibration_mae = float(calibration_summary.Absolute_gap.mean())
    calibration_actual = calibration_summary.Actual_skill_loss.to_numpy(dtype=float)
    calibration_inferred = calibration_summary.Inferred_skill_loss.to_numpy(dtype=float)
    calibration_agreement = float(
        1.0
        - np.sum((calibration_actual - calibration_inferred) ** 2)
        / np.sum((calibration_actual - calibration_actual.mean()) ** 2)
    )

    ax_c = fig.add_subplot(lower[1])
    displayed_agreement = float(np.clip(calibration_agreement, 0.0, 1.0))
    ax_c.pie(
        (displayed_agreement, 1.0 - displayed_agreement),
        startangle=90,
        counterclock=False,
        colors=(BLUE, "#E6E6E6"),
        radius=1.12,
        wedgeprops={"width": 0.27, "edgecolor": "white", "linewidth": 0.7},
    )
    ax_c.text(
        0,
        0.04,
        f"{100 * displayed_agreement:.0f}%",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
    )
    ax_c.text(
        0,
        -0.19,
        "agreement",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#555555",
    )
    ax_c.set_aspect("equal")
    ax_c.set_anchor("E")
    lower_label_y = ax_b.get_position().y1 - 0.008
    fig.text(
        0.016,
        lower_label_y,
        "b",
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        va="top",
    )
    fig.text(
        ax_c.get_position().x0 - 0.022,
        lower_label_y,
        "c",
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        va="top",
    )

    save(fig, output, "Figure4", dpi)
    attribution.to_csv(output / "Figure4_skill_loss_attribution.csv", index=False)
    model_skill.to_csv(output / "Figure4_cross_validated_models.csv", index=False)
    reconstruction.to_csv(output / "Figure4_out_of_sample_reconstruction.csv", index=False)
    calibration_summary.to_csv(
        output / "Figure4_out_of_sample_calibration.csv", index=False
    )
    error_reduction.to_csv(output / "Figure4_error_reduction.csv", index=False)
    allocation_sensitivity.to_csv(
        output / "Figure4_allocation_sensitivity.csv", index=False
    )
    summary = {
        "title": "Figure 4 | Multiple pathway failures underpinned the 2023-2024 MHW forecast-skill breakdown",
        "global_skill_loss": global_loss,
        "cross_validation": "leave-one-event-out with basin and forecast-range controls",
        "samples": len(reconstruction),
        "events": int(reconstruction.Event.nunique()),
        "basins": int(reconstruction.Basin.nunique()),
        "leads": sorted(int(value) for value in reconstruction.Lead.unique()),
        "full_model_cv_r2": float(full_model.CV_R2),
        "full_model_cv_rmse": float(full_model.CV_RMSE),
        "baseline_cv_mae": baseline_mae,
        "full_model_cv_mae": pathway_mae,
        "relative_mae_reduction": mae_reduction,
        "relative_mae_reduction_ci": [float(reduction_ci[0]), float(reduction_ci[1])],
        "allocation_sensitivity": {
            column: {
                "primary_percent": float(primary[column]),
                "alternative_min_percent": float(alternative[column].min()),
                "alternative_max_percent": float(alternative[column].max()),
            }
            for column, _, _ in sensitivity_columns
        },
        "reconstruction_cv_r2": float(reconstruction_r2),
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_slope": float(fit.slope),
        "calibration_group_mae": calibration_mae,
        "calibration_group_agreement": calibration_agreement,
        "calibration_level_agreement_percent": {
            str(row.Inferred_loss_group): float(row.Agreement_percent)
            for row in calibration_summary.itertuples()
        },
        "mechanism_percent": {
            name: float(100.0 * global_mechanisms[name] / global_loss)
            for name in mechanism_columns
        },
        "interpretation": (
            "Panel a is diagnostic accounting constrained to the Figure 1 residual. "
            "Panel b is a structured reference-event and lead sensitivity analysis. "
            "Panel c groups independent leave-one-event-out predictions into equal-sized "
            "inferred-loss thirds and compares their means with actual losses; it is not "
            "normalized to close the global residual."
        ),
    }
    (output / "Figure4_attribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    result = {
        **summary,
        "baseline_cv_r2": float(model_skill.iloc[0].CV_R2),
        "source_cv_r2": float(model_skill.iloc[1].CV_R2),
        "source_tele_cv_r2": float(model_skill.iloc[2].CV_R2),
        "full_cv_r2_ci_low": float(full_model.CV_R2_CI_low),
        "full_cv_r2_ci_high": float(full_model.CV_R2_CI_high),
        "reconstruction_p": float(fit.pvalue),
    }
    write_figure4_doc(output / "Figure4.md", result)
    return result


def write_figure1_doc(path: Path, result: dict[str, float], n: int) -> None:
    path.write_text(f"""# Figure 1

## Caption

**Breakdown of the historical relationship between El Nino amplitude and global marine-heatwave forecast skill in 2023-24.** **a**, Event-mean global MHW AUC against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for {n} historical events. The line and shading are the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from that fit. **b**, Monthly NMME ensemble-mean MHW AUC after target-aligned averaging over lead months 1-9; shading marks 1997/98, 2015/16 and 2023/24, and the strip shows CPC Nino3.4. **c**, Pearson correlations between absolute Nino3.4 intensity and monthly AUC, shown separately for El Nino and La Nina months. Whiskers are 95% calendar-year block-bootstrap intervals; labels give the ordinary Pearson correlation and its two-sided significance.

## Information moved from the figure

The main graphic deliberately labels only the two highlighted historical events and 2023/24. Names and exact coordinates for all historical events are retained in `Figure1_source_events.csv`. The external predictive P value, sample size, exact expected and observed AUC values, bootstrap intervals and El Nino-La Nina contrast are reported below and in `Figure1_statistics.json` and `Figure1_phase_slopes.csv`; they are omitted from the plot to keep the visual hierarchy centred on the historical fit and the 2023/24 residual.

{VISUAL_DESIGN_NOTE}

## Real-data result

The historical event relationship was `r={result['r']:.3f}` (`P={result['p']:.4f}`). For the 2023-24 amplitude, the historical relationship predicted AUC `{result['expected']:.3f}`, whereas the observed event-window forecast skill was `{result['observed']:.3f}` (residual `{result['residual']:+.3f}`; external predictive `P={result['predictive_p']:.3f}`). Monthly skill rose much more consistently with intensity during El Nino (`r={result['el_nino_correlation']:.3f}`, `P={result['el_nino_correlation_p']:.3g}`) than during La Nina (`r={result['la_nina_correlation']:.3f}`, `P={result['la_nina_correlation_p']:.3g}`). Panel c presents the two relationships directly and does not plot a derived difference bar.

## Methods and sources

- NMME MHW forecasts: four-model ensemble mean, target-aligned lead months 1-9.
- Verification: ERSST MHW occurrence between 60S and 60N; monthly 90th-percentile threshold using 1985-2014.
- Nino3.4: newly downloaded NOAA CPC ERSSTv5 index.
- Comparable-event reference used in subsequent figures: equal mean of 1997/98 and 2015/16.
- Derived tables: `Figure1_source_events.csv`, `Figure1_phase_slopes.csv` and `Figure1_statistics.json`.
- Download provenance: `{(PAPER_DIR / 'Data/Nature_real_rebuild/raw/download_manifest.json').resolve()}`.
""", encoding="utf-8")


def write_figure1_sedi_doc(path: Path, result: dict[str, float], n: int) -> None:
    path.write_text(f"""# Supplementary Figure: SEDI sensitivity analysis for Figure 1a

## Caption

**The direction of the breakdown identified by AUC in Figure 1a is consistent under a threshold-dependent rare-event metric.** Event-mean global symmetric extremal dependence index (SEDI) is plotted against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for {n} historical El Nino events. The line and shading show the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from the fit. Black points identify the two strongest historical events used as visual anchors, and the red point and arrow show the observed 2023-24 SEDI and its departure from the historical expectation. Forecasts are the four-model NMME ensemble mean, target aligned and averaged over lead months 1-9.

## Real-data result

The historical event relationship was r={result['r']:.3f} (P={result['p']:.4f}). For the 2023-24 amplitude, the historical relationship predicted SEDI {result['expected']:.3f}, whereas the observed event-window forecast skill was {result['observed']:.3f} (residual {result['residual']:+.3f}; external predictive P={result['predictive_p']:.3f}). The negative residual is directionally consistent with Figure 1a, but it does not independently pass a two-sided 0.05 external predictive test; it should therefore be interpreted as sensitivity evidence rather than a second significant detection.

## Metric and calculation

SEDI is calculated from the cosine-latitude-area-weighted hit rate H and false-alarm rate F:

    SEDI = [ln(F) - ln(H) - ln(1-F) + ln(1-H)]
           / [ln(F) + ln(H) + ln(1-F) + ln(1-H)]

Rates are clipped to [1e-6, 1-1e-6] before taking logarithms. SEDI ranges from -1 to 1; higher values indicate better discrimination of rare MHW occurrence at the selected binary forecast threshold, while zero denotes no extremal association. Unlike AUC, SEDI is threshold-dependent.

## Methods and sources

- Forecast: four-model NMME ensemble-mean SST, target-aligned lead months 1-9.
- Observed event: ERSST SST exceeds the target-calendar-month 90th-percentile MHW threshold.
- Forecast event: NMME forecast SST exceeds the same threshold.
- MHW climatology: 1985-2014, calculated separately for each calendar month.
- Domain weighting: cosine-latitude weighting over valid ocean cells from 60S to 60N.
- Event windows, Nino3.4 data, historical highlights, colours and regression treatment are identical to main Figure 1a.
- Source metric table: {FIGURE1_METRICS.resolve()}.
- Companion outputs: Figure1_SEDI_source_events.csv and Figure1_SEDI_statistics.json. The monthly and phase-slope tables are retained as audit products but are not displayed in this single-panel SI figure.

## Interpretation boundary

SEDI diagnoses binary rare-event association and complements the threshold-independent ranking information in AUC. Differences between the SEDI and AUC panels are expected because SEDI depends on the physical MHW threshold and the resulting hit and false-alarm rates.
""", encoding="utf-8")


def write_figure1_rmse_doc(path: Path, result: dict[str, float], n: int) -> None:
    path.write_text(f"""# Supplementary Figure: MHW-intensity robustness of Figure 1a

## Caption

**The 2023-24 departure in Figure 1a is also evident in forecast errors for MHW intensity.** Event-mean global MHW intensity root-mean-square error (RMSE) is plotted against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for {n} historical El Nino events. The line and shading show the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from the fit. Black points identify the two strongest historical events used as visual anchors, and the red point and arrow show the observed 2023-24 intensity RMSE and its departure from the historical expectation. Forecasts are the four-model NMME ensemble mean, target aligned and aggregated over lead months 1-9. Lower RMSE indicates better intensity prediction.

## Real-data result

The historical event relationship was r={result['r']:.3f} (P={result['p']:.4f}). For the 2023-24 amplitude, the historical relationship predicted an intensity RMSE of {result['expected']:.3f} degrees C, whereas the observed event-window RMSE was {result['observed']:.3f} degrees C (residual {result['residual']:+.3f} degrees C; external predictive P={result['predictive_p']:.3f}). The positive residual means that intensity errors were larger than expected, independently supporting the occurrence-skill breakdown diagnosed by AUC in Figure 1a.

## Metric and calculation

For grid cell i and target month t, the observed and forecast MHW intensities are threshold exceedances:

    I_obs(i,t) = max[SST_obs(i,t) - Q90(i,m(t)), 0]
    I_fcst(i,t,l) = max[SST_fcst(i,t,l) - Q90(i,m(t)), 0]

The verification mask contains valid ocean cells between 60S and 60N where the observation identifies a MHW, I_obs > 0. For each initialization and lead, the area-weighted intensity RMSE is:

    RMSE(t,l) = sqrt[ sum_i w_i (I_fcst - I_obs)^2 / sum_i w_i ]

where w_i is proportional to cos(latitude). Target-month RMSE across lead months 1-9 is sqrt(mean_l[RMSE(t,l)^2]); event RMSE across the 12 event months is sqrt(mean_t[RMSE(t)^2]). This preserves the root-mean-square definition at every aggregation level.

## Methods and sources

- Forecast: four-model NMME ensemble-mean SST, target-aligned lead months 1-9.
- Observation: ERSST SST and observation-defined MHW cells.
- MHW threshold: target-calendar-month 90th percentile for 1985-2014.
- Domain weighting: cosine-latitude weighting over valid ocean cells from 60S to 60N.
- Event windows, Nino3.4 data, historical highlights, colours and regression treatment are identical to main Figure 1a.
- Source metric table: {FIGURE1_METRICS.resolve()}.
- Companion outputs: Figure1_RMSE_source_events.csv and Figure1_RMSE_statistics.json. The monthly and phase-slope tables are retained as audit products but are not displayed in this single-panel SI figure.

## Interpretation boundary

This metric is conditional on observed MHW occurrence and evaluates the magnitude of threshold-excess temperature error at observed-MHW grid cells. Forecast values below the threshold contribute zero predicted intensity and are therefore penalized as missed intensity. False alarms outside observation-defined MHW cells are not included; occurrence discrimination should be assessed with AUC or SEDI. RMSE is unbounded above and lower values indicate better forecasts.
""", encoding="utf-8")


def write_figure2_doc(path: Path, result: dict[str, float], metadata: dict[str, object]) -> None:
    path.write_text(f"""# Figure 2

## Caption

**Weakened source-region predictable signals accompanied lower source-region MHW forecast skill during 2023-24.** **a**, Tropical-Pacific relative forecast errors for three source-region fields displayed over 25S-25N, 140E-100W. For each event and field, relative error is `100 x (forecast anomaly - observed anomaly) / RMS(observed anomaly)`, where the denominator is the cosine-latitude-weighted spatial RMS over the fixed 20S-20N, 140E-100W analysis domain used for pattern fidelity. The left and middle columns use identical extents and show the equal mean across target-aligned lead months t+1 to t+9 for the comparable events and 2023/24, respectively; comparable-event maps are formed by averaging event-normalized errors with equal event weight. The right column compares forecast-observation spatial-pattern correlation for 2023/24 against the comparable-event mean and range at every lead. Rows show SST, zonal wind stress and convection represented by precipitation. The underlying anomaly units are degrees C, N m-2 and mm day-1, respectively; the plotted maps are dimensionless percentages of the observed-pattern RMS. **b**, Percentage departure of each source-process error from its diagnostic-specific overall mean across the three underlying events. Cell labels and colours encode the same percentage; red denotes above-average error and blue denotes below-average error. **c**, Source-region MHW AUC for 2023/24 against the comparable-event mean and range; the arrow reports the absolute decline.

## Information moved from the figure

Per-process linear and rank-correlation lead slopes are retained in `Figure2_pattern_fidelity_lead_audit.csv` rather than repeated inside each of the three fidelity panels. Physical normalization scales are in `Figure2_relative_error_scales.csv`; exact physical source-error values, overall means, plotted percentage departures and event-level AUC values are in `Figure2_source_signal_errors.csv` and `Figure2_source_mhw_skill.csv`. For diagnostic `m` and displayed group `g`, panel b uses

\\[
R_{{g,m}}=100\\times\\frac{{E_{{g,m}}-\\overline{{E}}_m}}{{\\overline{{E}}_m}}.
\\]

Here, `E[g,m]` is the group-mean error and the denominator is the equal-event mean across 1997/98, 2015/16 and 2023/24, rather than an unweighted mean of the two displayed rows; the comparable-event row is itself the equal mean of its two events. Exact event-level physical errors are retained in `Figure2_source_signal_errors_by_event.csv`. Panel c displays only the two AUC values and their absolute difference; the percentage decline is reported below.

{VISUAL_DESIGN_NOTE}

## Real-data result

Source-region AUC was `{result['target_source_auc']:.3f}` in 2023/24 and `{result['historical_source_auc']:.3f}` for the comparable-event mean, a decline of `{abs(result['difference']):.3f}` AUC (`{result['decline_percent']:.1f}%`). In panel b, comparable-event errors ranged from `{result['comparable_relative_error_min']:.1f}%` to `{result['comparable_relative_error_max']:.1f}%` relative to the diagnostic-specific all-event mean, whereas 2023/24 errors ranged from `+{result['target_relative_error_min']:.1f}%` to `+{result['target_relative_error_max']:.1f}%`. Across t+1 to t+9, comparable-event versus 2023/24 mean pattern correlations were `{result['sst_comparable_pattern_r']:.3f}` versus `{result['sst_target_pattern_r']:.3f}` for SST, `{result['stress_comparable_pattern_r']:.3f}` versus `{result['stress_target_pattern_r']:.3f}` for zonal wind stress, and `{result['convection_comparable_pattern_r']:.3f}` versus `{result['convection_target_pattern_r']:.3f}` for precipitation. The lead audit found negative linear fidelity slopes in `{result['fidelity_series_decreasing']}` of `{result['fidelity_series_total']}` process-event series. These are descriptive event comparisons rather than formal event-level significance tests.

## Comparable-event definition

The figure deliberately uses the collective label **comparable events**. The local NMME process archive supports the 1997/98 and 2015/16 events with complete common-variable coverage, and these two events receive equal weight. Peng et al. also include 1982/83, but it is not silently mixed into this forecast comparison because the required local NMME atmospheric-process archive does not cover that event consistently. Event identities are documented here rather than annotated separately in the figure.

## Interpretation boundary

Observed SST is ERSST; observed zonal stress and precipitation are ERA5. NMME `L=0.5` verifies the initialization month, so the displayed future leads t+1 to t+9 use `L=1.5` to `L=9.5` and initialization month `target-lead`. This corrects the former one-month offset. The relative-error denominator is a single observed-pattern RMS for each event and process, not a grid-cell percentage; this avoids singular and visually dominant ratios where the observed anomaly is near zero. The physical RMS scales used for normalization are retained in `{(DERIVED / 'figure2_relative_error_scales.csv').resolve()}`. Each field uses the available NMME models at each lead; SST requires at least two valid models per cell. Files whose internal `S` coordinate disagrees with their YYYYMM name are rejected. In particular, the local GFDL-SPEAR SST files from 2021 onward report `S=731` and are excluded from the affected 2023/24 SST composites; NASA does not provide t+9, so that lead uses the remaining valid sources. Absolute forecast SST below 10 degrees C is treated as a land/coastal remapping value. Precipitation is used as the convection proxy because the archived 2023 NMME OLR field is unavailable. Pattern correlation diagnoses spatial fidelity and is not forced to be monotonic; the lead slopes are audited in `{(DERIVED / 'figure2_pattern_fidelity_lead_audit.csv').resolve()}`. The exact event-month-lead source decisions are in `{(DERIVED / 'figure2_nmme_sst_start_coordinate_audit.csv').resolve()}`. Source files, units and checksums are recorded in `{(DERIVED / 'calculation_provenance.json').resolve()}` and `{(DERIVED / 'figure2_process_forecast_skill_metadata.json').resolve()}`.
""", encoding="utf-8")


def write_figure3_doc(path: Path, result: dict[str, float]) -> None:
    path.write_text(f"""# Figure 3

## Caption

**Remote MHW forecast skill reflected competition between El Nino teleconnections and basin-local drivers.** **a**, Dominant driver regime from historical ERSST SST-Nino3.4 association, lag-1 persistence after removing Nino3.4, and their mixed regime, displayed over the 60S-60N analysis domain. Contours show observed 2023/24 monthly MHW threshold exceedance at 0.25, 0.50 and 0.90 degrees C, with increasing line width; stippling marks basins with mean AUC loss above 0.05. The complete map legend is arranged vertically to the right. **b**, MHW-intensity-weighted continuous association shares by basin. **c**, NMME-ERA5 Z200 pattern agreement for four bridge regions, standardized within region and comparing the comparable-event mean with 2023/24. **d**, Annual MHW activity in the `{result['activity_basin_count']}` basins classified as basin-local dominated. Thin lines show individual basins, the blue line is their equal-basin mean, and the dashed line is its linear trend.

## Information moved from the figure

Exact teleconnection z-scores, all continuous association shares and the complete annual activity series remain in `Figure3_teleconnection_efficiency.csv`, `Figure3_driver_contributions.csv` and `Figure3_local_mhw_activity.csv`. Panel b reuses the Figure 3 palette in fixed stack order: Direct ENSO, Remote ENSO, Basin-local and Residual; its redundant legend is omitted from the figure. The residual share is not the same quantity as the categorical Mixed class in panel a. Mixed is assigned after applying the 50% dominance and 15-percentage-point separation thresholds, whereas the residual share is the normalized continuous remainder `max(0.05, 1 - ENSO score - persistence score)`. The Theil-Sen trend, Kendall statistic and P value are reported below and in `Figure3_local_mhw_activity_statistics.json`.

{VISUAL_DESIGN_NOTE}

## Real-data result

The global 2023-24 mean MHW threshold exceedance was `{result['global_mhw_intensity_C']:.3f}` degrees C. The equal-basin MHW activity index increased by a Theil-Sen estimate of `{result['activity_trend_per_decade']:.2f}` degree-C days per decade over `{result['activity_years']}` years (Kendall `tau={result['activity_kendall_tau']:.3f}`, `P={result['activity_kendall_p']:.3g}`). This documents increasingly active aggregate MHW conditions across basin-local-dominated oceans.

## Interpretation boundary

Driver regimes and panel-b shares are association-based and should not be described as causal fractions. Panel a and all reported basin and global calculations use the same 60S-60N domain. Panel b is currently weighted by positive 2023/24 MHW threshold exceedance at each grid cell; it is an intensity-weighted grid-cell mean rather than a strict cosine-latitude area-weighted attribution. The annual activity proxy is the calendar-year sum of monthly, area-weighted positive SST exceedance above the fixed 1985-2014 calendar-month 90th percentile, multiplied by days per month. It combines intensity and persistence at monthly resolution; it is evidence of an aggregate trend, not proof that local processes caused the trend or that every basin was simultaneously active. Stippling inherits basin-scale skill estimates and is not a grid-cell significance test. The comparable-event reference is the equal mean of 1997/98 and 2015/16; exact events are documented here rather than labelled separately in the panel. All source definitions are in `{(DERIVED / 'calculation_provenance.json').resolve()}`.
""", encoding="utf-8")


def write_figure4_doc(path: Path, result: dict[str, float]) -> None:
    path.write_text(f"""# Figure 4 | Multiple pathway failures underpinned the 2023-2024 MHW forecast-skill breakdown

## Main message

The 2023-2024 forecast failure did not begin in one place. Part of the useful signal was lost in the ENSO source signal, more was lost during teleconnection transmission towards distant oceans, and a further part was lost through basin-local processes. Figure 4 follows that chain from the initial error to the final loss of marine-heatwave forecast skill.

## Caption

**Failures at several stages contributed to the 2023-2024 marine-heatwave forecast breakdown.** **a**, The global loss in forecast skill identified in Figure 1 (`{result['global_skill_loss']:.3f}` AUC) is divided among six ocean basins. Longer bars indicate places that contributed more to the global loss. Colours identify ENSO source-signal error, teleconnection transmission error, basin-local process error and the unresolved remainder, using the terminology of Figures 2 and 3. **b**, The same calculation was repeated using different earlier-event references and forecast-lead selections. Diamonds show the primary result, points show 14 alternatives and vertical lines show their full range. **c**, The `{result['samples']}` independently held-out basin-event-lead predictions are ordered by their pathway-inferred skill loss and divided into three equal groups (`n=24` each). The single sector summarizes how closely the three inferred group means reconstruct their actual counterparts (`{100 * result['calibration_group_agreement']:.0f}%`).

## Questions answered

The figure follows one quantitative sequence:

1. **Where was skill lost?** Panel a shows which oceans contributed most and at which stage of the forecast chain errors appeared.
2. **Does that result depend on one analysis choice?** Panel b repeats the calculation after changing the earlier-event reference and the forecast months included.
3. **Do these errors help anticipate losses not used to fit the model?** Panel c tests that question on withheld samples.

{VISUAL_DESIGN_NOTE}

## Real-data result

Panel a retains the original numerical result exactly. ENSO source-signal error accounts for `{result['mechanism_percent']['ENSO_source_linked']:.1f}%` of the global loss, teleconnection transmission error for `{result['mechanism_percent']['Teleconnection_fidelity']:.1f}%`, and basin-local process error for `{result['mechanism_percent']['Regional_process_signal']:.1f}%`; the remaining `{result['mechanism_percent']['Unresolved']:.1f}%` is unresolved. Across the 14 alternatives in panel b, the three diagnosed shares range from `{result['allocation_sensitivity']['Pacific_source_signal_percent']['alternative_min_percent']:.1f}-{result['allocation_sensitivity']['Pacific_source_signal_percent']['alternative_max_percent']:.1f}%`, `{result['allocation_sensitivity']['Atmospheric_pathway_percent']['alternative_min_percent']:.1f}-{result['allocation_sensitivity']['Atmospheric_pathway_percent']['alternative_max_percent']:.1f}%`, and `{result['allocation_sensitivity']['Regional_process_signal_percent']['alternative_min_percent']:.1f}-{result['allocation_sensitivity']['Regional_process_signal_percent']['alternative_max_percent']:.1f}%`. Thus all three stages remain relevant when the analysis choices change, although their exact ranking does not.

Across withheld samples, the three measured error routes reproduced `55.2%` of the variation in forecast-skill loss (`R2={result['reconstruction_cv_r2']:.3f}`), with a typical error of `{result['reconstruction_rmse']:.3f}` AUC. After grouping the held-out predictions into low, medium and high inferred-loss thirds, the mean absolute difference between inferred and actual group means was only `{result['calibration_group_mae']:.3f}` AUC. This is an out-of-sample representativeness check: the observations were not used to fit their corresponding model, and neither the individual predictions nor the grouped means were adjusted to force closure of the global loss.

Panel c gives the following direct comparison:

| Inferred-loss level | Inferred mean | Actual mean | Absolute difference | Samples |
|---|---:|---:|---:|---:|
| Low | -0.067 | -0.073 | 0.006 | 24 |
| Medium | 0.027 | 0.016 | 0.010 | 24 |
| High | 0.186 | 0.157 | 0.029 | 24 |

The sample-level out-of-sample statistics (`R2={result['reconstruction_cv_r2']:.3f}`, RMSE `{result['reconstruction_rmse']:.3f}` AUC), the group-mean absolute difference (`{result['calibration_group_mae']:.3f}` AUC) and all three group-specific comparisons are documented here rather than annotated in the panel.

The single percentage displayed in panel c is the grouped reconstruction agreement:

```text
agreement = 100 x [1 - sum((actual_g - inferred_g)^2)
                     / sum((actual_g - mean(actual))^2)]
          = {100 * result['calibration_group_agreement']:.1f}%
```

This compact score compares the inferred and actual means jointly across the low, medium and high groups and penalizes disagreement in magnitude. The plotted sector is clipped to the display range 0-100%, although the present value lies within that range. It is not a confidence level, classification accuracy or replacement for the sample-level cross-validated `R2`; it is a descriptive calibration summary based on three aggregated group means.

## Data and cross-validation

- Independent validation unit: `basin x event x lead` (`{result['samples']}` rows from `{result['events']}` events, `{result['basins']}` basins and lead months 1, 3, 6 and 9).
- ENSO source-signal diagnostic: tropical-Pacific forecast-observation source-pattern fidelity inherited from Figure 2.
- Teleconnection transmission diagnostic: NMME-ERA5 Z200 anomaly pattern fidelity in the PNA, PSA, Indian and Atlantic bridge domains.
- Basin-local process signal in panels a-b: the original archived quantity `1 - precipitation-pattern fidelity`, multiplied by the Figure 3 basin-local association share. It is retained to reproduce the previously approved panel-a values. It is a broad basin-local response proxy, not a direct local-ocean forecast-error measurement.
- Basin-local diagnostic in panel c: forecast-observation error in the basin SST threshold-excess pattern from Figure 3. This stricter quantity is used only for independent reconstruction and does not alter panel a.
- Target: basin MHW AUC at the same event and lead.
- Reference model: fixed basin and lead indicators only. These represent the typical difficulty of each region and forecast range without using pathway information.
- Cross-validation: each event is held out in full, so no lead or basin from that event enters model fitting. This prevents event leakage.
- Panel-c aggregation: all 72 out-of-sample predictions are ranked by inferred loss and divided into three equal-sized groups. Grouping is applied only after cross-validation. The lower and upper tertile boundaries are approximately `-0.015` and `0.084` AUC. The sector summarizes agreement across the three group means; sample-level `R2` and RMSE remain the formal performance statistics.
- Sensitivity analysis in panel b: the primary definition uses the equal 1997/98 and 2015/16 reference and leads 1, 3, 6 and 9. Alternatives use either reference event separately and the complete or leave-one-lead-out sets, yielding 15 definitions including the primary. These are structured sensitivity cases, not confidence intervals or independent replicates. Full values are stored in `Figure4_allocation_sensitivity.csv`.

## Panel-a accounting

Panel a remains an **association-based diagnostic accounting**, not an independent causal decomposition. The Figure 1 relationship gives expected global AUC `0.705`; observed 2023/24 AUC is `0.596`, producing the plotted gap of `{result['global_skill_loss']:.3f}`. Raw positive comparable-event minus 2023/24 basin deficits sum to `0.349`; they are normalized to shares and multiplied by `{result['global_skill_loss']:.3f}`. Therefore the basin bar lengths are allocated shares of the global gap, not the raw basin AUC differences. The scaling factor is `0.3124`.

The six allocated basin bars sum to `{result['global_skill_loss']:.3f}`, and the four segments of every bar close to its allocated total within numerical precision. The original descriptive path-model R2 (`0.758`) defines the accounted-for fraction. Within each basin, that amount is divided in proportion to the archived source, Z200 transmission and regional-process indices; the remainder is labelled not accounted for. These are the same numbers used in the earlier Figure 4. They identify co-occurring diagnostic pathways and must not be described as counterfactual causal effects.

No confidence interval is drawn on the panel-a basin totals because the available accounting table contains one event-composite deficit per basin rather than independently resampled monthly basin-deficit fields. Panel b provides a definition-sensitivity analysis instead of a sampling-confidence interval. Independent out-of-sample evidence is confined to panel c.

## Critical interpretation boundary

Only three events have complete common pathway fields. Panel b tests dependence on the two available historical references and four archived leads, but cannot replace a larger event sample. ENSO source-signal and teleconnection transmission errors are physically and statistically related. The basin-local process signal retained in panels a-b includes precipitation-pattern fidelity and is therefore a broad regional-response proxy, not a pure ocean-process attribution. The Atlantic bridge is shared by the three Atlantic basin targets. The title uses **underpinned** to denote convergent diagnostic and held-out predictive evidence, not formal causal mediation. Dedicated perturbation hindcasts and longer process archives would be required for causal attribution.

The primary earlier-event reference is the equal mean of 1997/98 and 2015/16. Reproducible outputs are `Figure4_skill_loss_attribution.csv`, `Figure4_allocation_sensitivity.csv`, `Figure4_cross_validated_models.csv`, `Figure4_error_reduction.csv`, `Figure4_out_of_sample_reconstruction.csv`, `Figure4_out_of_sample_calibration.csv` and `Figure4_attribution_summary.json`. Source definitions and checksums are recorded in `{(DERIVED / 'calculation_provenance.json').resolve()}`.
""", encoding="utf-8")


def main() -> int:
    args = parse_args()
    style()
    if args.figure1_sedi_only:
        directory = args.figures_root / "Figure1"
        figure1_sedi(directory, args.dpi, args.bootstrap, args.seed)
        print(f"[Saved] {directory / 'Figure1_SEDI.png'}", flush=True)
        return 0
    if args.figure1_rmse_only:
        directory = args.figures_root / "Figure1"
        figure1_rmse(directory, args.dpi, args.bootstrap, args.seed)
        print(f"[Saved] {directory / 'Figure1_RMSE.png'}", flush=True)
        return 0
    results = {}
    for index, function in enumerate((figure1, figure2, figure3, figure4), 1):
        directory = args.figures_root / f"Figure{index}"
        if index == 1:
            results[f"Figure{index}"] = function(directory, args.dpi, args.bootstrap, args.seed)
        else:
            results[f"Figure{index}"] = function(directory, args.dpi)
        print(f"[Saved] {directory / f'Figure{index}.png'}", flush=True)
    manifest = {
        "real_data_only": True,
        "simulated_values_used": False,
        "layout_reference_only": "The supplied simulated figures define panel topology only; every plotted value is recomputed from recorded observations, reanalyses or forecasts.",
        "figure_script": str(Path(__file__).resolve()),
        "calculation_script": str((PAPER_DIR / "Codes/k12_compute_nature_reference_products.py").resolve()),
        "derived_root": str(DERIVED.resolve()),
        "results": results,
    }
    (args.figures_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
