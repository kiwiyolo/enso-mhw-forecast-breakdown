#!/usr/bin/env python
"""Compute real-data products for the reference-layout Nature Figures 1-4."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parent
RAW = PAPER_DIR / "Data/Nature_real_rebuild/raw"
DEFAULT_OUTPUT = PAPER_DIR / "Data/Nature_real_rebuild/derived"
BASELINE_START = "1985-01"
BASELINE_END = "2014-12"
TARGET_EVENT = "2023/24"
HISTORICAL_FORECAST_EVENTS = ("1997/98", "2015/16")
HISTORICAL_GROUP = "Comparable events"
LEADS = (1, 3, 6, 9)
SOURCE_REGION_LEADS = tuple(range(1, 10))
NMME_MODELS = ("COLA-RSMAS-CCSM4", "NASA-GEOSS2S", "GFDL-SPEAR", "NCEP-CFSv2")
COMMON_LAT = np.arange(-89.5, 90.0, 1.0, dtype=np.float32)
COMMON_LON = np.arange(0.5, 360.0, 1.0, dtype=np.float32)
MHW_REFERENCE = PAPER_DIR / "output/Figure1/candidate09_metrics/nmme_preliminary/ERSST_MHW_reference_1985-2014.npz"
COMMON_EVALUATION_MASK = (
    PROJECT_DIR
    / "kw_99_OA-model/outputs/evaluation/v322_global_phase_safe_nio/"
    "candidate09_1985-2014_mhw_2022-2024/figure1_common_evaluation_mask.npz"
)

BASINS = {
    "North Pacific": (0.0, 60.0, 120.0, 280.0),
    "South Pacific": (-60.0, 0.0, 120.0, 290.0),
    "Indian Ocean": (-60.0, 30.0, 20.0, 120.0),
    "North Atlantic": (20.0, 60.0, 280.0, 360.0),
    "Tropical Atlantic": (-20.0, 20.0, 280.0, 360.0),
    "South Atlantic": (-60.0, 0.0, 290.0, 360.0),
    "Global 60S-60N": (-60.0, 60.0, 0.0, 360.0),
}

BRIDGES = {
    "PNA (N. Pacific)": (20.0, 60.0, 120.0, 260.0),
    "PSA (S. Pacific)": (-60.0, -20.0, 120.0, 290.0),
    "Indian bridge": (-20.0, 30.0, 40.0, 120.0),
    "Atlantic bridge": (0.0, 60.0, 280.0, 360.0),
}

BRIDGE_TO_BASIN = {
    "PNA (N. Pacific)": "North Pacific",
    "PSA (S. Pacific)": "South Pacific",
    "Indian bridge": "Indian Ocean",
    "Atlantic bridge": "North Atlantic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--skip-source-components", action="store_true")
    return parser.parse_args()


def normalize(data: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old, new in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
        if old in data.dims:
            rename[old] = new
    if rename:
        data = data.rename(rename)
    if "lon" in data.coords:
        data = data.assign_coords(lon=np.mod(data.lon.astype(float), 360.0)).sortby("lon")
    if "lat" in data.coords and data.lat[0] > data.lat[-1]:
        data = data.sortby("lat")
    return data


def monthly_anomaly(data: xr.DataArray) -> xr.DataArray:
    baseline = data.sel(time=slice(BASELINE_START, BASELINE_END))
    return data.groupby("time.month") - baseline.groupby("time.month").mean("time")


def weighted_mean(data: xr.DataArray, dims: tuple[str, ...] = ("lat", "lon")) -> xr.DataArray:
    if "lat" not in dims:
        return data.mean(dims, skipna=True)
    weights = np.cos(np.deg2rad(data.lat)).clip(min=0.0)
    return data.weighted(weights).mean(dims, skipna=True)


def period_range(start: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(start, end, freq="M")


def parse_cpc_nino(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        values = line.split()
        if len(values) < 10:
            continue
        rows.append(
            {
                "time": pd.Timestamp(int(values[0]), int(values[1]), 1),
                "nino12": float(values[3]),
                "nino3": float(values[5]),
                "nino4": float(values[7]),
                "nino34": float(values[9]),
            }
        )
    result = pd.DataFrame(rows).set_index("time").sort_index()
    result["nino34_3month"] = result.nino34.rolling(3, center=True, min_periods=3).mean()
    return result


def file_lookup(root: Path, variable: str) -> dict[pd.Period, Path]:
    result = {}
    for path in (root / variable).glob("*.nc"):
        match = re.search(r"_(\d{6})_(?:CONS|OPER)", path.name)
        if match:
            result[pd.Period(match.group(1), freq="M")] = path
    return result


def basin_mask(lat: xr.DataArray, lon: xr.DataArray, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    south, north, west, east = bounds
    return (lat >= south) & (lat <= north) & (lon >= west) & (lon < east)


def field_correlation(first: xr.DataArray, second: xr.DataArray, bounds: tuple[float, float, float, float]) -> float:
    second = second.interp(lat=first.lat, lon=first.lon)
    mask = basin_mask(first.lat, first.lon, bounds)
    a, b = xr.broadcast(first.where(mask), second.where(mask))
    weights = np.cos(np.deg2rad(a.lat)).broadcast_like(a)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(weights)
    x = a.where(valid).values.ravel()
    y = b.where(valid).values.ravel()
    w = weights.where(valid).values.ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if keep.sum() < 20:
        return np.nan
    x, y, w = x[keep], y[keep], w[keep]
    w = w / w.sum()
    x = x - np.sum(w * x)
    y = y - np.sum(w * y)
    denominator = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / denominator) if denominator > 0 else np.nan


def field_rms(data: xr.DataArray, bounds: tuple[float, float, float, float]) -> float:
    mask = basin_mask(data.lat, data.lon, bounds)
    masked = data.where(mask)
    return float(np.sqrt(weighted_mean(masked * masked)))


def normalize_nmme_sst(field: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old in ("Y", "y", "latitude"):
        if old in field.dims:
            rename[old] = "lat"
            break
    for old in ("X", "x", "longitude"):
        if old in field.dims:
            rename[old] = "lon"
            break
    if rename:
        field = field.rename(rename)
    field = field.assign_coords(lon=np.mod(np.asarray(field.lon), 360.0)).sortby("lon").sortby("lat")
    if str(field.attrs.get("units", "")).lower() in {"k", "kelvin"}:
        field = field - 273.15
    return field.where(field > -100.0)


def nmme_to_common(field: xr.DataArray) -> np.ndarray:
    field = normalize_nmme_sst(field)
    first = field.isel(lon=-1).assign_coords(lon=float(field.lon[-1]) - 360.0)
    last = field.isel(lon=0).assign_coords(lon=float(field.lon[0]) + 360.0)
    extended = xr.concat((first, field, last), dim="lon")
    return np.asarray(extended.interp(lat=COMMON_LAT, lon=COMMON_LON), dtype=np.float32)


def load_nmme_leads(initialization: pd.Timestamp, leads: tuple[int, ...] = LEADS) -> dict[int, np.ndarray]:
    model_values: dict[int, list[np.ndarray]] = {lead: [] for lead in leads}
    for model in NMME_MODELS:
        stamp = f"{initialization.year:04d}{initialization.month:02d}"
        path = PROJECT_DIR / "data" / model / f"sst_{stamp}.nc"
        if model == "GFDL-SPEAR":
            regridded = PROJECT_DIR / "data" / model / f"sst_regridded_{stamp}.nc"
            if regridded.is_file():
                path = regridded
        if not path.is_file():
            continue
        with xr.open_dataset(path, decode_times=False) as dataset:
            field = dataset["sst" if "sst" in dataset else next(iter(dataset.data_vars))]
            if "S" in field.dims:
                field = field.isel(S=0)
            if "M" in field.dims:
                field = field.mean("M", skipna=True)
            field = field.where((field >= -3.0) & (field <= 40.0))
            for lead in leads:
                # NMME L=0.5 verifies the initialization month, so t+lead is index lead.
                if lead < field.sizes.get("L", 1):
                    model_values[lead].append(nmme_to_common(field.isel(L=lead)))
    result = {}
    for lead, values in model_values.items():
        if not values:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            result[lead] = np.nanmean(np.stack(values), axis=0).astype(np.float32)
    return result


def weighted_array_auc(event: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(event) & np.isfinite(score) & np.isfinite(weight) & (weight > 0)
    truth = event[valid].astype(np.float64)
    values = score[valid].astype(np.float64)
    weights = weight[valid].astype(np.float64)
    positive = weights * truth
    negative = weights * (1.0 - truth)
    denominator = positive.sum() * negative.sum()
    if valid.sum() < 2 or denominator <= 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values, positive, negative = values[order], positive[order], negative[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    positive_group = np.add.reduceat(positive, starts)
    negative_group = np.add.reduceat(negative, starts)
    negative_before = np.cumsum(negative_group) - negative_group
    return float(np.sum(positive_group * (negative_before + 0.5 * negative_group)) / denominator)


def weighted_array_correlation(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weight) & (weight > 0)
    if valid.sum() < 20:
        return np.nan
    x, y, w = first[valid], second[valid], weight[valid]
    w = w / w.sum()
    x = x - np.sum(w * x)
    y = y - np.sum(w * y)
    denominator = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / denominator) if denominator > 0 else np.nan


def weighted_array_sedi(event: np.ndarray, forecast: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(event) & np.isfinite(forecast) & np.isfinite(weight) & (weight > 0)
    truth = event[valid].astype(bool)
    predicted = forecast[valid].astype(bool)
    selected = weight[valid].astype(np.float64)
    hit = selected[truth & predicted].sum()
    miss = selected[truth & ~predicted].sum()
    false_alarm = selected[~truth & predicted].sum()
    correct_negative = selected[~truth & ~predicted].sum()
    if hit + miss <= 0 or false_alarm + correct_negative <= 0:
        return np.nan
    hit_rate = np.clip(hit / (hit + miss), 1e-6, 1.0 - 1e-6)
    false_alarm_rate = np.clip(
        false_alarm / (false_alarm + correct_negative), 1e-6, 1.0 - 1e-6
    )
    numerator = (
        np.log(false_alarm_rate)
        - np.log(hit_rate)
        - np.log1p(-false_alarm_rate)
        + np.log1p(-hit_rate)
    )
    denominator = (
        np.log(false_alarm_rate)
        + np.log(hit_rate)
        + np.log1p(-false_alarm_rate)
        + np.log1p(-hit_rate)
    )
    return float(numerator / denominator)


def observed_mhw_intensity_rmse(
    observation: np.ndarray,
    forecast: np.ndarray,
    threshold: np.ndarray,
    weight: np.ndarray,
) -> float:
    valid = (
        np.isfinite(observation)
        & np.isfinite(forecast)
        & np.isfinite(threshold)
        & np.isfinite(weight)
        & (weight > 0)
    )
    observed_event = valid & (observation > threshold)
    if not observed_event.any():
        return np.nan
    observed_intensity = np.maximum(observation - threshold, 0.0)
    forecast_intensity = np.maximum(forecast - threshold, 0.0)
    return float(
        np.sqrt(
            np.average(
                np.square(forecast_intensity[observed_event] - observed_intensity[observed_event]),
                weights=weight[observed_event],
            )
        )
    )


def build_figure1(nino: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = PAPER_DIR / "output/Figure1/candidate09_metrics/Figure1_candidate09_metrics_by_init_lead.csv"
    events_path = PAPER_DIR / "output/Figure1/supplemental_materials/Figure1_decoupling_event_summary_lead_all_NMME_ERSST.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[
        (metrics.Method == "NMME ensemble mean") & metrics.Lead_Month.between(1, 9)
    ].copy()
    metrics["time"] = pd.to_datetime(metrics.Target_Month)
    monthly = metrics.groupby("time", as_index=False).agg(AUC=("AUC", "mean"), leads=("Lead_Month", "nunique"))
    monthly = monthly.loc[monthly["leads"].eq(9)].copy()
    monthly = monthly.merge(nino[["nino34"]], left_on="time", right_index=True, how="left")
    monthly = monthly[(monthly.time >= "1991-01-01") & (monthly.time <= "2024-12-01")]
    monthly.rename(columns={"nino34": "Nino34"}, inplace=True)

    source_events = pd.read_csv(events_path)
    event_rows = []
    for row in source_events.itertuples():
        start, end = pd.Timestamp(row.Start_Month), pd.Timestamp(row.End_Month)
        start_year = int(str(row.Start_Month)[:4])
        mature = [
            pd.Timestamp(start_year, 11, 1),
            pd.Timestamp(start_year, 12, 1),
            pd.Timestamp(start_year + 1, 1, 1),
            pd.Timestamp(start_year + 1, 2, 1),
        ]
        event_monthly = monthly[monthly.time.between(start, end)]
        peak = nino.reindex(mature).nino34_3month.max()
        event_rows.append(
            {
                "Event": row.Event,
                "Start_Month": row.Start_Month,
                "End_Month": row.End_Month,
                "Peak_Nino34": float(peak),
                "Mean_AUC": float(event_monthly.AUC.mean()),
                "Valid_Months": int(event_monthly.AUC.notna().sum()),
            }
        )
    events = pd.DataFrame(event_rows)
    monthly.to_csv(output / "figure1_monthly_t1_t9.csv", index=False)
    events.to_csv(output / "figure1_event_relation.csv", index=False)
    return monthly, events


def detect_strong_peaks(nino: pd.DataFrame) -> dict[str, pd.Timestamp]:
    series = nino.loc["1981-01":"2020-12", "nino34_3month"].dropna()
    candidates = series[(series >= 1.5) & (series == series.rolling(7, center=True).max())]
    selected: list[pd.Timestamp] = []
    for timestamp in candidates.sort_values(ascending=False).index:
        if all(abs((timestamp.year - old.year) * 12 + timestamp.month - old.month) >= 12 for old in selected):
            selected.append(timestamp)
    selected.sort()
    result = {}
    for stamp in selected:
        start_year = stamp.year if stamp.month >= 7 else stamp.year - 1
        result[f"{start_year}/{str(start_year + 1)[-2:]}"] = stamp
    return result


def equatorial_mean(data: xr.DataArray, south: float = -5.0, north: float = 5.0) -> xr.DataArray:
    return weighted_mean(data.sel(lat=slice(south, north)), dims=("lat",))


def aligned_hovmoller(data: xr.DataArray, peaks: list[pd.Timestamp]) -> xr.DataArray:
    aligned = []
    for peak in peaks:
        dates = pd.date_range(peak - pd.DateOffset(months=18), peak + pd.DateOffset(months=18), freq="MS")
        item = data.reindex(time=dates).assign_coords(time=np.arange(-18, 19)).rename(time="relative_month")
        if "month" in item.coords:
            item = item.drop_vars("month")
        aligned.append(item)
    return xr.concat(aligned, dim="event").mean("event", skipna=True)


def load_oras_equatorial_profile(periods: pd.PeriodIndex) -> xr.DataArray:
    lookup = file_lookup(PROJECT_DIR / "data/ORAS5/surface_variables", "sozotaux")
    arrays = []
    for period in periods:
        path = lookup.get(period)
        if path is None:
            continue
        with xr.open_dataset(path) as dataset:
            item = normalize(dataset.sozotaux.isel(time=0)).sel(lat=slice(-5, 5), lon=slice(120, 280)).load()
        arrays.append(equatorial_mean(item).expand_dims(time=[period.to_timestamp()]))
    return xr.concat(arrays, dim="time").sortby("time")


def build_hovmoller(nino: pd.DataFrame, output: Path) -> dict[str, list[str]]:
    with xr.open_dataset(RAW / "noaa_ersstv5_sst_monthly.nc") as dataset:
        sst = normalize(dataset.sst.sel(time=slice("1980-01", "2025-12"))).load()
    with xr.open_dataset(RAW / "noaa_olr_monthly_v03r00_197901_202606.nc") as dataset:
        olr = normalize(dataset.olr.sel(time=slice("1980-01", "2025-12"))).load()
    olr = olr.assign_coords(
        time=pd.DatetimeIndex(olr.time.values).to_period("M").to_timestamp()
    )
    sst_anomaly = monthly_anomaly(sst).sel(lon=slice(120, 280))
    olr_anomaly = monthly_anomaly(olr).sel(lon=slice(120, 280))
    sst_eq = equatorial_mean(sst_anomaly)
    olr_eq = equatorial_mean(olr_anomaly)

    strong = detect_strong_peaks(nino)
    historical_peaks = list(strong.values())
    target_peak = nino.loc["2023-07":"2024-06", "nino34_3month"].idxmax()
    periods = period_range("1985-01", "2014-12")
    for peak in [*historical_peaks, target_peak]:
        periods = periods.union(pd.period_range(peak - pd.DateOffset(months=18), peak + pd.DateOffset(months=18), freq="M"))
    stress = load_oras_equatorial_profile(periods.sort_values())
    stress_anomaly = monthly_anomaly(stress)

    historical = {
        "sst": aligned_hovmoller(sst_eq, historical_peaks),
        "stress": aligned_hovmoller(stress_anomaly, historical_peaks),
        "olr": aligned_hovmoller(olr_eq, historical_peaks),
    }
    target = {
        "sst": aligned_hovmoller(sst_eq, [target_peak]),
        "stress": aligned_hovmoller(stress_anomaly, [target_peak]),
        "olr": aligned_hovmoller(olr_eq, [target_peak]),
    }
    common_lon = np.arange(120.0, 281.0, 2.0)
    data_vars = {}
    for variable in historical:
        data_vars[f"historical_{variable}"] = historical[variable].interp(lon=common_lon)
        data_vars[f"target_{variable}"] = target[variable].interp(lon=common_lon)
    xr.Dataset(data_vars).to_netcdf(output / "figure2_hovmoller_real.nc")
    metadata = {
        "historical_strong_events": [f"{name} (peak {stamp:%Y-%m})" for name, stamp in strong.items()],
        "target_peak": f"{target_peak:%Y-%m}",
        "equatorial_band": "5S-5N",
        "climatology": f"{BASELINE_START} to {BASELINE_END}",
    }
    (output / "figure2_hovmoller_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def source_area_mean(data: xr.DataArray) -> float:
    return float(weighted_mean(normalize(data).sel(lat=slice(-5, 5), lon=slice(120, 280))))


def load_source_components(output: Path) -> pd.DataFrame:
    cache = output / "figure2_source_process_monthly.csv"
    if cache.is_file():
        cached = pd.read_csv(cache, parse_dates=["time"]).set_index("time").sort_index()
        required = {
            "z_thermocline_proxy", "z_zonal_advection", "z_surface_heat_flux", "z_ocean_dynamics_residual"
        }
        if required.issubset(cached.columns):
            return cached
    surface_root = PROJECT_DIR / "data/ORAS5/surface_variables"
    ocean_root = PROJECT_DIR / "data/ORAS5/ocean_variables"
    lookups = {
        "qnet": file_lookup(surface_root, "sohefldo"),
        "mld": file_lookup(ocean_root, "somxl030"),
        "ssh": file_lookup(surface_root, "sossheig"),
        "u": file_lookup(ocean_root, "vozocrte"),
    }
    with xr.open_dataset(RAW / "noaa_ersstv5_sst_monthly.nc") as dataset:
        sst = normalize(dataset.sst.sel(time=slice("1985-01", "2024-12"))).load()
    sst_anomaly = monthly_anomaly(sst)
    rows = []
    for period in period_range("1985-01", "2024-12"):
        if any(period not in lookup for lookup in lookups.values()):
            continue
        values = {}
        for name, variable in (("qnet", "sohefldo"), ("mld", "somxl030"), ("ssh", "sossheig")):
            with xr.open_dataset(lookups[name][period]) as dataset:
                values[name] = source_area_mean(dataset[variable].isel(time=0).load())
        with xr.open_dataset(lookups["u"][period]) as dataset:
            values["u"] = source_area_mean(dataset.vozocrte.isel(time=0, depth=0).load())
        timestamp = period.to_timestamp()
        current = sst_anomaly.sel(time=timestamp)
        west = source_area_mean(current.sel(lon=slice(120, 200)))
        east = source_area_mean(current.sel(lon=slice(200, 280)))
        source_sst = source_area_mean(current)
        seconds = period.days_in_month * 86400.0
        heat_tendency = values["qnet"] * seconds / (1025.0 * 3990.0 * max(values["mld"], 5.0))
        gradient = (east - west) / (80.0 * 111_000.0)
        zonal_advection = -values["u"] * gradient * seconds
        rows.append(
            {
                "time": timestamp,
                "sst_anomaly": source_sst,
                "thermocline_proxy": values["ssh"],
                "zonal_advection": zonal_advection,
                "surface_heat_flux": heat_tendency,
            }
        )
    frame = pd.DataFrame(rows).set_index("time").sort_index()
    frame["sst_tendency"] = frame.sst_anomaly.diff()
    frame["ocean_dynamics_residual"] = frame.sst_tendency - frame.surface_heat_flux - frame.zonal_advection
    for name in ("thermocline_proxy", "zonal_advection", "surface_heat_flux", "ocean_dynamics_residual"):
        climatology = frame.loc[BASELINE_START:BASELINE_END].groupby(frame.loc[BASELINE_START:BASELINE_END].index.month)[name].mean()
        anomaly = frame[name] - frame.index.month.map(climatology)
        scale = float(anomaly.loc[BASELINE_START:BASELINE_END].std())
        frame[f"z_{name}"] = anomaly / max(scale, 1e-8)
    frame.reset_index().to_csv(output / "figure2_source_process_monthly.csv", index=False)
    return frame


def source_component_shares(frame: pd.DataFrame, nino: pd.DataFrame, output: Path) -> pd.DataFrame:
    peaks = detect_strong_peaks(nino)
    target_peak = nino.loc["2023-07":"2024-06", "nino34_3month"].idxmax()
    components = (
        ("Ocean dynamics", "z_ocean_dynamics_residual"),
        ("Surface heat flux", "z_surface_heat_flux"),
        ("Zonal advection", "z_zonal_advection"),
        ("Thermocline proxy", "z_thermocline_proxy"),
    )
    rows = []
    for phase, offsets in (("Development", range(-6, 0)), ("Mature", range(0, 6))):
        historical_peaks = [peaks[event] for event in HISTORICAL_FORECAST_EVENTS if event in peaks]
        for group, group_peaks in ((HISTORICAL_GROUP, historical_peaks), ("2023/24", [target_peak])):
            scores = []
            for _, column in components:
                values = []
                for peak in group_peaks:
                    dates = [peak + pd.DateOffset(months=offset) for offset in offsets]
                    values.extend(frame.reindex(dates)[column].abs().dropna().tolist())
                scores.append(float(np.mean(values)))
            denominator = max(sum(scores), 1e-12)
            for (component, _), score in zip(components, scores, strict=True):
                rows.append({"Phase": phase, "Group": group, "Component": component, "Share": score / denominator})
    result = pd.DataFrame(rows)
    result.to_csv(output / "figure2_source_process_shares.csv", index=False)
    return result


def load_source_region_calculator():
    source_dir = PAPER_DIR / "Figure2_ENSO-forecast"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import k01_cal_source_region_prediction_errors as source_region

    return source_region


def build_figure2_process_forecast_skill(output: Path) -> dict[str, object]:
    source_region = load_source_region_calculator()
    # Retain the published analysis window while showing a slightly broader
    # tropical band so the two map columns share the reference framing.
    source_bounds = (-20.0, 20.0, 140.0, 260.0)
    map_bounds = (-25.0, 25.0, 140.0, 260.0)
    config = source_region.CalculationConfig(
        work_dir=PAPER_DIR / "Figure2_ENSO-forecast",
        data_dir=PROJECT_DIR / "data",
        output_dir=output,
        obs_label="ERSST",
        obs_path=PROJECT_DIR / "data/ersst_observation.nc",
        baseline_start="1991-01",
        baseline_end="2020-12",
        leads=SOURCE_REGION_LEADS,
        map_event=TARGET_EVENT,
        map_lead=9,
        lat_min=source_bounds[0],
        lat_max=source_bounds[1],
        lon_min=source_bounds[2],
        lon_max=source_bounds[3],
        # Tropical Pacific SST is safely above 10 C. Several NMME archives use
        # small positive land/coastal fill values that survive the old >1 C test
        # and become -20 C anomalies after climatology subtraction.
        forecast_min_valid_sst=10.0,
        # NASA provides t+1...t+8 and the local GFDL SST files from 2021 onward
        # fail the internal-start-month audit. Requiring two valid models keeps
        # t+9 while never admitting a single-source grid cell.
        forecast_min_sources_per_cell=2,
    )
    map_config = replace(
        config,
        lat_min=map_bounds[0],
        lat_max=map_bounds[1],
        lon_min=map_bounds[2],
        lon_max=map_bounds[3],
    )
    observation_sst = source_region.load_observation(config)
    climatology = source_region.compute_monthly_climatology(observation_sst, config)
    rows: list[dict[str, object]] = []
    sst_fields: dict[str, tuple[xr.DataArray, list[xr.DataArray]]] = {}
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        peak_start, peak_end = source_region.EVENT_PEAK_SEASONS[event]
        peak_months = source_region.month_range(peak_start, peak_end)
        observed, _, _ = source_region.seasonal_mean_anomaly(
            peak_months, "OBS", 0, observation_sst, climatology, config
        )
        if observed is None:
            raise RuntimeError(f"Missing observed source-region SST composite for {event}")
        event_forecasts: list[xr.DataArray] = []
        for lead in SOURCE_REGION_LEADS:
            forecast, member_month_count, source_month_count = source_region.seasonal_mean_anomaly(
                peak_months, "NMME", lead, observation_sst, climatology, config
            )
            if forecast is None:
                raise RuntimeError(f"Missing NMME source-region SST composite for {event}, lead {lead}")
            rows.append(
                {
                    "Process": "SST",
                    "Event": event,
                    "Lead": lead,
                    "Pattern_correlation": field_correlation(observed, forecast, source_bounds),
                    "NMME_L_months": lead + 0.5,
                    "Mean_model_count": source_month_count / len(peak_months),
                    "Member_month_count": member_month_count,
                }
            )
            event_forecasts.append(forecast.expand_dims(lead=[lead]))
        sst_fields[event] = (observed, event_forecasts)

    map_observation_sst = source_region.load_observation(map_config)
    map_climatology = source_region.compute_monthly_climatology(
        map_observation_sst, map_config
    )
    sst_map_fields: dict[str, tuple[xr.DataArray, list[xr.DataArray]]] = {}
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        peak_start, peak_end = source_region.EVENT_PEAK_SEASONS[event]
        peak_months = source_region.month_range(peak_start, peak_end)
        observed, _, _ = source_region.seasonal_mean_anomaly(
            peak_months,
            "OBS",
            0,
            map_observation_sst,
            map_climatology,
            map_config,
        )
        if observed is None:
            raise RuntimeError(f"Missing map-domain SST observation for {event}")
        event_forecasts = []
        for lead in SOURCE_REGION_LEADS:
            forecast, _, _ = source_region.seasonal_mean_anomaly(
                peak_months,
                "NMME",
                lead,
                map_observation_sst,
                map_climatology,
                map_config,
            )
            if forecast is None:
                raise RuntimeError(
                    f"Missing map-domain NMME SST composite for {event}, lead {lead}"
                )
            event_forecasts.append(forecast.expand_dims(lead=[lead]))
        sst_map_fields[event] = (observed, event_forecasts)

    map_root = PAPER_DIR / "teleconnection_data_products/maps"
    atmospheric_fields: dict[str, dict[str, tuple[xr.DataArray, list[xr.DataArray]]]] = {}
    for variable, process in (
        ("surface_stress_x", "Zonal wind stress"),
        ("precipitation", "Convection proxy"),
    ):
        observation = load_event_map(
            map_root / f"{variable}_anomaly_OBS_ERA5.nc", f"{variable}_anomaly"
        )
        if variable == "surface_stress_x":
            observation = observation / 86400.0
        else:
            observation = observation * 1000.0
        event_fields: dict[str, tuple[xr.DataArray, list[xr.DataArray]]] = {
            event: (observation.sel(event=event), [])
            for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT)
        }
        for lead in SOURCE_REGION_LEADS:
            # k02 products call L=0.5 "lead01". The common t+lead contract
            # therefore reads product lead=(lead+1): t+1 -> L=1.5 -> lead02.
            product_lead = lead + 1
            forecast_path = map_root / f"{variable}_anomaly_NMME_NMME_lead{product_lead:02d}.nc"
            if not forecast_path.is_file():
                raise FileNotFoundError(
                    f"Missing continuous-lead Figure 2 map: {forecast_path}. "
                    "Run k02_calculate_teleconnection.py for product leads 2-10."
                )
            forecast = load_event_map(forecast_path, f"{variable}_anomaly")
            for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
                rows.append(
                    {
                        "Process": process,
                        "Event": event,
                        "Lead": lead,
                        "Pattern_correlation": field_correlation(
                            observation.sel(event=event),
                            forecast.sel(event=event),
                            source_bounds,
                        ),
                        "NMME_L_months": lead + 0.5,
                        "Mean_model_count": np.nan,
                        "Member_month_count": np.nan,
                    }
                )
                event_fields[event][1].append(forecast.sel(event=event).expand_dims(lead=[lead]))
        atmospheric_fields[variable] = event_fields

    result = pd.DataFrame(rows).sort_values(["Process", "Event", "Lead"])
    result["Group"] = np.where(result.Event.eq(TARGET_EVENT), TARGET_EVENT, HISTORICAL_GROUP)
    result.to_csv(output / "figure2_process_forecast_skill.csv", index=False)

    reference_field = atmospheric_fields["surface_stress_x"][TARGET_EVENT][1][0]
    common_lat = reference_field.lat.sel(lat=slice(map_bounds[0], map_bounds[1]))
    common_lon = reference_field.lon.sel(lon=slice(map_bounds[2], map_bounds[3]))

    def common_grid(data: xr.DataArray) -> xr.DataArray:
        return data.reset_coords(drop=True).interp(lat=common_lat, lon=common_lon).astype(np.float32)

    relative_error_scales: list[dict[str, object]] = []

    def lead_mean_relative_error(
        observed: xr.DataArray,
        forecasts: list[xr.DataArray],
        process: str,
        event: str,
        physical_unit: str,
    ) -> xr.DataArray:
        forecast = xr.concat(
            forecasts, dim="lead", coords="minimal", compat="override"
        ).mean("lead", skipna=True)
        observed_on_forecast = observed.interp(lat=forecast.lat, lon=forecast.lon)
        observed_pattern_rms = field_rms(observed_on_forecast, source_bounds)
        if not np.isfinite(observed_pattern_rms) or observed_pattern_rms <= 0:
            raise ValueError(
                f"Invalid observed-pattern RMS for {process}, {event}: "
                f"{observed_pattern_rms}"
            )
        relative_error_scales.append(
            {
                "Process": process,
                "Event": event,
                "Observed_pattern_RMS": observed_pattern_rms,
                "Physical_unit": physical_unit,
                "Domain": "20S-20N, 140E-100W",
            }
        )
        relative_error = 100.0 * (forecast - observed_on_forecast) / observed_pattern_rms
        relative_error.attrs.update(
            {
                "units": "percent_of_observed_pattern_rms",
                "observed_pattern_rms": observed_pattern_rms,
                "observed_pattern_rms_unit": physical_unit,
            }
        )
        return common_grid(relative_error)

    def comparable_relative_error(
        fields: dict[str, tuple[xr.DataArray, list[xr.DataArray]]],
        process: str,
        physical_unit: str,
    ) -> xr.DataArray:
        errors = [
            lead_mean_relative_error(
                *fields[event], process, event, physical_unit
            ).expand_dims(event=[event])
            for event in HISTORICAL_FORECAST_EVENTS
        ]
        return xr.concat(errors, dim="event").mean("event", skipna=True)

    sst_comparable_relative_error = comparable_relative_error(
        sst_map_fields, "SST", "degree_C"
    )
    sst_target_relative_error = lead_mean_relative_error(
        *sst_map_fields[TARGET_EVENT], "SST", TARGET_EVENT, "degree_C"
    )
    stress_comparable_relative_error = comparable_relative_error(
        atmospheric_fields["surface_stress_x"], "Zonal wind stress", "N m-2"
    )
    stress_target_relative_error = lead_mean_relative_error(
        *atmospheric_fields["surface_stress_x"][TARGET_EVENT],
        "Zonal wind stress",
        TARGET_EVENT,
        "N m-2",
    )
    precipitation_comparable_relative_error = comparable_relative_error(
        atmospheric_fields["precipitation"], "Convection proxy", "mm day-1"
    )
    precipitation_target_relative_error = lead_mean_relative_error(
        *atmospheric_fields["precipitation"][TARGET_EVENT],
        "Convection proxy",
        TARGET_EVENT,
        "mm day-1",
    )
    map_dataset = xr.Dataset(
        {
            "sst_comparable_relative_error": sst_comparable_relative_error,
            "sst_target_relative_error": sst_target_relative_error,
            "stress_comparable_relative_error": stress_comparable_relative_error,
            "stress_target_relative_error": stress_target_relative_error,
            "precipitation_comparable_relative_error": precipitation_comparable_relative_error,
            "precipitation_target_relative_error": precipitation_target_relative_error,
        },
        attrs={
            "title": "Tropical-Pacific NMME relative process-forecast errors",
            "comparison_events": ",".join(HISTORICAL_FORECAST_EVENTS),
            "target_event": TARGET_EVENT,
            "lead_months": "1,2,3,4,5,6,7,8,9",
            "lead_contract": "t+1...t+9; NMME L=1.5...9.5",
            "domain": "25S-25N, 140E-100W",
            "analysis_domain": "20S-20N, 140E-100W",
            "error_definition": "100 * (equal-lead-mean forecast anomaly - observed anomaly) / area-weighted RMS(observed anomaly over 20S-20N, 140E-100W); relative errors are calculated per event before the equal-event comparable mean",
            "units": "percent_of_observed_pattern_rms",
        },
    )
    map_dataset.to_netcdf(output / "figure2_source_process_maps_t1_t9.nc")
    pd.DataFrame(relative_error_scales).to_csv(
        output / "figure2_relative_error_scales.csv", index=False
    )
    deprecated_profile = output / "figure2_source_latitude_error_profiles.csv"
    if deprecated_profile.exists():
        deprecated_profile.unlink()

    audit_rows = []
    for (process, event), frame in result.groupby(["Process", "Event"]):
        frame = frame.sort_values("Lead")
        linear = stats.linregress(frame.Lead, frame.Pattern_correlation)
        spearman = stats.spearmanr(frame.Lead, frame.Pattern_correlation)
        audit_rows.append(
            {
                "Process": process,
                "Event": event,
                "Linear_slope_per_month": float(linear.slope),
                "Pearson_r_lead_vs_fidelity": float(linear.rvalue),
                "Spearman_rho_lead_vs_fidelity": float(spearman.statistic),
                "Spearman_p": float(spearman.pvalue),
                "Decreases_with_lead": bool(linear.slope < 0),
            }
        )
    fidelity_audit = pd.DataFrame(audit_rows)
    fidelity_audit.to_csv(output / "figure2_pattern_fidelity_lead_audit.csv", index=False)

    source_audit_rows = []
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        peak_start, peak_end = source_region.EVENT_PEAK_SEASONS[event]
        for target_month in source_region.month_range(peak_start, peak_end):
            for lead in SOURCE_REGION_LEADS:
                init_month = source_region.add_months(target_month, -lead)
                expected_period = pd.Period(init_month, freq="M")
                expected_start = (
                    (expected_period.year - 1960) * 12 + expected_period.month - 1
                )
                for model in NMME_MODELS:
                    path = source_region.model_file_for_init(model, init_month, config)
                    actual_start = np.nan
                    available_leads = 0
                    start_matches = False
                    if path is not None:
                        with xr.open_dataset(path, decode_times=False) as dataset:
                            if "S" in dataset.coords:
                                starts = np.asarray(dataset.S.values, dtype=float).reshape(-1)
                                actual_start = float(starts[0]) if starts.size else np.nan
                                start_matches = bool(
                                    np.isfinite(actual_start)
                                    and abs(actual_start - expected_start) <= 0.1
                                )
                            else:
                                start_matches = True
                            variable = source_region.choose_data_var(dataset)
                            available_leads = int(dataset[variable].sizes.get("L", 0))
                    usable = bool(
                        path is not None and start_matches and available_leads > lead
                    )
                    reason = (
                        "usable"
                        if usable
                        else "missing_file"
                        if path is None
                        else "start_month_mismatch"
                        if not start_matches
                        else "lead_unavailable"
                    )
                    source_audit_rows.append(
                        {
                            "Event": event,
                            "Target_month": target_month,
                            "Lead": lead,
                            "Initialization_month": init_month,
                            "Model": model,
                            "Path": str(path.resolve()) if path is not None else "",
                            "Expected_S": expected_start,
                            "Actual_S": actual_start,
                            "Available_L_count": available_leads,
                            "Usable": usable,
                            "Reason": reason,
                        }
                    )
    source_audit = pd.DataFrame(source_audit_rows)
    source_audit.to_csv(output / "figure2_nmme_sst_start_coordinate_audit.csv", index=False)
    metadata = {
        "metric": "area-weighted forecast-observation anomaly-pattern correlation",
        "domain": "20S-20N, 140E-100W",
        "map_domain": "25S-25N, 140E-100W",
        "domain_reference": "Peng et al. (2025), Nature Geoscience, Fig. 1",
        "comparison_label": HISTORICAL_GROUP,
        "comparison_events_documentation_only": list(HISTORICAL_FORECAST_EVENTS),
        "target_event": TARGET_EVENT,
        "lead_months": list(SOURCE_REGION_LEADS),
        "lead_contract": "NMME L=0.5 is the initialization month; t+1...t+9 use L=1.5...9.5",
        "map_aggregation": "forecast minus observation, equal mean across target-aligned t+1...t+9, then equal event mean",
        "forecast_absolute_sst_validity_floor_C": 10.0,
        "forecast_minimum_NMME_sources_per_cell": 2,
        "forecast_start_coordinate_check": "internal S must match the YYYYMM file name",
        "known_rejected_source": "GFDL-SPEAR SST files from 2021 onward have S=731 and are rejected when the file-name month differs",
        "sst_source_audit_rows": int(len(source_audit)),
        "sst_source_audit_rejected_start_mismatch": int(
            source_audit.Reason.eq("start_month_mismatch").sum()
        ),
        "sst_source_audit_lead_unavailable": int(
            source_audit.Reason.eq("lead_unavailable").sum()
        ),
        "convection_proxy": "NMME precipitation; archived 2023 NMME OLR is unavailable",
        "fidelity_lead_audit": fidelity_audit.to_dict(orient="records"),
    }
    (output / "figure2_process_forecast_skill_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_event_map(path: Path, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as dataset:
        return normalize(dataset[variable].load())


def event_map_error(variable: str, lead: int, event: str, bounds: tuple[float, float, float, float], obs_scale: float = 1.0) -> float:
    root = PAPER_DIR / "teleconnection_data_products/maps"
    obs = load_event_map(root / f"{variable}_anomaly_OBS_ERA5.nc", f"{variable}_anomaly").sel(event=event) * obs_scale
    forecast = load_event_map(root / f"{variable}_anomaly_NMME_NMME_lead{lead:02d}.nc", f"{variable}_anomaly").sel(event=event)
    obs = obs.interp(lat=forecast.lat, lon=forecast.lon)
    mask = basin_mask(forecast.lat, forecast.lon, bounds)
    return field_rms((forecast - obs).where(mask), bounds)


def build_figure2_summaries(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = pd.read_csv(PAPER_DIR / "Figure2_ENSO-forecast/output/Figure2_ENSO_source_region_lead_errors.csv")
    source_skill = pd.read_csv(PAPER_DIR / "Figure2_ENSO-forecast/figures/Figure2_source_region_all_el_nino_event_summary.csv")
    process_skill = pd.read_csv(output / "figure2_process_forecast_skill.csv")
    groups = {
        HISTORICAL_GROUP: list(HISTORICAL_FORECAST_EVENTS),
        "2023/24": [TARGET_EVENT],
    }
    event_rows = []
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        selected_errors = errors[errors.Event_Label == event]
        selected_process = process_skill[process_skill.Event == event]
        values = {
            "Nino3.4 peak intensity error (C)": float(selected_errors.Peak_Intensity_Error.abs().mean()),
            "Peak timing error (months)": float(selected_errors.Peak_Timing_Error_Months.abs().mean()),
            "SST pattern error (1-r)": float(
                1.0
                - selected_process.loc[
                    selected_process.Process == "SST", "Pattern_correlation"
                ].mean()
            ),
            "Wind-stress pattern error (1-r)": float(
                1.0
                - selected_process.loc[
                    selected_process.Process == "Zonal wind stress", "Pattern_correlation"
                ].mean()
            ),
            "Convection pattern error* (1-r)": float(
                1.0
                - selected_process.loc[
                    selected_process.Process == "Convection proxy", "Pattern_correlation"
                ].mean()
            ),
        }
        for metric, value in values.items():
            event_rows.append({"Event": event, "Metric": metric, "Value": value})
    event_errors = pd.DataFrame(event_rows)
    overall_mean = event_errors.groupby("Metric").Value.mean().rename("Overall_mean_error")

    rows = []
    for group, events in groups.items():
        selected = event_errors[event_errors.Event.isin(events)]
        for metric, frame in selected.groupby("Metric", sort=False):
            rows.append(
                {
                    "Group": group,
                    "Metric": metric,
                    "Value": float(frame.Value.mean()),
                    "Group_event_count": int(frame.Event.nunique()),
                }
            )
    summary = pd.DataFrame(rows).merge(overall_mean, on="Metric", validate="many_to_one")
    summary["Relative_to_overall_mean_percent"] = 100.0 * (
        summary.Value - summary.Overall_mean_error
    ) / summary.Overall_mean_error.clip(lower=1e-9)
    summary = summary[
        [
            "Group",
            "Metric",
            "Value",
            "Group_event_count",
            "Overall_mean_error",
            "Relative_to_overall_mean_percent",
        ]
    ]
    summary.to_csv(output / "figure2_source_signal_errors.csv", index=False)
    event_errors.to_csv(
        output / "figure2_source_signal_errors_by_event.csv", index=False
    )

    historical = source_skill[source_skill.Event_Label.isin(HISTORICAL_FORECAST_EVENTS)].copy()
    target = source_skill[source_skill.Is_2023_24]
    skill_rows = pd.DataFrame(
        [
            *(
                {
                    "Group": HISTORICAL_GROUP,
                    "Event": row.Event_Label,
                    "Record": "Event",
                    "AUC": row.Source_AUC,
                    "Event_count": 1,
                    "SD": np.nan,
                }
                for row in historical.itertuples()
            ),
            {
                "Group": HISTORICAL_GROUP,
                "Event": "Mean",
                "Record": "Summary",
                "AUC": historical.Source_AUC.mean(),
                "Event_count": len(historical),
                "SD": historical.Source_AUC.std(ddof=1),
            },
            {
                "Group": "2023/24",
                "Event": TARGET_EVENT,
                "Record": "Event",
                "AUC": target.Source_AUC.iloc[0],
                "Event_count": 1,
                "SD": np.nan,
            },
        ]
    )
    skill_rows.to_csv(output / "figure2_source_mhw_skill.csv", index=False)
    return summary, skill_rows


def correlation_map(field: xr.DataArray, index: xr.DataArray) -> xr.DataArray:
    index = index.broadcast_like(field)
    valid = np.isfinite(field) & np.isfinite(index)
    x = field.where(valid)
    y = index.where(valid)
    x = x - x.mean("time", skipna=True)
    y = y - y.mean("time", skipna=True)
    denominator = np.sqrt(((x * x).sum("time")) * ((y * y).sum("time")))
    return (x * y).sum("time") / denominator.where(denominator > 0)


def build_driver_map(nino: pd.DataFrame, output: Path) -> tuple[xr.Dataset, pd.DataFrame]:
    with xr.open_dataset(RAW / "noaa_ersstv5_sst_monthly.nc") as dataset:
        sst = normalize(dataset.sst.sel(time=slice("1985-01", "2024-12"), lat=slice(60, -60))).load()
    sst = sst.sortby("lat")
    anomaly = monthly_anomaly(sst)
    history = anomaly.sel(time=slice("1985-01", "2020-12"))
    nino_series = xr.DataArray(
        nino.reindex(pd.DatetimeIndex(history.time.values)).nino34.to_numpy(),
        coords={"time": history.time},
        dims="time",
    )
    enso_r = correlation_map(history, nino_series)
    beta = ((history - history.mean("time")) * (nino_series - nino_series.mean())).sum("time") / (
        ((nino_series - nino_series.mean()) ** 2).sum("time")
    )
    residual = history - beta * nino_series
    local_r = correlation_map(residual.isel(time=slice(1, None)), residual.shift(time=1).isel(time=slice(1, None)))
    enso_score = (enso_r * enso_r).clip(0.0, 1.0).fillna(0.0)
    local_score = (local_r * local_r).clip(0.0, 1.0).fillna(0.0)
    residual_score = (1.0 - enso_score - local_score).clip(0.05, 1.0)
    source_mask = (enso_score.lat >= -10) & (enso_score.lat <= 10) & (enso_score.lon >= 120) & (enso_score.lon <= 280)
    direct = enso_score.where(source_mask, 0.0)
    remote = enso_score.where(~source_mask, 0.0)
    total = direct + remote + local_score + residual_score
    shares = xr.Dataset(
        {
            "direct_share": direct / total,
            "remote_share": remote / total,
            "local_share": local_score / total,
            "residual_share": residual_score / total,
        }
    )
    stack = xr.concat([shares.direct_share, shares.remote_share, shares.local_share], dim="driver")
    ordered = np.sort(stack.values, axis=0)
    maximum = np.nanmax(stack.values, axis=0)
    regime = np.nanargmax(stack.fillna(-1).values, axis=0).astype(np.int8)
    mixed = (maximum < 0.50) | ((ordered[-1] - ordered[-2]) < 0.15)
    regime[mixed] = 3
    regime[~np.isfinite(sst.isel(time=-1).values)] = -1

    baseline = sst.sel(time=slice(BASELINE_START, BASELINE_END))
    threshold = baseline.groupby("time.month").quantile(0.9, dim="time")
    target = sst.sel(time=slice("2023-05", "2024-08"))
    target_threshold = threshold.sel(month=xr.DataArray(target.time.dt.month, dims="time")).drop_vars("month")
    intensity = (target - target_threshold).where(target > target_threshold, 0.0).mean("time", skipna=True)
    monthly_excess = (sst - threshold.sel(month=xr.DataArray(sst.time.dt.month, dims="time")).drop_vars("month")).clip(min=0.0)

    result = xr.Dataset(
        {
            **shares.data_vars,
            "driver_regime": (("lat", "lon"), regime),
            "mhw_intensity_2023_24": intensity,
        }
    )
    result.attrs.update(
        {
            "driver_codes": "0=direct ENSO, 1=remote ENSO association, 2=basin-local persistence, 3=mixed, -1=land",
            "driver_method": "squared historical SST-Nino3.4 correlation and lag-1 persistence after removing Nino3.4",
            "mhw_definition": "monthly SST above the local calendar-month 90th percentile for 1985-2014",
        }
    )
    result.to_netcdf(output / "figure3_driver_regime_and_mhw_intensity.nc")

    rows = []
    for basin, bounds in BASINS.items():
        mask = basin_mask(result.lat, result.lon, bounds)
        weights = intensity.where(mask).clip(min=0.0)
        if float(weights.sum(skipna=True)) <= 0:
            weights = xr.ones_like(intensity).where(mask)
        base_series = weighted_mean(monthly_excess.where(mask))
        event_value = float(base_series.sel(time=slice("2023-05", "2024-08")).mean())
        comparable_windows = []
        for start_year in range(1985, 2014):
            value = base_series.sel(time=slice(f"{start_year}-05", f"{start_year + 1}-08")).mean()
            comparable_windows.append(float(value))
        window_values = np.asarray(comparable_windows, dtype=float)
        intensity_z = (event_value - float(np.nanmean(window_values))) / max(float(np.nanstd(window_values)), 1e-8)
        row = {"Basin": basin, "MHW_intensity_C": event_value, "MHW_intensity_z": intensity_z}
        for name in ("direct", "remote", "local", "residual"):
            row[f"{name}_share"] = float(result[f"{name}_share"].weighted(weights.fillna(0)).mean())
        rows.append(row)
    basin_shares = pd.DataFrame(rows)
    basin_shares.to_csv(output / "figure3_basin_driver_contributions.csv", index=False)
    build_local_mhw_activity(monthly_excess, basin_shares, output)
    return result, basin_shares


def build_local_mhw_activity(
    monthly_excess: xr.DataArray,
    basin_shares: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, float | int | list[str]]]:
    share_table = basin_shares.set_index("Basin")
    local_basins = [
        basin
        for basin in BASINS
        if basin != "Global 60S-60N"
        and share_table.loc[basin, "local_share"] > 0.50
        and share_table.loc[basin, "local_share"]
        == share_table.loc[
            basin, ["direct_share", "remote_share", "local_share", "residual_share"]
        ].max()
    ]
    days = xr.DataArray(
        pd.DatetimeIndex(monthly_excess.time.values).days_in_month.astype(np.float32),
        coords={"time": monthly_excess.time},
        dims="time",
    )
    rows: list[dict[str, float | int | str]] = []
    for basin in local_basins:
        mask = basin_mask(monthly_excess.lat, monthly_excess.lon, BASINS[basin])
        monthly_activity = weighted_mean(monthly_excess.where(mask)) * days
        annual = monthly_activity.groupby("time.year").sum("time", skipna=True)
        annual = annual.sel(year=slice(1991, 2024))
        for year, value in zip(annual.year.values, annual.values, strict=True):
            rows.append(
                {
                    "Year": int(year),
                    "Series": basin,
                    "MHW_activity_C_days": float(value),
                }
            )
    activity = pd.DataFrame(rows)
    multi_basin = (
        activity.groupby("Year", as_index=False).MHW_activity_C_days.mean()
        .assign(Series="Local-dominated basin mean")
    )
    activity = pd.concat([activity, multi_basin], ignore_index=True)
    activity.to_csv(output / "figure3_local_mhw_activity.csv", index=False)

    years = multi_basin.Year.to_numpy(dtype=float)
    values = multi_basin.MHW_activity_C_days.to_numpy(dtype=float)
    linear = stats.linregress(years, values)
    kendall = stats.kendalltau(years, values)
    theil = stats.theilslopes(values, years, alpha=0.95)
    result: dict[str, float | int | list[str]] = {
        "year_start": int(years.min()),
        "year_end": int(years.max()),
        "year_count": len(years),
        "local_dominated_basins": local_basins,
        "linear_slope_C_days_per_decade": float(linear.slope * 10.0),
        "pearson_r": float(linear.rvalue),
        "linear_p": float(linear.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "theil_sen_slope_C_days_per_decade": float(theil.slope * 10.0),
        "theil_sen_ci_low_C_days_per_decade": float(theil.low_slope * 10.0),
        "theil_sen_ci_high_C_days_per_decade": float(theil.high_slope * 10.0),
        "definition": (
            "For each local-dominated basin, monthly area-weighted positive SST excess above "
            "the local 1985-2014 calendar-month 90th percentile is multiplied by calendar-month "
            "days and summed by year. The plotted multi-basin series is the equal-basin mean."
        ),
    }
    (output / "figure3_local_mhw_activity_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return activity, result


def build_local_process_skill_samples(
    basin_shares: pd.DataFrame,
    output: Path,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | list[str]]]:
    if not MHW_REFERENCE.is_file() or not COMMON_EVALUATION_MASK.is_file():
        raise FileNotFoundError("Figure 3 local-process analysis requires the locked MHW reference and common mask")
    with np.load(MHW_REFERENCE) as reference:
        observations = {
            str(month): np.asarray(value, dtype=np.float32)
            for month, value in zip(reference["months"], reference["observations"], strict=True)
        }
        threshold = np.asarray(reference["threshold"], dtype=np.float32)
    with np.load(COMMON_EVALUATION_MASK) as mask_data:
        static_mask = np.asarray(mask_data["mask"], dtype=bool)

    shares = basin_shares.set_index("Basin")
    local_basins = [
        basin
        for basin in BASINS
        if basin != "Global 60S-60N"
        and shares.loc[basin, "local_share"] > 0.50
        and shares.loc[basin, "local_share"]
        == shares.loc[basin, ["direct_share", "remote_share", "local_share", "residual_share"]].max()
    ]
    evaluation_basins = local_basins
    event_table = pd.read_csv(
        PAPER_DIR / "output/Figure1/supplemental_materials/Figure1_decoupling_event_summary_lead_all_NMME_ERSST.csv"
    ).set_index("Event")
    forecast_cache: dict[str, dict[int, np.ndarray]] = {}
    area = np.broadcast_to(
        np.cos(np.deg2rad(COMMON_LAT))[:, None], (len(COMMON_LAT), len(COMMON_LON))
    ).copy()
    event_rows = []
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        event_info = event_table.loc[event]
        targets = pd.date_range(event_info.Start_Month, event_info.End_Month, freq="MS")
        for lead in LEADS:
            values: dict[str, list[tuple[float, float]]] = {basin: [] for basin in evaluation_basins}
            for target in targets:
                initialization = target - pd.DateOffset(months=lead)
                key = f"{initialization:%Y-%m}"
                if key not in forecast_cache:
                    forecast_cache[key] = load_nmme_leads(initialization)
                forecast = forecast_cache[key].get(lead)
                target_key = f"{target:%Y-%m}"
                if forecast is None or target_key not in observations:
                    continue
                observation = observations[target_key]
                target_threshold = threshold[target.month - 1]
                event_mask = (observation > target_threshold).astype(np.float32)
                forecast_excess = forecast - target_threshold
                observed_excess = observation - target_threshold
                for basin in evaluation_basins:
                    south, north, west, east = BASINS[basin]
                    spatial = (
                        static_mask
                        & (COMMON_LAT[:, None] >= south)
                        & (COMMON_LAT[:, None] < north)
                        & (COMMON_LON[None, :] >= west)
                        & (COMMON_LON[None, :] < east)
                    )
                    weights = area * spatial
                    values[basin].append(
                        (
                            weighted_array_auc(event_mask, forecast_excess, weights),
                            weighted_array_correlation(forecast_excess, observed_excess, weights),
                        )
                    )
            for basin, samples in values.items():
                array = np.asarray(samples, dtype=float)
                if len(array) != len(targets):
                    raise RuntimeError(f"Incomplete local-process samples for {event}, lead {lead}, {basin}")
                event_rows.append(
                    {
                        "Event": event,
                        "Lead": lead,
                        "Basin": basin,
                        "AUC": float(np.nanmean(array[:, 0])),
                        "Local_ocean_pattern_error": float(1.0 - np.nanmean(array[:, 1])),
                        "Valid_months": int(np.isfinite(array[:, 0]).sum()),
                        "Local_share": float(shares.loc[basin, "local_share"]),
                    }
                )
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(output / "figure3_local_process_event_lead_basin.csv", index=False)

    historical = (
        event_frame[event_frame.Event.isin(HISTORICAL_FORECAST_EVENTS)]
        .groupby(["Lead", "Basin"], as_index=False)
        .agg(
            Historical_AUC=("AUC", "mean"),
            Historical_local_ocean_pattern_error=("Local_ocean_pattern_error", "mean"),
        )
    )
    target = event_frame[event_frame.Event == TARGET_EVENT].rename(
        columns={
            "AUC": "Target_AUC",
            "Local_ocean_pattern_error": "Target_local_ocean_pattern_error",
        }
    )
    samples = historical.merge(
        target[["Lead", "Basin", "Target_AUC", "Target_local_ocean_pattern_error", "Local_share"]],
        on=["Lead", "Basin"],
        validate="one_to_one",
    )
    samples["Local_ocean_pattern_error_increase"] = (
        samples.Target_local_ocean_pattern_error - samples.Historical_local_ocean_pattern_error
    )
    samples["Total_skill_loss"] = samples.Historical_AUC - samples.Target_AUC
    samples = samples.sort_values(["Basin", "Lead"])
    samples.to_csv(output / "figure3_local_process_samples.csv", index=False)

    x = samples.Local_ocean_pattern_error_increase.to_numpy()
    y = samples.Total_skill_loss.to_numpy()
    fit = stats.linregress(x, y)
    basins = sorted(samples.Basin.unique())
    x_matrix = samples.pivot(index="Basin", columns="Lead", values="Local_ocean_pattern_error_increase").loc[basins, list(LEADS)].to_numpy()
    y_matrix = samples.pivot(index="Basin", columns="Lead", values="Total_skill_loss").loc[basins, list(LEADS)].to_numpy()
    permutation_r = []
    for permutation in itertools.permutations(range(len(basins))):
        permutation_r.append(stats.pearsonr(x_matrix[list(permutation)].ravel(), y_matrix.ravel()).statistic)
    permutation_r = np.asarray(permutation_r)
    permutation_p = float((1 + np.sum(np.abs(permutation_r) >= abs(fit.rvalue))) / (len(permutation_r) + 1))
    rng = np.random.default_rng(seed)
    bootstrap_r = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(basins), len(basins))
        x_sample = x_matrix[indices].ravel()
        y_sample = y_matrix[indices].ravel()
        if np.std(x_sample) > 1e-10 and np.std(y_sample) > 1e-10:
            bootstrap_r.append(stats.pearsonr(x_sample, y_sample).statistic)
    low, high = np.nanpercentile(bootstrap_r, (2.5, 97.5))
    result: dict[str, float | int | list[str]] = {
        "sample_count": len(samples),
        "basin_count": len(basins),
        "lead_months": list(LEADS),
        "local_dominated_basins": basins,
        "pearson_r": float(fit.rvalue),
        "ordinary_p": float(fit.pvalue),
        "basin_block_permutation_p": permutation_p,
        "basin_block_bootstrap_ci_low": float(low),
        "basin_block_bootstrap_ci_high": float(high),
        "definition": (
            "Target minus historical canonical mean of 1-r between forecast and observed "
            "SST threshold-excess patterns; skill loss is historical mean AUC minus target AUC "
            "for the same basin and lead."
        ),
    }
    (output / "figure3_local_process_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return samples, result


def map_bridge_metrics(variable: str) -> pd.DataFrame:
    root = PAPER_DIR / "teleconnection_data_products/maps"
    obs = load_event_map(root / f"{variable}_anomaly_OBS_ERA5.nc", f"{variable}_anomaly")
    rows = []
    for lead in LEADS:
        forecast = load_event_map(root / f"{variable}_anomaly_NMME_NMME_lead{lead:02d}.nc", f"{variable}_anomaly")
        for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
            for bridge, bounds in BRIDGES.items():
                rows.append(
                    {
                        "Event": event,
                        "Lead": lead,
                        "Bridge": bridge,
                        "Pattern_skill": field_correlation(obs.sel(event=event), forecast.sel(event=event), bounds),
                    }
                )
    return pd.DataFrame(rows)


def build_teleconnection_and_path(nino: pd.DataFrame, basin_shares: pd.DataFrame, output: Path, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z_skill = map_bridge_metrics("z200")
    p_skill = map_bridge_metrics("precipitation").rename(columns={"Pattern_skill": "Convection_skill"})
    source = pd.read_csv(PAPER_DIR / "Figure2_ENSO-forecast/output/Figure2_ENSO_source_region_pattern_correlation.csv")
    source = source.rename(columns={"Event_Label": "Event", "Pattern_Correlation": "Source_skill"})[["Event", "Lead", "Source_skill"]]
    raw = pd.read_csv(PAPER_DIR / "output/Figure1/candidate09_metrics/Figure1_candidate09_metrics_by_init_lead.csv")
    raw = raw[(raw.Method == "NMME ensemble mean") & raw.Lead_Month.isin(LEADS)].copy()
    raw["Target"] = pd.to_datetime(raw.Target_Month)
    auc_rows = []
    source_events = pd.read_csv(PAPER_DIR / "output/Figure1/supplemental_materials/Figure1_decoupling_event_summary_lead_all_NMME_ERSST.csv")
    for event in (*HISTORICAL_FORECAST_EVENTS, TARGET_EVENT):
        item = source_events[source_events.Event == event].iloc[0]
        for lead in LEADS:
            selected = raw[(raw.Lead_Month == lead) & raw.Target.between(pd.Timestamp(item.Start_Month), pd.Timestamp(item.End_Month))]
            auc_rows.append({"Event": event, "Lead": lead, "AUC": float(selected.AUC.mean())})
    auc = pd.DataFrame(auc_rows)
    path_rows = z_skill.merge(p_skill, on=["Event", "Lead", "Bridge"]).merge(source, on=["Event", "Lead"]).merge(auc, on=["Event", "Lead"])
    path_rows["Remote_predictable_component"] = path_rows.Source_skill * path_rows.Pattern_skill
    path_rows["Local_error"] = 1.0 - path_rows.Convection_skill.clip(-1.0, 1.0)
    path_rows.to_csv(output / "figure4_path_samples.csv", index=False)

    efficiency = z_skill.groupby(["Event", "Bridge"], as_index=False).Pattern_skill.mean()
    efficiency["Z_score"] = efficiency.groupby("Bridge").Pattern_skill.transform(
        lambda values: (values - values.mean()) / max(float(values.std(ddof=0)), 1e-9)
    )
    matrix_rows = []
    for bridge in BRIDGES:
        group = efficiency[efficiency.Bridge == bridge]
        historical = float(group[group.Event.isin(HISTORICAL_FORECAST_EVENTS)].Z_score.mean())
        target_value = float(group[group.Event == TARGET_EVENT].Z_score.iloc[0])
        matrix_rows.append(
            {"Bridge": bridge, "Historical": historical, "2023/24": target_value, "Difference": target_value - historical}
        )
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(output / "figure3_teleconnection_efficiency.csv", index=False)

    def zscore(values: np.ndarray) -> np.ndarray:
        return (values - np.nanmean(values)) / max(float(np.nanstd(values)), 1e-9)

    def coefficients(frame: pd.DataFrame) -> dict[str, float]:
        clean = frame.dropna().copy()
        source_z = zscore(clean.Source_skill.to_numpy())
        tele_z = zscore(clean.Pattern_skill.to_numpy())
        remote_z = zscore(clean.Remote_predictable_component.to_numpy())
        local_z = zscore(clean.Local_error.to_numpy())
        auc_z = zscore(clean.AUC.to_numpy())
        first = np.linalg.lstsq(np.column_stack((source_z, local_z, np.ones(len(clean)))), tele_z, rcond=None)[0]
        second = np.linalg.lstsq(np.column_stack((tele_z, local_z, np.ones(len(clean)))), remote_z, rcond=None)[0]
        third = np.linalg.lstsq(np.column_stack((remote_z, local_z, np.ones(len(clean)))), auc_z, rcond=None)[0]
        fitted = np.column_stack((remote_z, local_z, np.ones(len(clean)))) @ third
        return {
            "source_to_teleconnection": float(first[0]),
            "local_to_teleconnection": float(first[1]),
            "teleconnection_to_remote": float(second[0]),
            "local_to_remote": float(second[1]),
            "remote_to_skill": float(third[0]),
            "local_to_skill": float(third[1]),
            "skill_r2": float(1.0 - np.sum((auc_z - fitted) ** 2) / np.sum((auc_z - auc_z.mean()) ** 2)),
            "skill_residual_std": float(np.std(auc_z - fitted)),
        }

    point = coefficients(path_rows)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point if key != "skill_residual_std"}
    events = path_rows.Event.unique()
    for _ in range(bootstrap):
        sampled = rng.choice(events, size=len(events), replace=True)
        frame = pd.concat([path_rows[path_rows.Event == event] for event in sampled], ignore_index=True)
        try:
            result = coefficients(frame)
        except (np.linalg.LinAlgError, ValueError):
            continue
        for key in samples:
            samples[key].append(result[key])
    coefficient_rows = []
    for key, value in point.items():
        if key == "skill_residual_std":
            continue
        values = samples[key]
        low, high = np.nanpercentile(values, (2.5, 97.5)) if values else (np.nan, np.nan)
        coefficient_rows.append({"Path": key, "Coefficient": value, "CI_low": low, "CI_high": high})
    coefficients_frame = pd.DataFrame(coefficient_rows)
    coefficients_frame.to_csv(output / "figure4_path_coefficients.csv", index=False)

    basin_skill = pd.read_csv(
        PAPER_DIR / "Figures/Figure3/Figure3_basin_mhw_skill.csv"
    ).set_index("basin")
    event_relation = pd.read_csv(output / "figure1_event_relation.csv")
    historical_relation = event_relation[event_relation.Event != TARGET_EVENT]
    target_relation = event_relation[event_relation.Event == TARGET_EVENT].iloc[0]
    relation_fit = stats.linregress(
        historical_relation.Peak_Nino34, historical_relation.Mean_AUC
    )
    expected_auc = float(
        relation_fit.intercept + relation_fit.slope * target_relation.Peak_Nino34
    )
    observed_auc = float(target_relation.Mean_AUC)
    figure1_residual = observed_auc - expected_auc
    global_residual_loss = max(0.0, -figure1_residual)
    source_deficit = max(
        0.0,
        float(
            source[source.Event.isin(HISTORICAL_FORECAST_EVENTS)].Source_skill.mean()
            - source[source.Event == TARGET_EVENT].Source_skill.mean()
        ),
    )
    share_table = basin_shares.set_index("Basin")
    mapping = {
        "North Pacific": "PNA (N. Pacific)", "South Pacific": "PSA (S. Pacific)", "Indian Ocean": "Indian bridge",
        "North Atlantic": "Atlantic bridge", "Tropical Atlantic": "Atlantic bridge", "South Atlantic": "Atlantic bridge",
    }
    explained_fraction = float(np.clip(point["skill_r2"], 0.0, 1.0))
    raw_rows = []
    for basin in mapping:
        bridge = mapping[basin]
        bridge_rows = path_rows[path_rows.Bridge == bridge]
        history_rows = bridge_rows[bridge_rows.Event.isin(HISTORICAL_FORECAST_EVENTS)]
        target_rows = bridge_rows[bridge_rows.Event == TARGET_EVENT]
        tele_deficit = max(0.0, float(history_rows.Pattern_skill.mean() - target_rows.Pattern_skill.mean()))
        local_deficit = max(0.0, float(target_rows.Local_error.mean() - history_rows.Local_error.mean()))
        local_deficit *= float(share_table.loc[basin, "local_share"])
        raw_scores = np.array([source_deficit, tele_deficit, local_deficit])
        observed_basin_loss = max(
            0.0,
            float(basin_skill.loc[basin, "Canonical"] - basin_skill.loc[basin, "2023/24"]),
        )
        raw_rows.append(
            {
                "Basin": basin,
                "Observed_basin_skill_loss": observed_basin_loss,
                "Raw_scores": raw_scores,
                "Local_error_index": local_deficit,
            }
        )
    spatial_denominator = sum(row["Observed_basin_skill_loss"] for row in raw_rows)
    if spatial_denominator <= 0 or global_residual_loss <= 0:
        raise RuntimeError("Figure 4 requires a positive Figure 1 residual loss and basin deficits")
    attribution_rows = []
    for row in raw_rows:
        spatial_share = row["Observed_basin_skill_loss"] / spatial_denominator
        total_loss = global_residual_loss * spatial_share
        raw_scores = row["Raw_scores"]
        resolved_loss = total_loss * explained_fraction
        pieces = raw_scores / max(raw_scores.sum(), 1e-12) * resolved_loss
        attribution_rows.append(
            {
                "Basin": row["Basin"],
                "Total_skill_loss": total_loss,
                "ENSO_source_error": pieces[0],
                "Teleconnection_error": pieces[1],
                "Basin_local_error": pieces[2],
                "Irreducible": total_loss - resolved_loss,
                "Local_error_index": row["Local_error_index"],
                "Observed_basin_skill_loss": row["Observed_basin_skill_loss"],
                "Spatial_share_of_global_residual": spatial_share,
                "Expected_global_AUC": expected_auc,
                "Observed_global_AUC": observed_auc,
                "Figure1_residual": figure1_residual,
            }
        )
    attribution = pd.DataFrame(attribution_rows)
    global_values = attribution[
        ["ENSO_source_error", "Teleconnection_error", "Basin_local_error", "Irreducible"]
    ].sum()
    global_row = {
        "Basin": "Global 60S-60N",
        "Total_skill_loss": global_residual_loss,
        **global_values.to_dict(),
        "Local_error_index": float(
            np.average(
                attribution.Local_error_index,
                weights=attribution.Spatial_share_of_global_residual,
            )
        ),
        "Observed_basin_skill_loss": float(
            basin_skill.loc["Global 60S-60N", "Canonical"]
            - basin_skill.loc["Global 60S-60N", "2023/24"]
        ),
        "Spatial_share_of_global_residual": 1.0,
        "Expected_global_AUC": expected_auc,
        "Observed_global_AUC": observed_auc,
        "Figure1_residual": figure1_residual,
    }
    attribution = pd.concat([attribution, pd.DataFrame([global_row])], ignore_index=True)
    attribution.to_csv(output / "figure4_skill_loss_attribution.csv", index=False)
    mechanism_columns = (
        "ENSO_source_error",
        "Teleconnection_error",
        "Basin_local_error",
        "Irreducible",
    )
    explained = float(sum(global_row[name] for name in mechanism_columns[:3]))
    summary = {
        "expected_global_auc_from_figure1_relationship": expected_auc,
        "observed_2023_24_global_auc": observed_auc,
        "figure1_residual": figure1_residual,
        "allocated_loss_magnitude": global_residual_loss,
        "path_model_r2_used_for_resolved_fraction": explained_fraction,
        "explained_association_fraction": explained / global_residual_loss,
        "unresolved_fraction": float(global_row["Irreducible"] / global_residual_loss),
        "mechanism_percent": {
            name: float(100.0 * global_row[name] / global_residual_loss)
            for name in mechanism_columns
        },
        "basin_percent": {
            row.Basin: float(100.0 * row.Spatial_share_of_global_residual)
            for row in attribution[attribution.Basin != "Global 60S-60N"].itertuples()
        },
        "interpretation": (
            "Association-based allocation. Basin weights use positive comparable-event minus "
            "2023/24 basin AUC deficits; mechanism weights use independently computed source, "
            "teleconnection and local-process forecast deficits. The descriptive path-model R2 "
            "sets the resolved fraction; one minus R2 is retained as unresolved."
        ),
    }
    (output / "figure4_attribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return matrix, path_rows, attribution


def write_provenance(output: Path, args: argparse.Namespace, process_metadata: dict[str, object]) -> None:
    provenance = {
        "created_by": str(Path(__file__).resolve()),
        "climatology": f"{BASELINE_START} to {BASELINE_END}",
        "lead_aggregation": "target-aligned NMME lead months 1-9 for Figure 1",
        "download_manifest": str((RAW / "download_manifest.json").resolve()),
        "local_archives": {
            "ORAS5": str((PROJECT_DIR / "data/ORAS5").resolve()),
            "ERA5": str((PAPER_DIR / "teleconnection_data/observations/ERA5").resolve()),
            "NMME": str((PAPER_DIR / "teleconnection_data/NMME").resolve()),
        },
        "figure2_process_forecast_skill": process_metadata,
        "important_boundaries": [
            "Figure 2 NMME convection error uses precipitation because the archived 2023 NMME OLR field is all missing.",
            "Figure 2 source-region process maps and pattern correlations use the equal mean of target-aligned leads 1-9.",
            "Figure 3 driver classes are association-based statistical regimes, not a closed mixed-layer heat budget.",
            "Figure 3 annual MHW activity is a monthly-resolution threshold-excess proxy combining affected area, intensity and persistence.",
            "Figure 4 allocates the Figure 1 expected-minus-observed AUC residual by association; it is not causal identification.",
        ],
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
    }
    (output / "calculation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nino = parse_cpc_nino(RAW / "cpc_ersst5_nino_monthly_1991_2020_base.txt")
    build_figure1(nino, args.output_dir)
    process_metadata = build_figure2_process_forecast_skill(args.output_dir)
    if not args.skip_source_components:
        source_component_shares(load_source_components(args.output_dir), nino, args.output_dir)
    build_figure2_summaries(args.output_dir)
    _, basin_shares = build_driver_map(nino, args.output_dir)
    build_teleconnection_and_path(nino, basin_shares, args.output_dir, args.bootstrap, args.seed)
    write_provenance(args.output_dir, args, process_metadata)
    print(f"[Complete] Derived real-data products: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
