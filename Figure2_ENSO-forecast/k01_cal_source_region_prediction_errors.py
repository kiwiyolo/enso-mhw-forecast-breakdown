#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: k01_cal_source_region_prediction_errors.py
# Date: 2026-06-09
# Copyright (c) 2026
#
"""
Calculate ENSO source-region prediction errors from NMME SST forecasts.

The workflow compares NMME ensemble-mean SST anomaly forecasts with an
observational SST anomaly reference over the tropical Pacific. It exports the
time-series, lead-dependent errors, event-peak spatial patterns, ENSO flavor
indices, and metadata needed by k02_plot_source_region_prediction_errors.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


NMME_MODELS = ("COLA-RSMAS-CCSM4", "NASA-GEOSS2S", "GFDL-SPEAR", "NCEP-CFSv2")
NINO_REGIONS = {
    "Nino1+2": (-10.0, 0.0, 270.0, 280.0),
    "Nino3": (-5.0, 5.0, 210.0, 270.0),
    "Nino3.4": (-5.0, 5.0, 190.0, 240.0),
    "Nino4": (-5.0, 5.0, 160.0, 210.0),
}
EVENT_WINDOWS = {
    "1997/98": ("1997-01", "1998-12"),
    "2015/16": ("2015-01", "2016-12"),
    "2023/24": ("2023-01", "2024-12"),
}
EVENT_PEAK_SEASONS = {
    "1997/98": ("1997-12", "1998-02"),
    "2015/16": ("2015-12", "2016-02"),
    "2023/24": ("2023-12", "2024-02"),
}
PACIFIC_BOUNDS = (-20.0, 20.0, 120.0, 280.0)


@dataclass(frozen=True)
class CalculationConfig:
    """Configuration values for the source-region diagnostic calculation."""

    work_dir: Path
    data_dir: Path
    output_dir: Path
    obs_label: str
    obs_path: Path
    baseline_start: str
    baseline_end: str
    leads: tuple[int, ...]
    map_event: str
    map_lead: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    forecast_min_valid_sst: float
    forecast_min_sources_per_cell: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calculation workflow."""

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]
    parser = argparse.ArgumentParser(description="Calculate ENSO source-region prediction errors for Figure 2.")
    parser.add_argument("--work-dir", default=str(script_dir), help="Figure working directory.")
    parser.add_argument("--data-dir", default=str(project_dir / "data"), help="Input SST data directory.")
    parser.add_argument("--output-dir", default=str(script_dir / "output"), help="Output directory for diagnostic products.")
    parser.add_argument("--obs-label", default="ERSST", help="Observation label: ERSST or HadISST.")
    parser.add_argument("--obs-path", default="", help="Optional explicit observation NetCDF path.")
    parser.add_argument("--baseline", default="1991-2020", help="Monthly climatology baseline years.")
    parser.add_argument(
        "--leads",
        default="1,3,6,9",
        help="Comma-separated future lead months; t+1 reads NMME L=1.5, not the initialization-month L=0.5.",
    )
    parser.add_argument("--map-event", default="2023/24", help="Event label used for the row-3 map panels.")
    parser.add_argument("--map-lead", type=int, default=9, help="Lead month used for the row-3 forecast map.")
    parser.add_argument("--lat-min", type=float, default=PACIFIC_BOUNDS[0], help="Minimum latitude for tropical Pacific diagnostics.")
    parser.add_argument("--lat-max", type=float, default=PACIFIC_BOUNDS[1], help="Maximum latitude for tropical Pacific diagnostics.")
    parser.add_argument("--lon-min", type=float, default=PACIFIC_BOUNDS[2], help="Minimum longitude in 0-360 degrees.")
    parser.add_argument("--lon-max", type=float, default=PACIFIC_BOUNDS[3], help="Maximum longitude in 0-360 degrees.")
    parser.add_argument("--forecast-min-valid-sst", type=float, default=1.0, help="Minimum valid forecast SST in the tropical Pacific; lower values are treated as missing.")
    parser.add_argument(
        "--forecast-min-sources-per-cell",
        type=int,
        default=2,
        help="Minimum number of NMME models with a valid SST value at each grid cell.",
    )
    return parser.parse_args()


def parse_leads(text: str) -> tuple[int, ...]:
    """Parse lead-month values from a comma-separated string."""

    lead_values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not lead_values:
        raise ValueError("At least one lead month is required.")
    if any(lead < 1 for lead in lead_values):
        raise ValueError(f"Lead months must be positive integers: {lead_values}")
    return lead_values


def build_config(args: argparse.Namespace) -> CalculationConfig:
    """Build a typed calculation configuration from CLI arguments."""

    data_dir = Path(args.data_dir).resolve()
    obs_label = str(args.obs_label).strip()
    obs_path = Path(args.obs_path).resolve() if str(args.obs_path).strip() else observation_path_for_label(data_dir, obs_label)
    baseline_start, baseline_end = str(args.baseline).split("-", 1)
    return CalculationConfig(
        work_dir=Path(args.work_dir).resolve(),
        data_dir=data_dir,
        output_dir=Path(args.output_dir).resolve(),
        obs_label=obs_label,
        obs_path=obs_path,
        baseline_start=f"{int(baseline_start):04d}-01",
        baseline_end=f"{int(baseline_end):04d}-12",
        leads=parse_leads(args.leads),
        map_event=str(args.map_event),
        map_lead=int(args.map_lead),
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
        forecast_min_valid_sst=float(args.forecast_min_valid_sst),
        forecast_min_sources_per_cell=max(1, int(args.forecast_min_sources_per_cell)),
    )


def observation_path_for_label(data_dir: Path, obs_label: str) -> Path:
    """Return the default observation path for a supported observation label."""

    label = obs_label.upper()
    if label == "ERSST":
        return data_dir / "ersst_observation.nc"
    if label == "HADISST":
        return data_dir / "HadISST" / "HadISST_sst.nc"
    raise ValueError(f"Unsupported observation label: {obs_label}")


def month_range(start_month: str, end_month: str) -> list[str]:
    """Return an inclusive monthly range as YYYY-MM strings."""

    return [str(period) for period in pd.period_range(start_month, end_month, freq="M")]


def add_months(month_text: str, offset: int) -> str:
    """Add a signed integer month offset to a YYYY-MM string."""

    return str(pd.Period(month_text, freq="M") + int(offset))


def choose_data_var(ds: xr.Dataset) -> str:
    """Choose the SST-like variable from a dataset."""

    for name in ("sst", "sst_regridded", "tos"):
        if name in ds.data_vars:
            return name
    return next(iter(ds.data_vars))


def normalize_sst_units(da: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin SST values to Celsius when metadata indicates Kelvin."""

    units = str(da.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"}:
        out = da - 273.15
        out.attrs.update(da.attrs)
        out.attrs["units"] = "degree_C"
        return out
    return da


def normalize_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    """Rename spatial dimensions to lat/lon, convert longitude, and sort axes."""

    rename_map: dict[str, str] = {}
    for old_name in ("Y", "y", "latitude"):
        if old_name in da.dims:
            rename_map[old_name] = "lat"
            break
    for old_name in ("X", "x", "longitude"):
        if old_name in da.dims:
            rename_map[old_name] = "lon"
            break
    if rename_map:
        da = da.rename(rename_map)
    if "lon" in da.coords:
        lon_values = np.mod(np.asarray(da["lon"].values, dtype=float), 360.0)
        da = da.assign_coords(lon=lon_values).sortby("lon")
    if "lat" in da.coords:
        da = da.sortby("lat")
    return da


def select_lon_interval(da: xr.DataArray, lon_min: float, lon_max: float) -> xr.DataArray:
    """Select a longitude interval in 0-360 coordinates, including wrap-around intervals."""

    lon_min_norm = float(lon_min) % 360.0
    lon_max_norm = float(lon_max) % 360.0
    if lon_min_norm <= lon_max_norm:
        return da.sel(lon=slice(lon_min_norm, lon_max_norm))
    west = da.sel(lon=slice(lon_min_norm, 360.0))
    east = da.sel(lon=slice(0.0, lon_max_norm))
    return xr.concat([west, east], dim="lon")


def load_observation(config: CalculationConfig) -> xr.DataArray:
    """Load the observation SST field over the tropical Pacific domain."""

    with xr.open_dataset(config.obs_path) as ds:
        var_name = choose_data_var(ds)
        obs_da = normalize_spatial_dims(ds[var_name])
        obs_da = normalize_sst_units(obs_da)
        for depth_name in ("zlev", "depth", "lev", "level"):
            if depth_name in obs_da.dims:
                obs_da = obs_da.isel({depth_name: 0})
        obs_da = obs_da.where(obs_da > -100.0)
        obs_da = obs_da.sel(lat=slice(config.lat_min, config.lat_max))
        obs_da = select_lon_interval(obs_da, config.lon_min, config.lon_max)
        obs_da = obs_da.load()
    return obs_da


def compute_monthly_climatology(obs_da: xr.DataArray, config: CalculationConfig) -> xr.DataArray:
    """Compute monthly observed climatology for the configured baseline."""

    baseline_da = obs_da.sel(time=slice(config.baseline_start, config.baseline_end))
    return baseline_da.groupby("time.month").mean("time", skipna=True)


def target_months_for_events() -> list[str]:
    """Return the union of all months needed for the selected ENSO events."""

    months: list[str] = []
    for start_month, end_month in EVENT_WINDOWS.values():
        months.extend(month_range(start_month, end_month))
    return sorted(set(months))


def anomaly_for_month(field_da: xr.DataArray, target_month: str, climatology: xr.DataArray) -> xr.DataArray:
    """Subtract the observed monthly climatology from one target-month field."""

    month_number = int(target_month[5:7])
    return field_da - climatology.sel(month=month_number)


def observation_anomaly_for_month(obs_da: xr.DataArray, target_month: str, climatology: xr.DataArray) -> xr.DataArray | None:
    """Select and anomaly-transform one observed target month."""

    selected = obs_da.sel(time=slice(f"{target_month}-01", f"{target_month}-28"))
    if int(selected.sizes.get("time", 0)) == 0:
        return None
    field_da = selected.isel(time=0).drop_vars("time", errors="ignore")
    return anomaly_for_month(field_da, target_month, climatology)


def model_file_for_init(model_name: str, init_month: str, config: CalculationConfig) -> Path | None:
    """Find the NMME file for one model and initialization month."""

    model_dir = config.data_dir / model_name
    if not model_dir.exists():
        return None
    yyyymm = init_month.replace("-", "")
    matches = sorted(model_dir.glob(f"*{yyyymm}*.nc"))
    return matches[0] if matches else None


def start_coordinate_matches_file(ds: xr.Dataset, init_month: str) -> bool:
    """Reject files whose internal NMME start month disagrees with their name."""

    if "S" not in ds.coords:
        return True
    start = np.asarray(ds["S"].values, dtype=float).reshape(-1)
    if start.size == 0 or not np.isfinite(start[0]):
        return False
    period = pd.Period(init_month, freq="M")
    expected = (period.year - 1960) * 12 + (period.month - 1)
    return bool(abs(float(start[0]) - float(expected)) <= 0.1)


def forecast_members_for_model(model_name: str, target_month: str, lead: int, obs_template: xr.DataArray, config: CalculationConfig) -> xr.DataArray | None:
    """Load all members from one NMME model verifying one target month and lead."""

    # NMME L=0.5 verifies the initialization month. The common t+lead
    # contract therefore uses array index=lead and initialization=target-lead.
    init_month = add_months(target_month, -int(lead))
    file_path = model_file_for_init(model_name, init_month, config)
    if file_path is None:
        return None
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            if not start_coordinate_matches_file(ds, init_month):
                return None
            var_name = choose_data_var(ds)
            data_var = normalize_spatial_dims(ds[var_name])
            data_var = normalize_sst_units(data_var)
            if "S" in data_var.dims:
                data_var = data_var.isel(S=0)
            if "L" not in data_var.dims or int(data_var.sizes["L"]) <= int(lead):
                return None
            data_var = data_var.isel(L=int(lead))
            if "M" not in data_var.dims:
                data_var = data_var.expand_dims(M=[0])
            data_var = data_var.transpose("M", "lat", "lon")
            data_var = data_var.sel(lat=slice(config.lat_min, config.lat_max))
            data_var = select_lon_interval(data_var, config.lon_min, config.lon_max)
            data_var = data_var.where(data_var > float(config.forecast_min_valid_sst))
            data_var = data_var.interp(lat=obs_template["lat"], lon=obs_template["lon"], method="nearest")
            if not bool(np.isfinite(data_var).any()):
                return None
            member_labels = [f"{model_name}_m{member_index + 1:02d}" for member_index in range(int(data_var.sizes["M"]))]
            data_var = data_var.rename({"M": "member"}).assign_coords(member=member_labels)
            return data_var.load()
    except Exception:
        return None


def forecast_anomaly_for_month(target_month: str, lead: int, obs_template: xr.DataArray, climatology: xr.DataArray, config: CalculationConfig) -> tuple[xr.DataArray | None, int, int]:
    """Build an NMME ensemble-mean anomaly for one target month and lead."""

    member_arrays: list[xr.DataArray] = []
    source_count = 0
    for model_name in NMME_MODELS:
        model_members = forecast_members_for_model(model_name, target_month, lead, obs_template, config)
        if model_members is None:
            continue
        source_count += 1
        member_arrays.append(model_members)
    if not member_arrays:
        return None, 0, 0
    source_coverage = xr.concat(
        [members.notnull().any("member") for members in member_arrays],
        dim="source",
        coords="minimal",
        compat="override",
    ).sum("source")
    all_members = xr.concat(member_arrays, dim="member", coords="minimal", compat="override")
    forecast_mean = all_members.mean("member", skipna=True).where(
        source_coverage >= config.forecast_min_sources_per_cell
    )
    forecast_anomaly = anomaly_for_month(forecast_mean, target_month, climatology)
    return forecast_anomaly, int(all_members.sizes["member"]), int(source_count)


def area_weighted_mean(da: xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    """Compute a cosine-latitude weighted regional mean."""

    regional_da = da.sel(lat=slice(lat_min, lat_max))
    regional_da = select_lon_interval(regional_da, lon_min, lon_max)
    weights = xr.DataArray(np.cos(np.deg2rad(regional_da["lat"].values)), coords={"lat": regional_da["lat"]}, dims=("lat",))
    value = regional_da.weighted(weights).mean(("lat", "lon"), skipna=True)
    return float(value.values)


def nino_indices_from_anomaly(anomaly_da: xr.DataArray) -> dict[str, float]:
    """Calculate Niño-region indices from a tropical Pacific SST anomaly field."""

    index_values: dict[str, float] = {}
    for region_name, (lat_min, lat_max, lon_min, lon_max) in NINO_REGIONS.items():
        index_values[region_name] = area_weighted_mean(anomaly_da, lat_min, lat_max, lon_min, lon_max)
    index_values["EP_Index"] = index_values["Nino3"] - index_values["Nino4"]
    index_values["CP_Index"] = index_values["Nino4"] - index_values["Nino1+2"]
    return index_values


def event_label_for_month(target_month: str) -> str:
    """Return the configured event label containing one target month."""

    for event_label, (start_month, end_month) in EVENT_WINDOWS.items():
        if start_month <= target_month <= end_month:
            return event_label
    return "other"


def safe_correlation(left_values: Iterable[float], right_values: Iterable[float]) -> float:
    """Return a finite Pearson correlation or NaN when the sample is invalid."""

    left_array = np.asarray(list(left_values), dtype=float)
    right_array = np.asarray(list(right_values), dtype=float)
    valid_mask = np.isfinite(left_array) & np.isfinite(right_array)
    if int(valid_mask.sum()) < 2:
        return float("nan")
    if float(np.nanstd(left_array[valid_mask])) == 0.0 or float(np.nanstd(right_array[valid_mask])) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array[valid_mask], right_array[valid_mask])[0, 1])


def pattern_correlation(forecast_da: xr.DataArray, obs_da: xr.DataArray) -> float:
    """Calculate a spatial pattern correlation over finite tropical Pacific grid cells."""

    forecast_values = np.asarray(forecast_da.values, dtype=float).ravel()
    obs_values = np.asarray(obs_da.values, dtype=float).ravel()
    return safe_correlation(forecast_values, obs_values)


def calculate_timeseries(obs_da: xr.DataArray, climatology: xr.DataArray, config: CalculationConfig) -> pd.DataFrame:
    """Calculate monthly observed and forecast Niño indices for all events and leads."""

    rows: list[dict[str, object]] = []
    target_months = target_months_for_events()
    total_steps = len(target_months) * (1 + len(config.leads))
    progress = tqdm(total=total_steps, desc="Overall progress: calculate ENSO source-region time series", unit="step")
    for target_month in target_months:
        obs_anomaly = observation_anomaly_for_month(obs_da, target_month, climatology)
        if obs_anomaly is None:
            progress.update(1 + len(config.leads))
            continue
        obs_indices = nino_indices_from_anomaly(obs_anomaly)
        rows.append(
            {
                "Target_Month": target_month,
                "Date": f"{target_month}-01",
                "Event_Label": event_label_for_month(target_month),
                "Dataset": "OBS",
                "Observation_Label": config.obs_label,
                "Lead": 0,
                "Lead_Label": "OBS",
                "Member_Count": 1,
                "Source_Count": 1,
                **obs_indices,
            }
        )
        progress.update(1)
        for lead in config.leads:
            progress.set_postfix_str(f"{target_month} lead{lead:02d}")
            forecast_anomaly, member_count, source_count = forecast_anomaly_for_month(target_month, lead, obs_anomaly, climatology, config)
            if forecast_anomaly is not None:
                forecast_indices = nino_indices_from_anomaly(forecast_anomaly)
                rows.append(
                    {
                        "Target_Month": target_month,
                        "Date": f"{target_month}-01",
                        "Event_Label": event_label_for_month(target_month),
                        "Dataset": "NMME",
                        "Observation_Label": config.obs_label,
                        "Lead": int(lead),
                        "Lead_Label": f"lead{lead:02d}",
                        "Member_Count": int(member_count),
                        "Source_Count": int(source_count),
                        **forecast_indices,
                    }
                )
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def calculate_lead_errors(timeseries_df: pd.DataFrame, config: CalculationConfig) -> pd.DataFrame:
    """Calculate lead-dependent Niño3.4 bias, RMSE, ACC, and peak errors."""

    rows: list[dict[str, object]] = []
    for event_label in tqdm(EVENT_WINDOWS, desc="Step 2/5: lead-dependent Niño3.4 errors", unit="event"):
        event_obs = timeseries_df[(timeseries_df["Event_Label"] == event_label) & (timeseries_df["Dataset"] == "OBS")]
        for lead in config.leads:
            event_forecast = timeseries_df[
                (timeseries_df["Event_Label"] == event_label)
                & (timeseries_df["Dataset"] == "NMME")
                & (timeseries_df["Lead"] == int(lead))
            ]
            merged = event_obs[["Target_Month", "Nino3.4"]].merge(
                event_forecast[["Target_Month", "Nino3.4"]],
                on="Target_Month",
                suffixes=("_OBS", "_Forecast"),
            )
            difference = merged["Nino3.4_Forecast"] - merged["Nino3.4_OBS"]
            obs_peak_index = int(merged["Nino3.4_OBS"].idxmax()) if not merged.empty else -1
            forecast_peak_index = int(merged["Nino3.4_Forecast"].idxmax()) if not merged.empty else -1
            obs_peak_month = str(merged.loc[obs_peak_index, "Target_Month"]) if obs_peak_index >= 0 else ""
            forecast_peak_month = str(merged.loc[forecast_peak_index, "Target_Month"]) if forecast_peak_index >= 0 else ""
            peak_timing_error = (
                int(pd.Period(forecast_peak_month, freq="M").ordinal - pd.Period(obs_peak_month, freq="M").ordinal)
                if obs_peak_month and forecast_peak_month
                else math.nan
            )
            rows.append(
                {
                    "Event_Label": event_label,
                    "Lead": int(lead),
                    "Lead_Label": f"lead{lead:02d}",
                    "Nino34_Bias": float(difference.mean()) if len(difference) else math.nan,
                    "Nino34_RMSE": float(np.sqrt(np.nanmean(np.square(difference)))) if len(difference) else math.nan,
                    "Nino34_ACC": safe_correlation(merged["Nino3.4_OBS"], merged["Nino3.4_Forecast"]) if len(merged) else math.nan,
                    "OBS_Peak_Nino34": float(merged["Nino3.4_OBS"].max()) if len(merged) else math.nan,
                    "Forecast_Peak_Nino34": float(merged["Nino3.4_Forecast"].max()) if len(merged) else math.nan,
                    "Peak_Intensity_Error": float(merged["Nino3.4_Forecast"].max() - merged["Nino3.4_OBS"].max()) if len(merged) else math.nan,
                    "OBS_Peak_Month": obs_peak_month,
                    "Forecast_Peak_Month": forecast_peak_month,
                    "Peak_Timing_Error_Months": peak_timing_error,
                    "Sample_Months": int(len(merged)),
                }
            )
    return pd.DataFrame(rows)


def seasonal_mean_anomaly(months: list[str], dataset_name: str, lead: int, obs_da: xr.DataArray, climatology: xr.DataArray, config: CalculationConfig) -> tuple[xr.DataArray | None, int, int]:
    """Calculate a seasonal mean anomaly for observations or NMME forecasts."""

    anomaly_arrays: list[xr.DataArray] = []
    member_total = 0
    source_total = 0
    for target_month in months:
        if dataset_name == "OBS":
            obs_anomaly = observation_anomaly_for_month(obs_da, target_month, climatology)
            if obs_anomaly is not None:
                anomaly_arrays.append(obs_anomaly)
                member_total += 1
                source_total += 1
        else:
            obs_template = observation_anomaly_for_month(obs_da, target_month, climatology)
            if obs_template is None:
                continue
            forecast_anomaly, member_count, source_count = forecast_anomaly_for_month(target_month, lead, obs_template, climatology, config)
            if forecast_anomaly is not None:
                anomaly_arrays.append(forecast_anomaly)
                member_total += member_count
                source_total += source_count
    if not anomaly_arrays:
        return None, 0, 0
    season_anomaly = xr.concat(anomaly_arrays, dim="season_month").mean("season_month", skipna=True)
    return season_anomaly, int(member_total), int(source_total)


def calculate_event_patterns(obs_da: xr.DataArray, climatology: xr.DataArray, config: CalculationConfig) -> tuple[pd.DataFrame, pd.DataFrame, xr.Dataset]:
    """Calculate seasonal pattern correlations, flavor indices, and the selected map fields."""

    flavor_rows: list[dict[str, object]] = []
    pattern_rows: list[dict[str, object]] = []
    map_event_labels: list[str] = []
    map_peak_seasons: list[str] = []
    map_obs_fields: list[xr.DataArray] = []
    map_forecast_fields: list[xr.DataArray] = []
    map_error_fields: list[xr.DataArray] = []
    for event_label, (start_month, end_month) in tqdm(EVENT_PEAK_SEASONS.items(), desc="Step 3/5: event-peak spatial patterns", unit="event"):
        peak_months = month_range(start_month, end_month)
        obs_season, _, _ = seasonal_mean_anomaly(peak_months, "OBS", 0, obs_da, climatology, config)
        if obs_season is None:
            continue
        obs_indices = nino_indices_from_anomaly(obs_season)
        flavor_rows.append(
            {
                "Event_Label": event_label,
                "Dataset": "OBS",
                "Lead": 0,
                "Lead_Label": "OBS",
                "Peak_Season": f"{start_month} to {end_month}",
                "Member_Count": 1,
                "Source_Count": 1,
                **obs_indices,
            }
        )
        for lead in config.leads:
            forecast_season, member_count, source_count = seasonal_mean_anomaly(peak_months, "NMME", lead, obs_da, climatology, config)
            if forecast_season is None:
                continue
            forecast_indices = nino_indices_from_anomaly(forecast_season)
            pattern_value = pattern_correlation(forecast_season, obs_season)
            pattern_rows.append(
                {
                    "Event_Label": event_label,
                    "Lead": int(lead),
                    "Lead_Label": f"lead{lead:02d}",
                    "Peak_Season": f"{start_month} to {end_month}",
                    "Pattern_Correlation": float(pattern_value),
                    "Member_Count": int(member_count),
                    "Source_Count": int(source_count),
                }
            )
            flavor_rows.append(
                {
                    "Event_Label": event_label,
                    "Dataset": "NMME",
                    "Lead": int(lead),
                    "Lead_Label": f"lead{lead:02d}",
                    "Peak_Season": f"{start_month} to {end_month}",
                    "Member_Count": int(member_count),
                    "Source_Count": int(source_count),
                    **forecast_indices,
                }
            )
            if int(lead) == int(config.map_lead):
                map_event_labels.append(event_label)
                map_peak_seasons.append(f"{start_month} to {end_month}")
                map_obs_fields.append(obs_season.astype(np.float32))
                map_forecast_fields.append(forecast_season.astype(np.float32))
                map_error_fields.append((forecast_season - obs_season).astype(np.float32))
    if map_obs_fields:
        event_index = pd.Index(map_event_labels, dtype=object, name="event")
        map_dataset = xr.Dataset(
            data_vars={
                "obs_anomaly": xr.concat(map_obs_fields, dim=event_index, coords="minimal", compat="override"),
                "forecast_anomaly": xr.concat(map_forecast_fields, dim=event_index, coords="minimal", compat="override"),
                "forecast_error": xr.concat(map_error_fields, dim=event_index, coords="minimal", compat="override"),
            },
            coords={"peak_season": ("event", map_peak_seasons)},
            attrs={
                "title": "Tropical Pacific SST anomaly pattern matrix for ENSO source-region diagnostics",
                "lead_label": f"lead{config.map_lead:02d}",
                "anomaly_reference": f"{config.obs_label} monthly climatology {config.baseline_start} to {config.baseline_end}",
            },
        )
    else:
        empty_lat = obs_da["lat"]
        empty_lon = obs_da["lon"]
        empty_field = xr.DataArray(
            np.full((1, len(empty_lat), len(empty_lon)), np.nan, dtype=np.float32),
            coords={"event": ["unavailable"], "lat": empty_lat, "lon": empty_lon},
            dims=("event", "lat", "lon"),
        )
        map_dataset = xr.Dataset(
            data_vars={"obs_anomaly": empty_field, "forecast_anomaly": empty_field, "forecast_error": empty_field},
            attrs={"title": "Empty map dataset because the requested map lead was unavailable."},
        )
    return pd.DataFrame(flavor_rows), pd.DataFrame(pattern_rows), map_dataset


def write_metadata(config: CalculationConfig, output_paths: dict[str, Path]) -> Path:
    """Write metadata documenting the calculation settings and outputs."""

    metadata_path = config.output_dir / "Figure2_ENSO_source_region_metadata.json"
    metadata = {
        "figure_title": "Figure 3. Source-region prediction errors reveal an atypical forecast failure of the 2023-24 El Nino",
        "core_message": "The model failure in 2023-24 was not limited to remote MHW prediction, but already emerged in the ENSO source region.",
        "observation_label": config.obs_label,
        "observation_path": str(config.obs_path),
        "baseline": f"{config.baseline_start} to {config.baseline_end}",
        "leads": list(config.leads),
        "nino_regions": NINO_REGIONS,
        "event_windows": EVENT_WINDOWS,
        "event_peak_seasons": EVENT_PEAK_SEASONS,
        "map_event": config.map_event,
        "map_lead": config.map_lead,
        "tropical_pacific_bounds": {
            "lat_min": config.lat_min,
            "lat_max": config.lat_max,
            "lon_min": config.lon_min,
            "lon_max": config.lon_max,
        },
        "forecast_min_valid_sst": config.forecast_min_valid_sst,
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def run_calculation(config: CalculationConfig) -> dict[str, Path]:
    """Run the complete ENSO source-region diagnostic calculation."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    obs_da = load_observation(config)
    climatology = compute_monthly_climatology(obs_da, config)
    timeseries_df = calculate_timeseries(obs_da, climatology, config)
    lead_errors_df = calculate_lead_errors(timeseries_df, config)
    flavor_df, pattern_df, map_dataset = calculate_event_patterns(obs_da, climatology, config)
    output_paths = {
        "timeseries": config.output_dir / "Figure2_ENSO_source_region_timeseries.csv",
        "lead_errors": config.output_dir / "Figure2_ENSO_source_region_lead_errors.csv",
        "flavor": config.output_dir / "Figure2_ENSO_source_region_flavor.csv",
        "pattern_correlation": config.output_dir / "Figure2_ENSO_source_region_pattern_correlation.csv",
        "map_fields": config.output_dir / "Figure2_ENSO_source_region_pattern_fields.nc",
    }
    for step_label in tqdm(["time_series", "lead_errors", "flavor", "pattern_correlation", "map_fields"], desc="Step 4/5: write outputs", unit="file"):
        if step_label == "time_series":
            timeseries_df.to_csv(output_paths["timeseries"], index=False)
        elif step_label == "lead_errors":
            lead_errors_df.to_csv(output_paths["lead_errors"], index=False)
        elif step_label == "flavor":
            flavor_df.to_csv(output_paths["flavor"], index=False)
        elif step_label == "pattern_correlation":
            pattern_df.to_csv(output_paths["pattern_correlation"], index=False)
        elif step_label == "map_fields":
            map_dataset.to_netcdf(output_paths["map_fields"])
    metadata_path = write_metadata(config, output_paths)
    output_paths["metadata"] = metadata_path
    tqdm.write("Step 5/5: metadata written")
    return output_paths


def main() -> int:
    """Run the command-line calculation workflow."""

    config = build_config(parse_args())
    output_paths = run_calculation(config)
    for label, output_path in output_paths.items():
        print(f"Saved {label}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
