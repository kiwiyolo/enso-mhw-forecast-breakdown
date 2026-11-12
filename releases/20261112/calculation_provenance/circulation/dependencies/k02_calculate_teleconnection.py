#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: k02_calculate_teleconnection.py
# Date: 2026-06-08
# Copyright (c) 2026
#
"""
Calculate ENSO teleconnection diagnostics from forecasts and reanalyses.

Outputs include event-mean anomaly maps, observation-based canonical ENSO
templates, and region metrics for pattern correlation, amplitude, projection,
teleconnection efficiency, and forecast-observation spatial skill.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# Writable Matplotlib cache is set for consistency with the plotting pipeline.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Gravitational acceleration converts ERA5 geopotential to geopotential height.
GRAVITY = 9.80665

# Default variables used by Figure 4 diagnostics.
DEFAULT_FORECAST_PREFIXES = "z200,z250,z500,precipitation,olr,surface_stress_x,surface_stress_y"

# Candidate names for known NMME and ERA5 variables.
FIELD_VARIABLE_CANDIDATES = {
    "z200": ("h200", "hgt", "hght", "zg", "gz", "z", "geopotential_height", "height", "geopotential"),
    "z250": ("h250", "hgt", "hght", "zg", "gz", "z", "geopotential_height", "height", "geopotential"),
    "z500": ("h500", "hgt", "hght", "zg", "gz", "z", "geopotential_height", "height", "geopotential"),
    "precipitation": ("prec", "precip", "prate", "tp", "total_precipitation", "precipitation"),
    "evaporation": ("evap", "e", "evaporation"),
    "olr": ("olr", "ulwrf", "top_net_thermal_radiation", "ttr"),
    "surface_stress_x": ("taux", "eastward_turbulent_surface_stress", "ewss", "tauu"),
    "surface_stress_y": ("tauy", "northward_turbulent_surface_stress", "nsss", "tauv"),
}

# Human-readable labels for output metadata and figures.
FIELD_LABELS = {
    "z200": "Z200",
    "z250": "Z250",
    "z500": "Z500",
    "precipitation": "Precipitation",
    "evaporation": "Evaporation",
    "olr": "OLR",
    "surface_stress_x": "Eastward surface stress",
    "surface_stress_y": "Northward surface stress",
}

# Height-like fields need pressure selection and geopotential-unit conversion.
HEIGHT_FIELDS = {"z200", "z250", "z500"}

# Candidate names for sea-surface temperature variables used to compute NINO3.4.
SST_VARIABLE_CANDIDATES = ("sst", "tos", "SST")

# Region definitions used for ENSO teleconnection diagnostics.
TELECONNECTION_REGIONS = {
    "PNA_NorthAmerica": (20.0, 70.0, 120.0, 330.0),
    "PSA_SouthPacific": (-70.0, -20.0, 120.0, 300.0),
    "Atlantic_Bridge": (0.0, 60.0, 280.0, 360.0),
    "Global_60SN": (-60.0, 60.0, 0.0, 360.0),
}

# Default event windows include whole-event and phase-resolved diagnostics.
DEFAULT_EVENTS = (
    "1997/98:1997-01:1998-12,2015/16:2015-01:2016-12,2023/24:2023-01:2024-12,"
    "1997/98 development:1997-05:1997-10,1997/98 mature:1997-11:1998-02,1997/98 decay:1998-03:1998-08,"
    "2015/16 development:2015-05:2015-10,2015/16 mature:2015-11:2016-02,2015/16 decay:2016-03:2016-08,"
    "2023/24 development:2023-05:2023-10,2023/24 mature:2023-11:2024-02,2023/24 decay:2024-03:2024-08"
)


@dataclass(frozen=True)
class EventWindow:
    """One event window used for teleconnection compositing."""

    label: str
    start_month: str
    end_month: str


@dataclass(frozen=True)
class CalculationConfig:
    """Runtime configuration for teleconnection calculations."""

    work_dir: Path
    teleconnection_data_dir: Path
    output_dir: Path
    skip_log: Path
    nino34_csv: Path | None
    nino34_sst_path: Path
    forecast_prefixes: tuple[str, ...]
    target_pressure: float
    models: tuple[str, ...]
    obs_sources: tuple[str, ...]
    leads: tuple[int, ...]
    events: tuple[EventWindow, ...]
    template_events: tuple[str, ...]
    baseline_start: str
    baseline_end: str
    lat_min: float
    lat_max: float
    workers: int


def setup_logger() -> logging.Logger:
    """Create a compact logger for calculation progress."""

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("k02_calculate_teleconnection")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    work_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Calculate ENSO teleconnection maps and metrics.")
    parser.add_argument("--work-dir", default=str(work_dir), help="Working directory containing this script.")
    parser.add_argument("--teleconnection-data-dir", "--data-dir", default=str(work_dir / "teleconnection_data"), help="Downloaded teleconnection data root.")
    parser.add_argument("--output-dir", default=str(work_dir / "teleconnection_data_products"), help="Output directory.")
    parser.add_argument("--skip-log", default="", help="CSV log for skipped calculation inputs.")
    parser.add_argument("--models", default="COLA-RSMAS-CCSM4,NASA-GEOSS2S,GFDL-SPEAR,NCEP-CFSv2,NMME", help="Comma-separated model labels.")
    parser.add_argument("--forecast-prefix", default=DEFAULT_FORECAST_PREFIXES, help="Comma-separated forecast file prefixes, such as z200,z250,z500,precipitation,olr,surface_stress_x,surface_stress_y.")
    parser.add_argument("--target-pressure", type=float, default=200.0, help="Pressure level selected from multi-level height files.")
    parser.add_argument("--obs-sources", default="ERA5", help="Comma-separated reanalysis labels; ERA5 is the unified default.")
    parser.add_argument("--leads", default="1,3,6,9", help="Comma-separated lead numbers; lead1 means L=0.5.")
    parser.add_argument(
        "--events",
        default=DEFAULT_EVENTS,
        help="Comma-separated label:start-month:end-month event windows.",
    )
    parser.add_argument("--template-events", default="1997/98,2015/16", help="Events used to define the canonical strong-El-Nino template.")
    parser.add_argument("--baseline", default="1991-2020", help="Monthly climatology baseline years.")
    parser.add_argument("--lat-min", type=float, default=-60.0, help="Minimum latitude retained.")
    parser.add_argument("--lat-max", type=float, default=60.0, help="Maximum latitude retained.")
    parser.add_argument("--nino34-csv", default="", help="Optional CSV with month/date and NINO3.4 columns.")
    parser.add_argument("--nino34-sst-path", default=str(work_dir.parent / "data" / "ersst_observation.nc"), help="SST NetCDF used when NINO3.4 CSV is not provided.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TELECONNECTION_CALC_WORKERS", os.environ.get("Z200_CALC_WORKERS", "1"))), help="Maximum parallel workers for model-lead forecast calculations.")
    return parser.parse_args()


def parse_csv_labels(text: str) -> tuple[str, ...]:
    """Parse comma-separated labels into an ordered tuple."""

    labels = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def parse_forecast_prefixes(text: str) -> tuple[str, ...]:
    """Parse requested forecast field prefixes."""

    clean_text = str(text).strip()
    if clean_text in {"all", "all_requested", "figure4"}:
        clean_text = DEFAULT_FORECAST_PREFIXES
    prefixes = parse_csv_labels(clean_text)
    unsupported = [prefix for prefix in prefixes if prefix not in FIELD_VARIABLE_CANDIDATES]
    if unsupported:
        raise ValueError(f"Unsupported forecast prefixes: {unsupported}")
    return tuple(dict.fromkeys(prefixes))


def parse_leads(text: str) -> tuple[int, ...]:
    """Parse comma-separated lead numbers."""

    lead_values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not lead_values:
        raise ValueError("At least one lead is required.")
    if any(lead_value < 1 for lead_value in lead_values):
        raise ValueError("Lead numbers must be >= 1.")
    return lead_values


def parse_events(text: str) -> tuple[EventWindow, ...]:
    """Parse event windows from label:start:end text."""

    events: list[EventWindow] = []
    for item in str(text).split(","):
        clean_item = item.strip()
        if not clean_item:
            continue
        parts = clean_item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Event must use label:start-month:end-month format: {clean_item}")
        label, start_month, end_month = parts
        start_period = pd.Period(start_month, freq="M")
        end_period = pd.Period(end_month, freq="M")
        if end_period < start_period:
            raise ValueError(f"Event end precedes start: {clean_item}")
        events.append(EventWindow(label, str(start_period), str(end_period)))
    if not events:
        raise ValueError("At least one event is required.")
    return tuple(events)


def build_config(args: argparse.Namespace) -> CalculationConfig:
    """Build a validated CalculationConfig object."""

    baseline_start, baseline_end = str(args.baseline).split("-", 1)
    return CalculationConfig(
        work_dir=Path(args.work_dir).resolve(),
        teleconnection_data_dir=Path(args.teleconnection_data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        skip_log=Path(args.skip_log).resolve() if str(args.skip_log).strip() else Path(args.output_dir).resolve() / "teleconnection_calculation_skipped.csv",
        nino34_csv=Path(args.nino34_csv).resolve() if str(args.nino34_csv).strip() else None,
        nino34_sst_path=Path(args.nino34_sst_path).resolve(),
        forecast_prefixes=parse_forecast_prefixes(args.forecast_prefix),
        target_pressure=float(args.target_pressure),
        models=parse_csv_labels(args.models),
        obs_sources=parse_csv_labels(args.obs_sources),
        leads=parse_leads(args.leads),
        events=parse_events(args.events),
        template_events=parse_csv_labels(args.template_events),
        baseline_start=f"{int(baseline_start):04d}-01",
        baseline_end=f"{int(baseline_end):04d}-12",
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        workers=max(1, int(args.workers)),
    )


def safe_label(label: str) -> str:
    """Return a file-safe label."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))


def append_skip_record(skip_log: Path, stage: str, item: str, reason: str) -> None:
    """Append one skipped input record to a CSV log."""

    skip_log.parent.mkdir(parents=True, exist_ok=True)
    write_header = not skip_log.exists()
    with skip_log.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=("stage", "item", "reason"))
        if write_header:
            writer.writeheader()
        writer.writerow({"stage": stage, "item": item, "reason": reason})


def month_to_timestamp(month_text: str) -> pd.Timestamp:
    """Convert a YYYY-MM month string to a timestamp at month start."""

    return pd.Period(str(month_text), freq="M").to_timestamp()


def event_months(event: EventWindow) -> pd.DatetimeIndex:
    """Return the monthly timestamps inside one event window."""

    return pd.period_range(event.start_month, event.end_month, freq="M").to_timestamp()


def choose_data_variable(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """Choose a variable from a dataset using preferred candidate names."""

    for variable_name in candidates:
        if variable_name in dataset.data_vars:
            return variable_name
    if not dataset.data_vars:
        raise ValueError("Dataset contains no data variables.")
    return next(iter(dataset.data_vars))


def find_coord_name(data_array: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching dimension or coordinate name."""

    for candidate in candidates:
        if candidate in data_array.dims or candidate in data_array.coords:
            return candidate
    return None


def normalize_longitudes(data_array: xr.DataArray) -> xr.DataArray:
    """Convert longitude coordinates to 0-360 and sort them."""

    lon_name = find_coord_name(data_array, ("lon", "longitude", "X"))
    if lon_name is None:
        return data_array
    normalized_lon = np.mod(np.asarray(data_array[lon_name].values, dtype=float), 360.0)
    data_array = data_array.assign_coords({lon_name: normalized_lon})
    return data_array.sortby(lon_name)


def standardize_spatial_names(data_array: xr.DataArray) -> xr.DataArray:
    """Rename common spatial dimension names to lat and lon."""

    rename_map: dict[str, str] = {}
    lat_name = find_coord_name(data_array, ("lat", "latitude", "Y"))
    lon_name = find_coord_name(data_array, ("lon", "longitude", "X"))
    if lat_name is not None and lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name is not None and lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        data_array = data_array.rename(rename_map)
    data_array = normalize_longitudes(data_array)
    if "lat" in data_array.coords:
        data_array = data_array.sortby("lat")
    return data_array


def select_pressure_level(data_array: xr.DataArray, target_pressure: float) -> xr.DataArray:
    """Select a target pressure level when a pressure dimension exists."""

    pressure_name = find_coord_name(data_array, ("level", "pressure_level", "plev", "P", "P1", "P2", "P3", "isobaricInhPa"))
    if pressure_name is None:
        return data_array
    return data_array.sel({pressure_name: float(target_pressure)}, method="nearest")


def standardize_height_units(data_array: xr.DataArray, variable_name: str) -> xr.DataArray:
    """Convert geopotential to geopotential height in meters when needed."""

    units_text = str(data_array.attrs.get("units", "")).lower()
    if variable_name == "z" or "m**2" in units_text or "m2 s-2" in units_text or "m^2" in units_text:
        converted = data_array / GRAVITY
        converted.attrs["units"] = "m"
        return converted
    return data_array


def choose_field_variable(dataset: xr.Dataset, field_prefix: str) -> str:
    """Choose the NetCDF variable matching one configured field prefix."""

    candidates = FIELD_VARIABLE_CANDIDATES[field_prefix]
    for variable_name in candidates:
        if variable_name in dataset.data_vars:
            return variable_name
    for variable_name, data_array in dataset.data_vars.items():
        metadata_text = " ".join(
            str(data_array.attrs.get(attr_name, ""))
            for attr_name in ("standard_name", "long_name", "description", "units")
        ).lower()
        if field_prefix in HEIGHT_FIELDS and "geopotential" in metadata_text and ("height" in metadata_text or variable_name == "z"):
            return variable_name
        if field_prefix == "precipitation" and ("precipitation" in metadata_text or "precip" in metadata_text):
            return variable_name
        if field_prefix == "evaporation" and "evaporation" in metadata_text:
            return variable_name
        if field_prefix == "olr" and ("longwave" in metadata_text or "thermal radiation" in metadata_text):
            return variable_name
        if field_prefix == "surface_stress_x" and ("eastward" in metadata_text and "stress" in metadata_text):
            return variable_name
        if field_prefix == "surface_stress_y" and ("northward" in metadata_text and "stress" in metadata_text):
            return variable_name
    raise ValueError(f"No variable found for {field_prefix}. Available variables: {list(dataset.data_vars)}")


def preprocess_field_dataarray(dataset: xr.Dataset, field_prefix: str, target_pressure: float) -> xr.DataArray:
    """Return a standardized data array for one field prefix."""

    variable_name = choose_field_variable(dataset, field_prefix)
    data_array = dataset[variable_name]
    if field_prefix in HEIGHT_FIELDS:
        pressure_value = target_pressure
        if field_prefix == "z250":
            pressure_value = 250.0
        elif field_prefix == "z500":
            pressure_value = 500.0
        data_array = select_pressure_level(data_array, pressure_value)
        data_array = standardize_height_units(data_array, variable_name)
    data_array = standardize_spatial_names(data_array)
    return data_array


def drop_auxiliary_coordinates(data_array: xr.DataArray) -> xr.DataArray:
    """Drop scalar auxiliary coordinates that are not needed after selection."""

    # Selected pressure, lead, and start coordinates can differ across archives.
    return data_array.reset_coords(drop=True)


def squeeze_singleton_nonspatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    """Remove singleton nonspatial dimensions left by model-specific fields."""

    keep_dims = {"member", "target_time", "lat", "lon"}
    squeeze_dims = [dim_name for dim_name in data_array.dims if dim_name not in keep_dims and data_array.sizes.get(dim_name, 0) == 1]
    if squeeze_dims:
        data_array = data_array.isel({dim_name: 0 for dim_name in squeeze_dims})
    return data_array


def subset_latitude(data_array: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    """Subset a data array to the configured latitude band."""

    if "lat" not in data_array.coords:
        return data_array
    return data_array.sel(lat=slice(float(lat_min), float(lat_max)))


def normalize_member_dimension(data_array: xr.DataArray) -> xr.DataArray:
    """Rename or create a member dimension for forecast arrays."""

    member_name = find_coord_name(data_array, ("member", "M", "realization"))
    if member_name is None:
        return data_array.expand_dims(member=[1])
    if member_name != "member":
        data_array = data_array.rename({member_name: "member"})
    return data_array


def select_lead(data_array: xr.DataArray, lead_number: int) -> xr.DataArray:
    """Select a NMME lead where lead1 corresponds to L=0.5 months."""

    lead_name = find_coord_name(data_array, ("L", "lead", "forecast_period"))
    if lead_name is None:
        return data_array
    target_lead_value = float(int(lead_number) - 0.5)
    lead_values = np.asarray(data_array[lead_name].values, dtype=float)
    nearest_index = int(np.nanargmin(np.abs(lead_values - target_lead_value)))
    return data_array.isel({lead_name: nearest_index})


def average_start_dimension(data_array: xr.DataArray) -> xr.DataArray:
    """Collapse the NMME start-time dimension after one file is selected."""

    start_name = find_coord_name(data_array, ("S", "forecast_reference_time"))
    if start_name is not None and start_name in data_array.dims:
        return data_array.mean(start_name, skipna=True)
    return data_array


def lead_target_to_init(target_month: pd.Timestamp, lead_number: int) -> pd.Period:
    """Convert target month and lead number to the required initialization month."""

    target_period = pd.Period(target_month, freq="M")
    return target_period - (int(lead_number) - 1)


def load_forecast_field(
    file_path: Path,
    field_prefix: str,
    lead_number: int,
    lat_min: float,
    lat_max: float,
    target_pressure: float,
    expected_init: pd.Period | None = None,
) -> xr.DataArray:
    """Load one NMME forecast file for one selected lead."""

    with xr.open_dataset(file_path, decode_times=False) as dataset:
        if expected_init is not None and "S" in dataset.coords:
            start = np.asarray(dataset["S"].values, dtype=float).reshape(-1)
            expected = (expected_init.year - 1960) * 12 + (expected_init.month - 1)
            if start.size == 0 or not np.isfinite(start[0]) or abs(float(start[0]) - expected) > 0.1:
                actual = float(start[0]) if start.size else float("nan")
                raise ValueError(
                    f"internal start S={actual:g} does not match {expected_init} (expected S={expected})"
                )
        data_array = preprocess_field_dataarray(dataset, field_prefix, target_pressure)
        data_array = select_lead(data_array, lead_number)
        data_array = average_start_dimension(data_array)
        data_array = normalize_member_dimension(data_array)
        data_array = subset_latitude(data_array, lat_min, lat_max)
        data_array = squeeze_singleton_nonspatial_dims(data_array)
        data_array = data_array.transpose("member", "lat", "lon").load()
        data_array = drop_auxiliary_coordinates(data_array)
    return data_array


def model_file_path(teleconnection_data_dir: Path, model_name: str, init_period: pd.Period, forecast_prefix: str) -> Path:
    """Return the local forecast file path for one model initialization."""

    return teleconnection_data_dir / "NMME" / model_name / f"{forecast_prefix}_{init_period.year}{init_period.month:02d}.nc"


def load_forecast_series(
    teleconnection_data_dir: Path,
    field_prefix: str,
    model_name: str,
    lead_number: int,
    target_times: pd.DatetimeIndex,
    lat_min: float,
    lat_max: float,
    forecast_prefix: str,
    target_pressure: float,
) -> xr.DataArray | None:
    """Load forecast fields aligned to requested target months."""

    forecast_fields: list[xr.DataArray] = []
    forecast_times: list[pd.Timestamp] = []
    source_models = [model_name]
    if model_name == "NMME":
        source_models = ["COLA-RSMAS-CCSM4", "NASA-GEOSS2S", "GFDL-SPEAR", "NCEP-CFSv2"]
    for target_time in target_times:
        member_fields: list[xr.DataArray] = []
        init_period = lead_target_to_init(target_time, lead_number)
        for source_model in source_models:
            file_path = model_file_path(teleconnection_data_dir, source_model, init_period, forecast_prefix)
            if not file_path.exists():
                continue
            try:
                member_fields.append(
                    load_forecast_field(
                        file_path,
                        field_prefix,
                        lead_number,
                        lat_min,
                        lat_max,
                        target_pressure,
                        expected_init=init_period,
                    )
                )
            except Exception as exc:
                warnings.warn(f"Skipping forecast file {file_path}: {exc}")
        if not member_fields:
            continue
        combined_members = xr.concat(member_fields, dim="member", coords="minimal", compat="override", join="outer")
        combined_members = combined_members.assign_coords(member=np.arange(1, combined_members.sizes["member"] + 1))
        forecast_fields.append(combined_members.expand_dims(target_time=[np.datetime64(target_time)]))
        forecast_times.append(target_time)
    if not forecast_fields:
        return None
    return xr.concat(forecast_fields, dim="target_time", coords="minimal", compat="override", join="outer")


def observation_files(teleconnection_data_dir: Path, source_name: str, field_prefix: str) -> list[Path]:
    """Return local reanalysis files for one observation source."""

    source_dir = teleconnection_data_dir / "observations" / source_name
    if not source_dir.exists():
        return []
    if source_name == "ERA5":
        if field_prefix in HEIGHT_FIELDS:
            era5_pressure_files = sorted(source_dir.glob("era5_pressure_levels_*.nc"))
            if era5_pressure_files:
                return era5_pressure_files
        era5_single_files = sorted(source_dir.glob("era5_single_levels_*.nc"))
        if era5_single_files:
            return era5_single_files
    patterns = ("*.nc", "*.grib", "*.grib2")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(source_dir.glob(pattern)))
    return files


def load_observation_series(
    teleconnection_data_dir: Path,
    field_prefix: str,
    source_name: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    lat_min: float,
    lat_max: float,
    target_pressure: float,
) -> xr.DataArray | None:
    """Load reanalysis fields for a source and time span."""

    fields: list[xr.DataArray] = []
    for file_path in observation_files(teleconnection_data_dir, source_name, field_prefix):
        try:
            with xr.open_dataset(file_path, decode_times=True) as dataset:
                data_array = preprocess_field_dataarray(dataset, field_prefix, target_pressure)
                time_name = find_coord_name(data_array, ("time", "valid_time", "target_time"))
                if time_name is None:
                    continue
                if time_name != "target_time":
                    data_array = data_array.rename({time_name: "target_time"})
                data_array = subset_latitude(data_array, lat_min, lat_max)
                data_array = data_array.sel(target_time=slice(start_time, end_time))
                if data_array.sizes.get("target_time", 0) == 0:
                    continue
                data_array = data_array.transpose("target_time", "lat", "lon").load()
                fields.append(drop_auxiliary_coordinates(data_array))
        except Exception as exc:
            warnings.warn(f"Skipping observation file {file_path}: {exc}")
    if not fields:
        return None
    combined = xr.concat(fields, dim="target_time", coords="minimal", compat="override", join="outer").sortby("target_time")
    _, unique_index = np.unique(pd.to_datetime(combined["target_time"].values), return_index=True)
    return combined.isel(target_time=np.sort(unique_index))


def monthly_anomalies(series: xr.DataArray, baseline_start: str, baseline_end: str) -> xr.DataArray:
    """Calculate monthly anomalies relative to a monthly baseline climatology."""

    baseline = series.sel(target_time=slice(month_to_timestamp(baseline_start), month_to_timestamp(baseline_end)))
    if baseline.sizes.get("target_time", 0) == 0:
        raise ValueError("No baseline months are available for anomaly calculation.")
    climatology = baseline.groupby("target_time.month").mean("target_time", skipna=True)
    return series.groupby("target_time.month") - climatology


def composite_events(anomalies: xr.DataArray, events: tuple[EventWindow, ...]) -> xr.DataArray:
    """Average anomaly fields over each configured event window."""

    event_arrays: list[xr.DataArray] = []
    event_labels: list[str] = []
    for event in events:
        event_array = anomalies.sel(target_time=slice(month_to_timestamp(event.start_month), month_to_timestamp(event.end_month)))
        if event_array.sizes.get("target_time", 0) == 0:
            continue
        event_mean = event_array.mean("target_time", skipna=True)
        if "member" in event_mean.dims:
            event_mean = event_mean.mean("member", skipna=True)
        event_arrays.append(event_mean.expand_dims(event=[event.label]))
        event_labels.append(event.label)
    if not event_arrays:
        raise ValueError("No event composites could be calculated.")
    return xr.concat(event_arrays, dim="event").assign_coords(event=event_labels)


def template_from_events(event_maps: xr.DataArray, template_events: tuple[str, ...]) -> xr.DataArray:
    """Build a canonical ENSO teleconnection template from selected event composites."""

    available_events = [event for event in template_events if event in set(event_maps["event"].values.astype(str))]
    if not available_events:
        raise ValueError("No requested template events are available in event maps.")
    return event_maps.sel(event=available_events).mean("event", skipna=True)


def select_region(data_array: xr.DataArray, region: tuple[float, float, float, float]) -> xr.DataArray:
    """Subset a data array to a lat-lon region using 0-360 longitude."""

    lat_min, lat_max, lon_min, lon_max = region
    data_array = normalize_longitudes(data_array)
    lat_subset = data_array.sel(lat=slice(lat_min, lat_max))
    if lon_min <= lon_max:
        return lat_subset.sel(lon=slice(lon_min, lon_max))
    left = lat_subset.sel(lon=slice(lon_min, 360.0))
    right = lat_subset.sel(lon=slice(0.0, lon_max))
    return xr.concat([left, right], dim="lon")


def area_weights(data_array: xr.DataArray) -> xr.DataArray:
    """Return cosine-latitude weights broadcastable to a map."""

    weights = np.cos(np.deg2rad(data_array["lat"]))
    return xr.DataArray(weights, dims=("lat",), coords={"lat": data_array["lat"]})


def weighted_pattern_correlation(field: xr.DataArray, template: xr.DataArray) -> float:
    """Calculate weighted spatial pattern correlation."""

    template_interp = template.interp(lat=field["lat"], lon=field["lon"], method="linear")
    valid_mask = np.isfinite(field) & np.isfinite(template_interp)
    if int(valid_mask.sum()) < 3:
        return float("nan")
    weights = area_weights(field).broadcast_like(field).where(valid_mask)
    field_values = field.where(valid_mask)
    template_values = template_interp.where(valid_mask)
    weight_sum = float(weights.sum(skipna=True))
    if weight_sum <= 0:
        return float("nan")
    field_mean = float((field_values * weights).sum(skipna=True) / weight_sum)
    template_mean = float((template_values * weights).sum(skipna=True) / weight_sum)
    field_anom = field_values - field_mean
    template_anom = template_values - template_mean
    numerator = float((field_anom * template_anom * weights).sum(skipna=True))
    denominator = float(np.sqrt((field_anom**2 * weights).sum(skipna=True) * (template_anom**2 * weights).sum(skipna=True)))
    return numerator / denominator if denominator > 0 else float("nan")


def weighted_rms_amplitude(field: xr.DataArray) -> float:
    """Calculate weighted root-mean-square anomaly amplitude."""

    valid_mask = np.isfinite(field)
    if int(valid_mask.sum()) < 3:
        return float("nan")
    weights = area_weights(field).broadcast_like(field).where(valid_mask)
    weight_sum = float(weights.sum(skipna=True))
    if weight_sum <= 0:
        return float("nan")
    return float(np.sqrt(((field.where(valid_mask) ** 2) * weights).sum(skipna=True) / weight_sum))


def weighted_projection(field: xr.DataArray, template: xr.DataArray) -> float:
    """Project a field onto the template pattern."""

    template_interp = template.interp(lat=field["lat"], lon=field["lon"], method="linear")
    valid_mask = np.isfinite(field) & np.isfinite(template_interp)
    if int(valid_mask.sum()) < 3:
        return float("nan")
    weights = area_weights(field).broadcast_like(field).where(valid_mask)
    numerator = float((field.where(valid_mask) * template_interp.where(valid_mask) * weights).sum(skipna=True))
    denominator = float(((template_interp.where(valid_mask) ** 2) * weights).sum(skipna=True))
    return numerator / denominator if denominator > 0 else float("nan")


def load_nino34_from_csv(csv_path: Path) -> dict[str, float]:
    """Load NINO3.4 values from a user-provided CSV file."""

    table = pd.read_csv(csv_path)
    month_column = next((column for column in table.columns if column.lower() in {"month", "date", "target_month", "time"}), None)
    value_column = next((column for column in table.columns if "nino" in column.lower()), None)
    if month_column is None or value_column is None:
        raise ValueError(f"Could not find month/date and NINO columns in {csv_path}")
    month_values = pd.to_datetime(table[month_column]).dt.to_period("M").astype(str)
    return dict(zip(month_values, pd.to_numeric(table[value_column], errors="coerce"), strict=False))


def area_mean_sst(data_array: xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    """Calculate weighted area-mean SST over one lon-lat box."""

    data_array = standardize_spatial_names(data_array)
    data_array = normalize_longitudes(data_array)
    subset = data_array.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    weights = np.cos(np.deg2rad(subset["lat"]))
    return subset.weighted(weights).mean(("lat", "lon"), skipna=True)


def load_nino34_from_sst(sst_path: Path, baseline_start: str, baseline_end: str) -> dict[str, float]:
    """Compute NINO3.4 SST anomalies from a local SST observation file."""

    if not sst_path.exists():
        return {}
    with xr.open_dataset(sst_path, decode_times=True) as dataset:
        variable_name = choose_data_variable(dataset, SST_VARIABLE_CANDIDATES)
        sst = dataset[variable_name]
        time_name = find_coord_name(sst, ("time", "target_time"))
        if time_name is None:
            return {}
        if time_name != "target_time":
            sst = sst.rename({time_name: "target_time"})
        nino34 = area_mean_sst(sst, -5.0, 5.0, 190.0, 240.0)
        baseline = nino34.sel(target_time=slice(month_to_timestamp(baseline_start), month_to_timestamp(baseline_end)))
        climatology = baseline.groupby("target_time.month").mean("target_time", skipna=True)
        anomaly = nino34.groupby("target_time.month") - climatology
        months = pd.to_datetime(anomaly["target_time"].values).to_period("M").astype(str)
        return dict(zip(months, anomaly.values.astype(float), strict=False))


def load_nino34(config: CalculationConfig, logger: logging.Logger) -> dict[str, float]:
    """Load or calculate monthly NINO3.4 anomaly values."""

    if config.nino34_csv is not None and config.nino34_csv.exists():
        logger.info("Loading NINO3.4 from CSV: %s", config.nino34_csv)
        return load_nino34_from_csv(config.nino34_csv)
    logger.info("Calculating NINO3.4 from SST file: %s", config.nino34_sst_path)
    return load_nino34_from_sst(config.nino34_sst_path, config.baseline_start, config.baseline_end)


def nino34_event_mean(nino34_by_month: dict[str, float], event: EventWindow) -> float:
    """Return the mean NINO3.4 anomaly inside one event window."""

    months = pd.period_range(event.start_month, event.end_month, freq="M").astype(str)
    values = np.asarray([nino34_by_month.get(month, np.nan) for month in months], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def metric_rows_for_maps(
    field_prefix: str,
    event_maps: xr.DataArray,
    template_map: xr.DataArray,
    nino34_by_month: dict[str, float],
    events: tuple[EventWindow, ...],
    source_type: str,
    source_name: str,
    model_name: str,
    lead_number: int | None,
    obs_event_maps: xr.DataArray | None,
) -> list[dict[str, object]]:
    """Calculate all region metrics for one set of event maps."""

    rows: list[dict[str, object]] = []
    event_by_label = {event.label: event for event in events}
    for event_label in event_maps["event"].values.astype(str):
        if event_label not in event_by_label:
            continue
        event = event_by_label[event_label]
        field = event_maps.sel(event=event_label)
        obs_field = obs_event_maps.sel(event=event_label) if obs_event_maps is not None and event_label in obs_event_maps["event"].values else None
        for region_name, region_box in TELECONNECTION_REGIONS.items():
            field_region = select_region(field, region_box)
            template_region = select_region(template_map, region_box)
            projection_value = weighted_projection(field_region, template_region)
            nino_value = nino34_event_mean(nino34_by_month, event)
            forecast_skill = weighted_pattern_correlation(field_region, select_region(obs_field, region_box)) if obs_field is not None else float("nan")
            rows.append(
                {
                    "source_type": source_type,
                    "field": field_prefix,
                    "field_label": FIELD_LABELS.get(field_prefix, field_prefix),
                    "source": source_name,
                    "model": model_name,
                    "lead": lead_number if lead_number is not None else "",
                    "event": event_label,
                    "region": region_name,
                    "nino34": nino_value,
                    "pattern_correlation_to_template": weighted_pattern_correlation(field_region, template_region),
                    "amplitude_rms_m": weighted_rms_amplitude(field_region),
                    "projection_coefficient": projection_value,
                    "teleconnection_efficiency": projection_value / nino_value if np.isfinite(nino_value) and abs(nino_value) > 1.0e-12 else float("nan"),
                    "forecast_observation_pattern_skill": forecast_skill,
                }
            )
    return rows


def save_event_maps(field_prefix: str, event_maps: xr.DataArray, output_path: Path, attrs: dict[str, object]) -> None:
    """Save event anomaly maps to NetCDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = event_maps.to_dataset(name=f"{field_prefix}_anomaly")
    dataset.attrs.update({key: str(value) for key, value in attrs.items()})
    dataset.attrs["field"] = field_prefix
    dataset.attrs["field_label"] = FIELD_LABELS.get(field_prefix, field_prefix)
    dataset.to_netcdf(output_path)


def all_needed_times(config: CalculationConfig) -> pd.DatetimeIndex:
    """Return the union of baseline and event months needed for calculations."""

    baseline_times = pd.period_range(config.baseline_start, config.baseline_end, freq="M").to_timestamp()
    event_times = [event_months(event) for event in config.events]
    return pd.DatetimeIndex(sorted(set(baseline_times).union(*[set(times) for times in event_times])))


def calculate_observation_products(
    config: CalculationConfig,
    field_prefix: str,
    nino34_by_month: dict[str, float],
    logger: logging.Logger,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray], list[dict[str, object]]]:
    """Calculate observation event maps, templates, and metrics."""

    obs_maps: dict[str, xr.DataArray] = {}
    obs_templates: dict[str, xr.DataArray] = {}
    metric_rows: list[dict[str, object]] = []
    needed_times = all_needed_times(config)
    for obs_source in tqdm(config.obs_sources, desc="Observation calculations"):
        series = load_observation_series(
            config.teleconnection_data_dir,
            field_prefix,
            obs_source,
            needed_times.min(),
            needed_times.max(),
            config.lat_min,
            config.lat_max,
            config.target_pressure,
        )
        if series is None:
            logger.warning("No observation %s data found for %s.", field_prefix, obs_source)
            append_skip_record(config.skip_log, "observation", f"{obs_source} {field_prefix}", "no readable observation files found")
            continue
        anomalies = monthly_anomalies(series, config.baseline_start, config.baseline_end)
        event_maps = composite_events(anomalies, config.events)
        template_map = template_from_events(event_maps, config.template_events)
        obs_maps[obs_source] = event_maps
        obs_templates[obs_source] = template_map
        save_event_maps(
            field_prefix,
            event_maps,
            config.output_dir / "maps" / f"{field_prefix}_anomaly_OBS_{safe_label(obs_source)}.nc",
            {"source_type": "OBS", "source": obs_source, "field": field_prefix},
        )
        save_event_maps(
            field_prefix,
            template_map.expand_dims(event=["ENSO_template"]),
            config.output_dir / "templates" / f"{field_prefix}_enso_template_{safe_label(obs_source)}.nc",
            {"source_type": "OBS", "source": obs_source, "field": field_prefix, "template_events": ",".join(config.template_events)},
        )
        metric_rows.extend(
            metric_rows_for_maps(field_prefix, event_maps, template_map, nino34_by_month, config.events, "OBS", obs_source, obs_source, None, None)
        )
    return obs_maps, obs_templates, metric_rows


def calculate_forecast_products(
    config: CalculationConfig,
    field_prefix: str,
    obs_maps: dict[str, xr.DataArray],
    obs_templates: dict[str, xr.DataArray],
    nino34_by_month: dict[str, float],
    logger: logging.Logger,
) -> list[dict[str, object]]:
    """Calculate forecast event maps and metrics for all model-lead combinations."""

    metric_rows: list[dict[str, object]] = []
    needed_times = all_needed_times(config)
    jobs = [(model_name, lead_number) for model_name in config.models for lead_number in config.leads]
    max_workers = max(1, min(int(config.workers), len(jobs)))
    logger.info("Calculating %d forecast model-lead products with %d workers.", len(jobs), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                calculate_one_forecast_product,
                config,
                field_prefix,
                model_name,
                lead_number,
                needed_times,
                obs_maps,
                obs_templates,
                nino34_by_month,
            ): (model_name, lead_number)
            for model_name, lead_number in jobs
        }
        with tqdm(total=len(jobs), desc="Forecast calculations") as progress_bar:
            for future in as_completed(future_to_job):
                model_name, lead_number = future_to_job[future]
                progress_bar.set_postfix_str(f"{model_name} lead{lead_number:02d}")
                try:
                    job_rows, skip_record = future.result()
                except Exception as exc:
                    logger.warning("Failed forecast calculation for %s %s lead%02d: %s", field_prefix, model_name, lead_number, exc)
                    append_skip_record(config.skip_log, "forecast", f"{field_prefix} {model_name} lead{lead_number:02d}", str(exc))
                    progress_bar.update(1)
                    continue
                if skip_record is not None:
                    logger.warning("No forecast %s data found for %s lead%02d.", field_prefix, model_name, lead_number)
                    append_skip_record(config.skip_log, *skip_record)
                metric_rows.extend(job_rows)
                progress_bar.update(1)
    return metric_rows


def calculate_one_forecast_product(
    config: CalculationConfig,
    field_prefix: str,
    model_name: str,
    lead_number: int,
    needed_times: pd.DatetimeIndex,
    obs_maps: dict[str, xr.DataArray],
    obs_templates: dict[str, xr.DataArray],
    nino34_by_month: dict[str, float],
) -> tuple[list[dict[str, object]], tuple[str, str, str] | None]:
    """Calculate one model-lead forecast product for threaded execution."""

    series = load_forecast_series(
        config.teleconnection_data_dir,
        field_prefix,
        model_name,
        lead_number,
        needed_times,
        config.lat_min,
        config.lat_max,
        field_prefix,
        config.target_pressure,
    )
    if series is None:
        return [], ("forecast", f"{field_prefix} {model_name} lead{lead_number:02d}", "no readable forecast files found")
    anomalies = monthly_anomalies(series, config.baseline_start, config.baseline_end)
    event_maps = composite_events(anomalies, config.events)
    save_event_maps(
        field_prefix,
        event_maps,
        config.output_dir / "maps" / f"{field_prefix}_anomaly_NMME_{safe_label(model_name)}_lead{lead_number:02d}.nc",
        {"source_type": "NMME", "source": model_name, "model": model_name, "lead": lead_number, "field": field_prefix},
    )
    metric_rows: list[dict[str, object]] = []
    for obs_source, template_map in obs_templates.items():
        obs_event_maps = obs_maps.get(obs_source)
        metric_rows.extend(
            metric_rows_for_maps(
                field_prefix,
                event_maps,
                template_map,
                nino34_by_month,
                config.events,
                "NMME",
                obs_source,
                model_name,
                lead_number,
                obs_event_maps,
            )
        )
    return metric_rows, None


def write_metadata(config: CalculationConfig, output_path: Path) -> None:
    """Write a JSON file documenting the calculation run."""

    metadata = {
        "teleconnection_data_dir": str(config.teleconnection_data_dir),
        "forecast_prefixes": config.forecast_prefixes,
        "target_pressure": config.target_pressure,
        "skip_log": str(config.skip_log),
        "models": config.models,
        "obs_sources": config.obs_sources,
        "leads": config.leads,
        "events": [event.__dict__ for event in config.events],
        "template_events": config.template_events,
        "baseline_start": config.baseline_start,
        "baseline_end": config.baseline_end,
        "lat_min": config.lat_min,
        "lat_max": config.lat_max,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    """Run the full teleconnection calculation workflow."""

    logger = setup_logger()
    config = build_config(parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.skip_log.unlink(missing_ok=True)
    nino34_by_month = load_nino34(config, logger)
    all_metric_rows: list[dict[str, object]] = []
    for field_prefix in tqdm(config.forecast_prefixes, desc="Field products"):
        logger.info("Calculating field products for %s.", field_prefix)
        obs_maps, obs_templates, obs_metric_rows = calculate_observation_products(config, field_prefix, nino34_by_month, logger)
        forecast_metric_rows = calculate_forecast_products(config, field_prefix, obs_maps, obs_templates, nino34_by_month, logger)
        all_metric_rows.extend(obs_metric_rows + forecast_metric_rows)
    metric_table = pd.DataFrame(all_metric_rows)
    metric_path = config.output_dir / "teleconnection_metrics.csv"
    metric_table.to_csv(metric_path, index=False)
    write_metadata(config, config.output_dir / "teleconnection_metadata.json")
    logger.info("Saved metrics: %s", metric_path)
    logger.info("Saved maps and templates under: %s", config.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
