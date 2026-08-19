#!/usr/bin/env python
"""Run lightweight integrity checks that do not require the raw data archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = (
    "Codes/k11_download_nature_reference_data.py",
    "Codes/k12_compute_nature_reference_products.py",
    "Codes/k13_plot_nature_reference_layout_real.py",
    "Codes/k18_plot_figure3_driver_method_sensitivity.py",
    "Codes/validate_nature_main_figures.py",
    "Figure2_ENSO-forecast/k01_cal_source_region_prediction_errors.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    for relative in SCRIPTS:
        path = ROOT / relative
        require(path.is_file(), f"Missing script: {relative}")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    derived = ROOT / "Data/Nature_real_rebuild/derived"
    monthly = pd.read_csv(derived / "figure1_monthly_t1_t9.csv")
    require(monthly.AUC.between(0.0, 1.0).all(), "Figure 1 AUC outside [0, 1]")
    require((monthly.leads == 9).all(), "Figure 1 does not use exactly t+1...t+9")

    process = pd.read_csv(derived / "figure2_process_forecast_skill.csv")
    require(set(process.Lead) == set(range(1, 10)), "Figure 2 lead coverage is incomplete")
    require(np.isfinite(process.Pattern_correlation).all(), "Non-finite Figure 2 skill")

    with xr.open_dataset(derived / "figure2_source_process_maps_t1_t9.nc") as dataset:
        require(len(dataset.data_vars) >= 6, "Figure 2 map product is incomplete")
    with xr.open_dataset(derived / "figure3_driver_regime_and_mhw_intensity.nc") as dataset:
        require(len(dataset.data_vars) >= 2, "Figure 3 map product is incomplete")

    figure_manifest = json.loads((ROOT / "Figures/figure_manifest.json").read_text())
    require(figure_manifest.get("real_data_only") is True, "Figure manifest is not real-data-only")
    require(figure_manifest.get("simulated_values_used") is False, "Simulated values are flagged")

    for index in range(1, 5):
        directory = ROOT / f"Figures/Figure{index}"
        for suffix in ("png", "pdf", "md"):
            path = directory / f"Figure{index}.{suffix}"
            require(path.is_file() and path.stat().st_size > 0, f"Missing output: {path}")
        with Image.open(directory / f"Figure{index}.png") as image:
            require(image.width >= 1800 and image.height >= 1200, f"Low-resolution Figure {index}")
            require(float(np.asarray(image.convert("L"), dtype=np.float32).std()) > 8.0, f"Blank Figure {index}")

    print("[PASS] release scripts compile")
    print("[PASS] lightweight derived products are readable and finite")
    print("[PASS] Figure 1 uses target-aligned leads t+1...t+9")
    print("[PASS] Figures 1-4 and companion documentation are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
