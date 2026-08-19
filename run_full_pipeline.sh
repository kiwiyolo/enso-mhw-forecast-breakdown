#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/enso-mhw-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
cd "${ROOT}"

"${PYTHON_BIN}" Codes/k11_download_nature_reference_data.py
"${PYTHON_BIN}" Codes/k12_compute_nature_reference_products.py
"${PYTHON_BIN}" Codes/k13_plot_nature_reference_layout_real.py --figures-root Figures
"${PYTHON_BIN}" Codes/k13_plot_nature_reference_layout_real.py --figures-root Figures --figure1-sedi-only
"${PYTHON_BIN}" Codes/k13_plot_nature_reference_layout_real.py --figures-root Figures --figure1-rmse-only
"${PYTHON_BIN}" Codes/k18_plot_figure3_driver_method_sensitivity.py --output-dir Figures/Figure3
"${PYTHON_BIN}" Codes/validate_nature_main_figures.py
