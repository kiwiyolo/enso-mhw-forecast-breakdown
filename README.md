# ENSO-MHW forecast-skill breakdown

Code and lightweight derived products supporting the manuscript
**"Breakdown of El Nino-enhanced marine heatwave predictability in
2023-2024"**.

Repository: <https://github.com/kiwiyolo/enso-mhw-forecast-breakdown>

This repository contains the active calculation and plotting workflow for
Figures 1-4 and the associated Figure 1 SEDI/RMSE and Figure 3 method-
sensitivity supplements. It also contains the compact derived products needed
to regenerate the published figures without redistributing the large NMME,
ERA5 and ORAS5 archives.

## Scope

The release supports two reproducibility levels:

1. **Figure reproduction from archived derived products.** This is the default,
   lightweight route and requires only this repository plus the documented
   Python environment.
2. **Full recalculation from source fields.** This additionally requires the
   locally archived forecast and reanalysis products described in
   [DATA_SOURCES.md](DATA_SOURCES.md). Those files are not redistributed here
   because of size and provider-specific access/licensing conditions.

No simulated values are used in the final figures. The supplied mock figures
were used only to define panel topology. The numerical products are calculated
from observations, reanalyses and archived forecasts; the exact boundaries of
the diagnostic interpretation are recorded under [Scientific boundaries](#scientific-boundaries).

## Repository structure

```text
.
|-- Codes/
|   |-- k11_download_nature_reference_data.py
|   |-- k12_compute_nature_reference_products.py
|   |-- k13_plot_nature_reference_layout_real.py
|   |-- k18_plot_figure3_driver_method_sensitivity.py
|   `-- validate_nature_main_figures.py
|-- Figure2_ENSO-forecast/
|   `-- k01_cal_source_region_prediction_errors.py
|-- Data/Nature_real_rebuild/
|   |-- raw/download_manifest.json
|   `-- derived/
|-- output/Figure1/candidate09_metrics/
|-- Figures/
|   |-- Figure1/
|   |-- Figure2/
|   |-- Figure3/
|   |-- Figure4/
|   |-- figure_manifest.json
|   `-- validation_report.json
|-- Experiments/2002_03_NMME_phase_space_correction/results/
|   `-- source_replacement_manifest.json
|-- tools/validate_release.py
|-- run_plotting.sh
|-- run_full_pipeline.sh
|-- CITATION.cff
|-- .zenodo.json
|-- environment.yml
`-- requirements.txt
```

## Active workflow

| Order | Script | Role | Main outputs |
|---:|---|---|---|
| 1 | `Codes/k11_download_nature_reference_data.py` | Download and checksum the openly hosted NOAA observational inputs | `Data/Nature_real_rebuild/raw/` |
| 2 | `Codes/k12_compute_nature_reference_products.py` | Calculate the Figure 1-4 diagnostic products | `Data/Nature_real_rebuild/derived/` |
| 3 | `Codes/k13_plot_nature_reference_layout_real.py` | Generate the four main figures and Figure 1 SEDI/RMSE supplements | `Figures/Figure1/` to `Figures/Figure4/` |
| 4 | `Codes/k18_plot_figure3_driver_method_sensitivity.py` | Generate the Figure 3 driver-classification sensitivity supplement | `Figures/Figure3/` |
| 5 | `Codes/validate_nature_main_figures.py` | Validate source checksums, derived products and final graphics in the full project layout | `Figures/validation_report.json` |

`k12_compute_nature_reference_products.py` imports
`Figure2_ENSO-forecast/k01_cal_source_region_prediction_errors.py` for the
source-region SST forecast calculations.

## Quick start: regenerate the figures

Create the exact tested environment with Conda:

```bash
conda env create -f environment.yml
conda activate enso-mhw-figures
```

Or install the Python dependencies into an existing environment:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
bash run_plotting.sh
```

This regenerates:

```text
Figures/Figure1/Figure1.png
Figures/Figure1/Figure1_SEDI.png
Figures/Figure1/Figure1_RMSE.png
Figures/Figure2/Figure2.png
Figures/Figure3/Figure3.png
Figures/Figure3/Figure3_driver_method_sensitivity.png
Figures/Figure4/Figure4.png
```

PDF versions and the panel-level CSV/JSON/NetCDF products are written beside
the PNG files. Random resampling is deterministic because the plotting and
calculation scripts use recorded seeds.

The quick route invokes `k18_plot_figure3_driver_method_sensitivity.py` with
`--plot-only`, so the included gridded diagnostics are redrawn without
redistributing the large ERSST source file. The full pipeline omits this flag
and recalculates those diagnostics from the downloaded source fields.

Run the lightweight release checks with:

```bash
python tools/validate_release.py
```

## Full recalculation

The analysis scripts retain the original project-relative path contract. For a
complete source-field recalculation, clone the repository as `kw_99_paper`
inside a project root with the following sibling data layout:

```text
PROJECT_ROOT/
|-- data/
|-- kw_99_OA-model/
`-- kw_99_paper/        # this repository
```

Populate the source archives listed in [DATA_SOURCES.md](DATA_SOURCES.md), then
run:

```bash
bash run_full_pipeline.sh
```

The full workflow is intentionally not silent about missing inputs: it stops at
the first absent source file instead of substituting synthetic values.

## Figure-to-product map

| Figure | Central question | Principal derived inputs |
|---|---|---|
| Figure 1 | Did the historical relationship between El Nino amplitude and global MHW forecast skill break down in 2023-2024? | `figure1_monthly_t1_t9.csv`, `figure1_event_relation.csv` |
| Figure 2 | Were tropical Pacific source-region forecast signals less faithful in 2023-2024 than in comparable events? | `figure2_source_process_maps_t1_t9.nc`, `figure2_process_forecast_skill.csv`, `figure2_source_signal_errors.csv` |
| Figure 3 | How did remote teleconnection and basin-local associations vary among ocean basins? | `figure3_driver_regime_and_mhw_intensity.nc`, `figure3_basin_driver_contributions.csv`, `figure3_teleconnection_efficiency.csv`, `figure3_local_mhw_activity.csv` |
| Figure 4 | Which pathway-linked errors account diagnostically for the 2023-2024 global AUC deficit? | `figure4_skill_loss_attribution.csv`, `figure4_path_samples.csv`, `figure4_attribution_summary.json` |

Panel-level definitions, uncertainty methods and captions are in the Markdown
file stored beside each figure.

## Core analysis conventions

- MHW thresholds use the 1985-2014 monthly climatological reference recorded in
  the release products.
- Figure 1 aggregates target-aligned NMME lead months `t+1` to `t+9`.
- Figure 2 uses target-aligned leads 1-9 and an equal lead mean for the process
  error maps.
- The primary comparable-event reference for Figures 2-4 is the equal mean of
  1997/98 and 2015/16; the exact cohort is retained in machine-readable
  provenance even where the labels are simplified in the graphics.
- Spatial means use latitude weighting where specified in the calculation
  scripts.
- Bootstrap iteration counts and random seeds are exposed as command-line
  arguments and recorded in the outputs.

## Scientific boundaries

The release distinguishes measured forecast skill from diagnostic
interpretation:

- The Figure 2 convection field is NMME precipitation, used as a convection
  proxy because the archived 2023 NMME OLR field is unavailable.
- Figure 3 driver classes are association-based statistical regimes, not a
  closed mixed-layer heat-budget attribution.
- The basin-local metric is a proxy based on ENSO-removed SST persistence and
  related diagnostics; it should not be interpreted as direct causal
  identification of one ocean process.
- Figure 4 allocates the Figure 1 expected-minus-observed AUC residual by
  association and cross-validated reconstruction. It is diagnostic accounting,
  not a causal intervention analysis.
- Exact Figure 1 source-field reconstruction depends on the NMME archive state
  recorded in
  `Experiments/2002_03_NMME_phase_space_correction/results/source_replacement_manifest.json`.
  The release does not redistribute those large forecast files. Independent
  downloads can differ unless their versions and checksums match the manifest.

These qualifications are also stored in
`Data/Nature_real_rebuild/derived/calculation_provenance.json`.

## Reproducibility records

- `archive_manifest.json`: SHA256 and byte size for every file assembled into
  the local release.
- `Data/Nature_real_rebuild/raw/download_manifest.json`: URL, provider, size and
  SHA256 for downloaded NOAA inputs.
- `Data/Nature_real_rebuild/derived/calculation_provenance.json`: climatology,
  lead alignment, local archive roots and known interpretation boundaries.
- `Figures/figure_manifest.json`: headline numerical results and active scripts.
- `Figures/validation_report.json`: completed validation checks for the released
  figure set.

## Citation

Please cite the archived software release using the citation generated from
`CITATION.cff` and the version-specific Zenodo DOI once available. The source
repository is <https://github.com/kiwiyolo/enso-mhw-forecast-breakdown>. The
software version in this release is `1.0.0`.

## License

The source code and original repository documentation are released under the
MIT License. Input data and third-party forecast products remain subject to the
terms of their respective providers; the MIT License does not relicense those
products.

## Contact

Correspondence concerning the manuscript should be addressed to Zhenzhong Zeng
(`zengzz@sustech.edu.cn`). Technical questions about this release can be opened
as issues in the GitHub repository.
