# Data sources and local layout

The repository includes lightweight derived products for direct figure
reproduction. Full source-field recalculation additionally requires the data
archives below. Large forecast and reanalysis files are not redistributed in
the GitHub/Zenodo software release.

## Openly downloaded inputs

`Codes/k11_download_nature_reference_data.py` downloads and verifies:

| Product | Provider | Use |
|---|---|---|
| ERSSTv5 monthly SST | NOAA Physical Sciences Laboratory | Observed SST and anomaly fields |
| Monthly OLR CDR v03r00 | NOAA NCEI | Observed convection diagnostic |
| ERSSTv5 monthly Nino indices | NOAA Climate Prediction Center | ENSO event detection and amplitude |
| Detrended Nino3.4 index | NOAA Climate Prediction Center | ENSO sensitivity checks |

URLs and SHA256 values for the downloaded snapshot are recorded in
`Data/Nature_real_rebuild/raw/download_manifest.json`.

## Local forecast and reanalysis archives

The complete calculation expects these paths relative to `PROJECT_ROOT`:

```text
data/COLA-RSMAS-CCSM4/
data/NASA-GEOSS2S/
data/GFDL-SPEAR/
data/NCEP-CFSv2/
data/ersst_observation.nc
data/ORAS5/
kw_99_paper/teleconnection_data/NMME/
kw_99_paper/teleconnection_data/observations/ERA5/
kw_99_OA-model/outputs/evaluation/
```

The NMME SST files use initialization (`S`), lead (`L`), ensemble member (`M`)
and horizontal coordinates. The workflow audits the internal initialization
coordinate before using a file and aligns each forecast to its target month.
NMME leads begin with the initialization month in several source archives; the
analysis converts these source coordinates to the explicit target-aligned
`t+1` to `t+9` contract used in the paper.

ORAS5 and ERA5 are used for ocean-process and atmospheric pathway diagnostics.
The model-evaluation archive supplies the fixed common evaluation mask used by
the MHW skill calculations.

## Included derived products

The complete `Data/Nature_real_rebuild/derived/` directory is included because
it is compact and allows every released figure to be regenerated without the
large source archives. CSV and JSON files are plain text; NetCDF files preserve
the spatial fields used by Figures 2 and 3.

## Data-use boundaries

- Provider data retain their original licenses and citation requirements.
- The repository license covers code and original documentation only.
- The current NMME source state used for Figure 1 is identified by the checksum
  manifest under
  `Experiments/2002_03_NMME_phase_space_correction/results/`.
- The release does not claim that association-based driver classes are causal
  heat-budget attribution.
