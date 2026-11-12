# November 2026 code and source-data snapshot

**Unexpectedly low marine heatwave forecast skill during the 2023-2024 El Niño**

This dated snapshot supersedes the September numerical and graphical products for the current manuscript. It contains the final Fig. 1-4 source tables and gridded fields, all 17 Supplementary Figures and captions, calculation scripts, frozen original plotting scripts, and portability adaptations. The older repository-root products remain historical and must not be combined with this snapshot.

## Reproduction

Use Python 3.10 and the versions in `requirements.txt`.

```bash
python validate_snapshot.py
python main13/render.py
python main4/plot_figure.py
python supplementary/revise_si.py --only 1 2 3 4 6 8 9 10 11 12 13 14 15 16 17
python supplementary/matched_s5/render.py
python repackage_si.py
```

Run these commands from this directory. Plot outputs are written under each package's `figures/`. The original final PDFs are frozen delivery products. Supplementary Fig. 7 is retained as its unchanged original export; its original plotting implementation is also included. Cartopy may download public Natural Earth coastlines on first use. Fonts are included with their licence notices.

Portability edits change source lookup and entry points only. They do not alter data, metrics, map smoothing, colour ranges or statistical definitions. All four main figures and sixteen supplementary figures were re-rendered in the recorded environment and matched the final reference artwork exactly at 120 dpi (maximum pixel difference zero). Supplementary Fig. 7 remains the unchanged original export. Verification is recorded in `validation.json` and `reproduction_validation.json`; no full raw-field recalculation or clean external-environment run is claimed.

## What is and is not included

- `main13/`: data and plotting code for Fig. 1-3.
- `main4/`: local AUC map, signed regional accounting, variance fractions and Fig. 4 code.
- `supplementary/`: supplementary data, code, individual figures, captions and compiled SI.
- `calculation_provenance/`: calculation scripts for the original research filesystem, including model/target selection and local predictor diagnostics. Full source-field recalculation additionally requires the provider data and path configuration described in `DATA_SOURCES.md`.
- `original_scripts/`: unmodified production plotting code, retained to audit portability changes.
- `source_provenance.json` and `checksums.json`: input provenance and packaged-file checksums.

This is a derived-data release. It does not redistribute full NMME, ERA5, ORAS5 or CFSv2 archives, nor does it claim those raw-field workflows have been rerun in a clean external environment.

## Scientific interpretation

The archived SST initialization-selection issue is documented in the manuscript and supplementary methods and is not corrected by this release. Fig. 2 retains variable-specific time windows; Supplementary Fig. 5 is the matched single-model comparison. The Fig. 4 accounting is a statistical allocation using historical SST-variance fractions, not a measured causal decomposition of forecast-process errors. The net mapped AUC quantity is distinct from Fig. 1's global AUC gap. Validating packaged checksums or regenerating artwork does not establish scientific submission clearance.

The public repository URL remains https://github.com/kiwiyolo/enso-mhw-forecast-breakdown . A previous Zenodo DOI identifies a historical release; it must not be described as archiving these files until a new record version has been verified.
