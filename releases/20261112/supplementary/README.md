# Supplementary figure revision v5: compact layout

All 17 figures regenerated with reader-facing captions and supplementary methods. Earlier releases and main Figures 1–4 are unchanged. S1c now displays all seven regions and all 63 lead values; all previously computed numerical tables are unchanged.

- Supplementary_Information.pdf: 5 methods/coverage pages and 17 captioned figures.
- Supplementary_Figures_with_captions.pdf: 17 full-width figures and captions; page heights follow content, avoiding unused A4 foot space.
- Supplementary_Figures_only.pdf: vector artwork only.
- Individual_SI_PDFs.zip: individual artwork and captioned PDFs.
- SI_overview.pdf: compact contact sheets.
- source_manifest.json, source_data/, derived_data/: input hashes and numerical products.
- LAYOUT_REVISION.md: current changes; qa/layout_comparison.csv: dimensions before and after.
- REVIEWER_RESPONSE.md: item-by-item response; SUBMISSION_BLOCKERS.md: remaining scientific checks.

S15 now matches Figure4_net_normalized_20261111_v27: +120.790563% and −20.790563% of the signed net map difference. S5 masks three unverified wind-stress lead-9 values, while retaining all 81 original records in source_data.

Reproduce: OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mhw_si_v5 /home/luoq/miniconda3/envs/ocean_sst/bin/python revise_si.py; then run validate_release.py and finalize_release.py.

Technical checks are not scientific submission clearance. SST initialization-selection and tropical-input cohort issues remain explicitly documented; no unsupported causal interpretation or independent-event claim is introduced.
