#!/usr/bin/env python
"""Validate the downloaded inputs, real-data products and Nature figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PAPER_DIR / "Data/Nature_real_rebuild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-root", type=Path, default=PAPER_DIR / "Figures")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_downloads(root: Path) -> dict[str, object]:
    manifest_path = root / "raw/download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    require(len(sources) >= 4, "Download manifest does not contain all required official sources")
    checked = []
    for source in sources:
        path = Path(source["path"])
        require(path.is_file(), f"Missing downloaded source: {path}")
        require(path.stat().st_size == int(source["bytes"]), f"Byte-size mismatch: {path}")
        checksum = sha256(path)
        require(checksum == source["sha256"], f"SHA256 mismatch: {path}")
        checked.append({"product": source["product"], "bytes": path.stat().st_size, "sha256": checksum})
    return {"manifest": str(manifest_path.resolve()), "source_count": len(checked), "sources": checked}


def validate_images(root: Path) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for index in range(1, 5):
        name = f"Figure{index}"
        directory = root / name
        png = directory / f"{name}.png"
        pdf = directory / f"{name}.pdf"
        markdown = directory / f"{name}.md"
        for path in (png, pdf, markdown):
            require(path.is_file() and path.stat().st_size > 0, f"Missing or empty output: {path}")
        require("simulated" not in markdown.read_text(encoding="utf-8").lower(), f"Simulated-data wording remains in {markdown}")
        with Image.open(png) as image:
            grey = np.asarray(image.convert("L"), dtype=np.float32)
            require(image.width >= 1800 and image.height >= 1200, f"Low-resolution image: {png}")
            require(float(grey.std()) > 8.0, f"Image appears blank: {png}")
            results[name] = {
                "width": image.width,
                "height": image.height,
                "grayscale_std": float(grey.std()),
                "png_bytes": png.stat().st_size,
                "pdf_bytes": pdf.stat().st_size,
            }
    return results


def require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"Non-finite values in {label}: {columns}")


def validate_figure1(derived: Path) -> dict[str, object]:
    monthly = pd.read_csv(derived / "figure1_monthly_t1_t9.csv")
    events = pd.read_csv(derived / "figure1_event_relation.csv")
    audit = pd.read_csv(derived / "figure1_2002_03_auc_audit.csv")
    audit_summary = json.loads(
        (derived / "figure1_2002_03_auc_audit.json").read_text(encoding="utf-8")
    )
    require_finite(monthly, ["AUC", "leads", "Nino34"], "Figure 1 monthly table")
    require(monthly["AUC"].between(0, 1).all(), "Figure 1 AUC outside [0, 1]")
    require((monthly["leads"] == 9).all(), "Figure 1 must aggregate exactly target-aligned leads 1-9")
    require_finite(events, ["Peak_Nino34", "Mean_AUC", "Valid_Months"], "Figure 1 event table")
    require("2023/24" in set(events["Event"]), "Figure 1 target event is missing")
    require(len(audit) == 108, "Figure 1 2002/03 audit must contain 12 months x 9 leads")
    require(audit["Target_Month"].nunique() == 12, "Figure 1 2002/03 audit must contain 12 target months")
    require(set(audit["Lead_Month"]) == set(range(1, 10)), "Figure 1 2002/03 audit has unexpected leads")
    recalculated_column = (
        "Current_source_recalculated_AUC"
        if "Current_source_recalculated_AUC" in audit.columns
        else "Raw_recalculated_AUC"
    )
    require_finite(audit, ["Stored_AUC", recalculated_column], "Figure 1 source-field AUC audit")
    require(audit_summary["status"] == "passed", "Figure 1 source-field AUC audit did not pass")
    require(
        float(audit_summary["maximum_absolute_difference_from_raw_recalculation"]) <= 1e-12,
        "Figure 1 stored and source-field 2002/03 AUC values differ",
    )
    return {
        "monthly_start": str(monthly["time"].min()),
        "monthly_end": str(monthly["time"].max()),
        "event_count": len(events),
        "lead_count": int(monthly["leads"].iloc[0]),
        "event_2002_03_auc": float(audit_summary["stored_event_mean_auc"]),
        "event_2002_03_raw_max_delta": float(
            audit_summary["maximum_absolute_difference_from_raw_recalculation"]
        ),
    }


def validate_figure2(derived: Path) -> dict[str, object]:
    process_skill = pd.read_csv(derived / "figure2_process_forecast_skill.csv")
    require_finite(process_skill, ["Lead", "Pattern_correlation"], "Figure 2 process forecast skill")
    require(
        set(process_skill["Process"]) == {"SST", "Zonal wind stress", "Convection proxy"},
        "Figure 2 process set is incomplete",
    )
    canonical_events = {"1997/98", "2015/16"}
    require(
        set(process_skill["Event"]) == canonical_events | {"2023/24"},
        "Figure 2 event set is incomplete",
    )
    require(set(process_skill["Lead"]) == set(range(1, 10)), "Figure 2 must contain every lead from 1 to 9")
    process_means: dict[str, dict[str, float]] = {}
    for process, frame in process_skill.groupby("Process"):
        historical = float(frame[frame.Event != "2023/24"].Pattern_correlation.mean())
        target = float(frame[frame.Event == "2023/24"].Pattern_correlation.mean())
        require(target < historical, f"Figure 2 does not show lower 2023/24 fidelity for {process}")
        process_means[process] = {"historical": historical, "2023/24": target}
    with xr.open_dataset(derived / "figure2_source_process_maps_t1_t9.nc") as maps:
        map_variables = {
            "sst_comparable_relative_error",
            "sst_target_relative_error",
            "stress_comparable_relative_error",
            "stress_target_relative_error",
            "precipitation_comparable_relative_error",
            "precipitation_target_relative_error",
        }
        require(map_variables.issubset(maps.data_vars), "Figure 2 source-process maps are incomplete")
        require(
            maps.attrs.get("units") == "percent_of_observed_pattern_rms",
            "Figure 2 maps must use relative forecast errors",
        )
        for name in map_variables:
            require(np.isfinite(maps[name].values).any(), f"Figure 2 map contains no finite data: {name}")
    relative_scales = pd.read_csv(derived / "figure2_relative_error_scales.csv")
    require_finite(
        relative_scales,
        ["Observed_pattern_RMS"],
        "Figure 2 relative-error scales",
    )
    require(
        (relative_scales["Observed_pattern_RMS"] > 0).all(),
        "Figure 2 relative-error scales must be positive",
    )
    require(
        set(relative_scales.Process)
        == {"SST", "Zonal wind stress", "Convection proxy"},
        "Figure 2 relative-error process set is incomplete",
    )
    fidelity_audit = pd.read_csv(derived / "figure2_pattern_fidelity_lead_audit.csv")
    require_finite(
        fidelity_audit,
        ["Linear_slope_per_month", "Spearman_rho_lead_vs_fidelity"],
        "Figure 2 pattern-fidelity lead audit",
    )
    require(
        set(fidelity_audit.Process) == {"SST", "Zonal wind stress", "Convection proxy"},
        "Figure 2 fidelity-audit process set is incomplete",
    )
    target_audit = fidelity_audit[fidelity_audit.Event == "2023/24"]
    require(
        len(target_audit) == 3 and target_audit.Decreases_with_lead.astype(bool).all(),
        "Figure 2 target-event pattern fidelity should decline as forecast lead increases",
    )
    source_audit = pd.read_csv(derived / "figure2_nmme_sst_start_coordinate_audit.csv")
    require(
        {"usable", "start_month_mismatch", "lead_unavailable"}.issubset(set(source_audit.Reason)),
        "Figure 2 SST source audit does not exercise all expected source states",
    )
    require(
        source_audit.loc[source_audit.Usable.astype(bool), "Reason"].eq("usable").all(),
        "Figure 2 SST source audit marks an invalid source as usable",
    )
    errors = pd.read_csv(derived / "figure2_source_signal_errors.csv")
    require_finite(
        errors,
        ["Value", "Overall_mean_error", "Relative_to_overall_mean_percent"],
        "Figure 2 source errors",
    )
    require(set(errors.Group) == {"Comparable events", "2023/24"}, "Figure 2 error groups are inconsistent")
    require(
        (errors.Overall_mean_error > 0).all(),
        "Figure 2 overall source-error means must be positive",
    )
    expected_relative = 100.0 * (
        errors.Value - errors.Overall_mean_error
    ) / errors.Overall_mean_error
    require(
        np.allclose(
            errors.Relative_to_overall_mean_percent,
            expected_relative,
            rtol=0.0,
            atol=1e-10,
        ),
        "Figure 2 source-error labels and colour values use different transforms",
    )
    require(
        set(errors.loc[errors.Group == "Comparable events", "Group_event_count"]) == {2}
        and set(errors.loc[errors.Group == "2023/24", "Group_event_count"]) == {1},
        "Figure 2 source-error event weights are inconsistent",
    )
    event_errors = pd.read_csv(derived / "figure2_source_signal_errors_by_event.csv")
    require_finite(event_errors, ["Value"], "Figure 2 event-level source errors")
    require(
        set(event_errors.Event) == canonical_events | {"2023/24"},
        "Figure 2 event-level source-error cohort is inconsistent",
    )
    recalculated_overall = event_errors.groupby("Metric").Value.mean()
    stored_overall = errors.groupby("Metric").Overall_mean_error.first()
    require(
        np.allclose(
            stored_overall.sort_index(),
            recalculated_overall.sort_index(),
            rtol=0.0,
            atol=1e-12,
        ),
        "Figure 2 overall source-error means do not match the event-level data",
    )
    skill = pd.read_csv(derived / "figure2_source_mhw_skill.csv")
    require_finite(skill, ["AUC", "Event_count"], "Figure 2 source skill")
    require(skill["AUC"].between(0, 1).all(), "Figure 2 AUC outside [0, 1]")
    historical_summary = skill[
        (skill["Group"] == "Comparable events") & (skill["Record"] == "Summary")
    ]
    target_event = skill[(skill["Group"] == "2023/24") & (skill["Record"] == "Event")]
    require(len(historical_summary) == 1 and len(target_event) == 1, "Figure 2 comparison rows are incomplete")
    require(
        int(historical_summary.Event_count.iloc[0]) == 2,
        "Figure 2 historical canonical summary must contain two events",
    )
    require(
        float(target_event.AUC.iloc[0]) < float(historical_summary.AUC.iloc[0]),
        "Figure 2 target source-region AUC is not below the historical canonical mean",
    )
    return {
        "process_pattern_correlation_means": process_means,
        "lead_count": process_skill.Lead.nunique(),
        "decreasing_fidelity_series": int(fidelity_audit.Decreases_with_lead.astype(bool).sum()),
        "fidelity_series": int(len(fidelity_audit)),
        "historical_canonical_auc": float(historical_summary.AUC.iloc[0]),
        "target_auc": float(target_event.AUC.iloc[0]),
    }


def validate_figure3(derived: Path) -> dict[str, object]:
    with xr.open_dataset(derived / "figure3_driver_regime_and_mhw_intensity.nc") as dataset:
        regime = dataset["driver_regime"].values
        require(np.isin(regime, (-1, 0, 1, 2, 3)).all(), "Unexpected Figure 3 driver code")
        ocean = regime >= 0
        grid_sum = sum(dataset[name].values for name in ("direct_share", "remote_share", "local_share", "residual_share"))
        require(np.isfinite(grid_sum[ocean]).all(), "Figure 3 ocean driver shares contain missing values")
        require(np.allclose(grid_sum[ocean], 1.0, atol=1e-6), "Figure 3 ocean driver shares do not sum to one")
    basin = pd.read_csv(derived / "figure3_basin_driver_contributions.csv")
    columns = ["direct_share", "remote_share", "local_share", "residual_share"]
    require_finite(basin, ["MHW_intensity_C", *columns], "Figure 3 basin contributions")
    require(np.allclose(basin[columns].sum(axis=1), 1.0, atol=1e-8), "Figure 3 basin shares do not sum to one")
    teleconnection = pd.read_csv(derived / "figure3_teleconnection_efficiency.csv")
    require_finite(teleconnection, ["Historical", "2023/24", "Difference"], "Figure 3 teleconnection table")
    activity = pd.read_csv(derived / "figure3_local_mhw_activity.csv")
    require_finite(activity, ["Year", "MHW_activity_C_days"], "Figure 3 local-basin MHW activity")
    mean_label = "Local-dominated basin mean"
    mean_activity = activity[activity.Series == mean_label]
    basin_activity = activity[activity.Series != mean_label]
    require(not mean_activity.empty, "Figure 3 multi-basin mean activity is missing")
    require(set(mean_activity.Year) == set(range(1991, 2025)), "Figure 3 activity must cover 1991-2024")
    require(basin_activity.Series.nunique() >= 2, "Figure 3 needs multiple local-dominated basins")
    require(
        (basin_activity.groupby("Series").Year.nunique() == len(mean_activity)).all(),
        "Figure 3 basin activity time series have inconsistent coverage",
    )
    statistics = json.loads(
        (derived / "figure3_local_mhw_activity_statistics.json").read_text(encoding="utf-8")
    )
    require(np.isfinite(float(statistics["kendall_tau"])), "Figure 3 activity trend is non-finite")
    require(float(statistics["kendall_tau"]) > 0, "Figure 3 local-basin activity trend is not positive")
    require(
        0 <= float(statistics["kendall_p"]) <= 1,
        "Figure 3 Kendall trend P value is invalid",
    )
    return {
        "basin_count": len(basin),
        "bridge_count": len(teleconnection),
        "activity_year_count": len(mean_activity),
        "local_dominated_basin_count": basin_activity.Series.nunique(),
        "activity_theil_sen_per_decade": float(statistics["theil_sen_slope_C_days_per_decade"]),
        "activity_kendall_tau": float(statistics["kendall_tau"]),
        "activity_kendall_p": float(statistics["kendall_p"]),
    }


def validate_figure4(derived: Path, figures: Path) -> dict[str, object]:
    # The pre-plot product retains the Figure 1 residual and basin allocation.
    original = pd.read_csv(derived / "figure4_skill_loss_attribution.csv")
    original_global = original[original.Basin == "Global 60S-60N"]
    original_basins = original[original.Basin != "Global 60S-60N"]
    require(
        len(original_global) == 1 and len(original_basins) == 6,
        "Figure 4 source basin allocation is incomplete",
    )
    original_global = original_global.iloc[0]
    require(
        abs(float(original_basins.Total_skill_loss.sum()) - float(original_global.Total_skill_loss)) < 1e-10,
        "Figure 4 source basin losses do not sum to the global residual",
    )

    figure_dir = figures / "Figure4"
    attribution = pd.read_csv(figure_dir / "Figure4_skill_loss_attribution.csv")
    pieces = [
        "ENSO_source_linked",
        "Teleconnection_fidelity",
        "Regional_process_signal",
        "Unresolved",
    ]
    require_finite(attribution, ["Total_skill_loss", *pieces], "Figure 4 attribution")
    closure = np.abs(attribution[pieces].sum(axis=1) - attribution["Total_skill_loss"])
    require(float(closure.max()) < 1e-10, "Figure 4 attribution components do not close")
    require(len(attribution) == 6, "Figure 4 plotted basin allocation is incomplete")
    require(
        abs(float(attribution.Total_skill_loss.sum()) - float(original_global.Total_skill_loss)) < 1e-10,
        "Figure 4 basin losses do not sum to the global residual",
    )
    expected_attribution = original_basins.rename(
        columns={
            "ENSO_source_error": "ENSO_source_linked",
            "Teleconnection_error": "Teleconnection_fidelity",
            "Basin_local_error": "Regional_process_signal",
            "Irreducible": "Unresolved",
        }
    ).set_index("Basin")
    plotted_attribution = attribution.set_index("Basin")
    require(
        np.allclose(
            plotted_attribution.loc[expected_attribution.index, ["Total_skill_loss", *pieces]],
            expected_attribution[["Total_skill_loss", *pieces]],
            rtol=0,
            atol=1e-12,
        ),
        "Figure 4 panel-a values differ from the original attribution product",
    )
    events = pd.read_csv(derived / "figure1_event_relation.csv")
    historical = events[events.Event != "2023/24"]
    target = events[events.Event == "2023/24"].iloc[0]
    slope, intercept = np.polyfit(historical.Peak_Nino34, historical.Mean_AUC, 1)
    expected = float(intercept + slope * target.Peak_Nino34)
    residual_loss = max(0.0, expected - float(target.Mean_AUC))
    require(
        abs(float(original_global.Total_skill_loss) - residual_loss) < 1e-10,
        "Figure 4 total does not equal the Figure 1 expected-minus-observed residual",
    )
    summary = json.loads(
        (figure_dir / "Figure4_attribution_summary.json").read_text(encoding="utf-8")
    )
    require(
        abs(float(summary["global_skill_loss"]) - residual_loss) < 1e-10,
        "Figure 4 summary residual does not match Figure 1",
    )
    mechanism_percent = summary["mechanism_percent"]
    require(
        abs(sum(float(value) for value in mechanism_percent.values()) - 100.0) < 1e-8,
        "Figure 4 mechanism percentages do not sum to 100",
    )
    models = pd.read_csv(figure_dir / "Figure4_cross_validated_models.csv")
    require(len(models) == 4, "Figure 4 cross-validation table is incomplete")
    require(
        models.Model.iloc[0] == "Location + lead",
        "Figure 4 reference model is missing",
    )
    require_finite(
        models,
        ["CV_R2", "CV_R2_CI_low", "CV_R2_CI_high", "CV_RMSE", "CV_MAE"],
        "Figure 4 cross-validated models",
    )
    error_reduction = pd.read_csv(figure_dir / "Figure4_error_reduction.csv")
    require(len(error_reduction) == 4, "Figure 4 paired error table is incomplete")
    require_finite(
        error_reduction,
        ["Baseline_MAE", "Pathway_MAE", "Relative_MAE_reduction"],
        "Figure 4 paired error comparison",
    )
    overall_error = error_reduction.iloc[0]
    require(
        float(overall_error.Relative_MAE_reduction) > 0,
        "Figure 4 pathway model does not improve held-out MAE",
    )
    sensitivity = pd.read_csv(figure_dir / "Figure4_allocation_sensitivity.csv")
    require(len(sensitivity) == 15, "Figure 4 allocation sensitivity table is incomplete")
    require(
        int(sensitivity.Is_primary_definition.sum()) == 1,
        "Figure 4 allocation sensitivity has an invalid primary definition",
    )
    require_finite(
        sensitivity,
        [
            "Pacific_source_signal_percent",
            "Atmospheric_pathway_percent",
            "Regional_process_signal_percent",
        ],
        "Figure 4 allocation sensitivity",
    )
    reconstruction = pd.read_csv(figure_dir / "Figure4_out_of_sample_reconstruction.csv")
    require_finite(
        reconstruction,
        [
            "Observed_skill_loss",
            "Diagnosed_skill_loss",
            "Cross_validated_baseline_AUC",
            "Cross_validated_AUC",
        ],
        "Figure 4 out-of-sample reconstruction",
    )
    require(
        len(reconstruction) == 72
        and reconstruction.Event.nunique() == 3
        and reconstruction.Basin.nunique() == 6
        and set(reconstruction.Lead) == {1, 3, 6, 9},
        "Figure 4 validation samples do not cover 3 events x 6 basins x 4 leads",
    )
    calibration = pd.read_csv(
        figure_dir / "Figure4_out_of_sample_calibration.csv"
    )
    require(
        list(calibration.Inferred_loss_group) == ["Low", "Medium", "High"]
        and list(calibration.Samples) == [24, 24, 24],
        "Figure 4 calibration groups are not three equal inferred-loss thirds",
    )
    require_finite(
        calibration,
        [
            "Inferred_skill_loss",
            "Actual_skill_loss",
            "Absolute_gap",
            "Agreement_percent",
        ],
        "Figure 4 grouped out-of-sample calibration",
    )
    require(
        calibration.Inferred_skill_loss.is_monotonic_increasing
        and calibration.Actual_skill_loss.is_monotonic_increasing,
        "Figure 4 inferred and actual grouped losses do not share the low-to-high ordering",
    )
    require(
        float(calibration.Absolute_gap.mean()) < 0.025,
        "Figure 4 grouped inferred and actual losses are insufficiently aligned",
    )
    inferred_magnitude = calibration.Inferred_skill_loss.abs().to_numpy(dtype=float)
    actual_magnitude = calibration.Actual_skill_loss.abs().to_numpy(dtype=float)
    larger_magnitude = np.maximum(inferred_magnitude, actual_magnitude)
    same_direction = (
        np.sign(calibration.Inferred_skill_loss.to_numpy(dtype=float))
        == np.sign(calibration.Actual_skill_loss.to_numpy(dtype=float))
    )
    expected_level_agreement = np.where(
        larger_magnitude < 1e-12,
        100.0,
        np.where(
            same_direction,
            100.0 * np.minimum(inferred_magnitude, actual_magnitude) / larger_magnitude,
            0.0,
        ),
    )
    require(
        np.allclose(
            calibration.Agreement_percent.to_numpy(dtype=float),
            expected_level_agreement,
            rtol=0,
            atol=1e-12,
        )
        and calibration.Agreement_percent.between(0.0, 100.0).all(),
        "Figure 4 level-specific agreement percentages are inconsistent",
    )
    summary_level_agreement = summary["calibration_level_agreement_percent"]
    require(
        all(
            abs(float(summary_level_agreement[level]) - expected_level_agreement[index])
            < 1e-12
            for index, level in enumerate(("Low", "Medium", "High"))
        ),
        "Figure 4 summary level-specific agreement percentages are inconsistent",
    )
    actual_group = calibration.Actual_skill_loss.to_numpy(dtype=float)
    inferred_group = calibration.Inferred_skill_loss.to_numpy(dtype=float)
    grouped_agreement = 1.0 - np.sum((actual_group - inferred_group) ** 2) / np.sum(
        (actual_group - actual_group.mean()) ** 2
    )
    require(
        abs(float(summary["calibration_group_agreement"]) - grouped_agreement) < 1e-12
        and 0.0 <= grouped_agreement <= 1.0,
        "Figure 4 grouped agreement percentage is inconsistent",
    )
    require(
        0.0 <= float(summary["full_model_cv_r2"]) <= 1.0
        and 0.0 <= float(summary["reconstruction_cv_r2"]) <= 1.0,
        "Figure 4 cross-validated skill is invalid",
    )
    return {
        "basin_count": len(attribution),
        "global_residual_auc": residual_loss,
        "maximum_closure_error": float(closure.max()),
        "validation_samples": len(reconstruction),
        "held_out_events": reconstruction.Event.nunique(),
        "full_model_cv_r2": float(summary["full_model_cv_r2"]),
        "relative_mae_reduction": float(summary["relative_mae_reduction"]),
        "reconstruction_cv_r2": float(summary["reconstruction_cv_r2"]),
        "reconstruction_rmse": float(summary["reconstruction_rmse"]),
        "calibration_group_mae": float(summary["calibration_group_mae"]),
        "calibration_group_agreement": float(summary["calibration_group_agreement"]),
        "calibration_level_agreement_percent": {
            level: float(summary_level_agreement[level])
            for level in ("Low", "Medium", "High")
        },
    }


def main() -> int:
    args = parse_args()
    figures = args.figures_root.resolve()
    data_root = args.data_root.resolve()
    derived = data_root / "derived"
    manifest = json.loads((figures / "figure_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("real_data_only") is True, "Figure manifest does not assert real-data-only output")
    require(manifest.get("simulated_values_used") is False, "Figure manifest reports simulated values")
    report = {
        "status": "passed",
        "figures_root": str(figures),
        "data_root": str(data_root),
        "downloads": validate_downloads(data_root),
        "images": validate_images(figures),
        "Figure1": validate_figure1(derived),
        "Figure2": validate_figure2(derived),
        "Figure3": validate_figure3(derived),
        "Figure4": validate_figure4(derived, figures),
    }
    output = figures / "validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
