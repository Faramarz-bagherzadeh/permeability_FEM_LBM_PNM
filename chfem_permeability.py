"""Directional permeability computation with CHFEM.

The input phase convention is 0 = pore/fluid and 1 = solid.  The public
function deliberately runs one CHFEM solve for each of x, y, and z instead of
using ``direction="all"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import ArrayLike


def compute_directional_permeability(
    binary: ArrayLike,
    *,
    voxel_size: float = 120e-6,
    directions: Sequence[str] = ("x", "y", "z"),
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
    velocity_output_dir: str | Path | None = None,
) -> dict[str, float] | dict[str, dict[str, float] | dict[str, str]]:
    """Compute CHFEM permeability independently in each requested direction.

    Parameters
    ----------
    binary
        Three-dimensional binary array. CHFEM's phase convention used here is
        ``0 = pore/fluid`` and ``1 = solid``.
    voxel_size
        Isotropic voxel edge length in metres.
    directions
        Ordered subset of ``("x", "y", "z")``. Each entry is passed to a
        separate ``chfem.compute_property`` call.
    solver, solver_tolerance, solver_maxiter, precondition
        Parameters forwarded directly to CHFEM.

    Returns
    -------
    dict
        Direction-to-permeability mapping in square metres. When
        ``velocity_output_dir`` is supplied, returns permeability and saved
        velocity paths in separate mappings.

    Notes
    -----
    This function expects a solver-ready volume. It does not downsample the
    image or fill disconnected pores.
    """
    try:
        import chfem
    except ImportError as exc:
        raise ImportError(
            "CHFEM is required. Install it with `conda install -c conda-forge "
            "chfem=2.0`."
        ) from exc

    domain = np.asarray(binary)
    if domain.ndim != 3:
        raise ValueError(f"binary must be a 3-D array; got shape {domain.shape}")
    if domain.size == 0:
        raise ValueError("binary must not be empty")
    if not np.all((domain == 0) | (domain == 1)):
        values = np.unique(domain)
        raise ValueError(
            "binary must contain only phase labels 0 and 1; "
            f"found {values[:20].tolist()}"
        )
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be a positive finite number")
    if solver_tolerance <= 0 or not np.isfinite(solver_tolerance):
        raise ValueError("solver_tolerance must be a positive finite number")
    if isinstance(solver_maxiter, bool) or not isinstance(
        solver_maxiter, (int, np.integer)
    ):
        raise TypeError("solver_maxiter must be an integer")
    if solver_maxiter <= 0:
        raise ValueError("solver_maxiter must be positive")

    requested = tuple(str(direction).lower() for direction in directions)
    if not requested:
        raise ValueError("directions must contain at least one direction")
    invalid = [
        direction for direction in requested if direction not in ("x", "y", "z")
    ]
    if invalid:
        raise ValueError(f"directions must be selected from x, y, z; got {invalid}")
    if len(set(requested)) != len(requested):
        raise ValueError("directions must not contain duplicates")

    domain = np.ascontiguousarray(domain, dtype=np.uint8)
    field_dir = Path(velocity_output_dir) if velocity_output_dir is not None else None
    if field_dir is not None:
        field_dir.mkdir(parents=True, exist_ok=True)
    if not np.any(domain == 0):
        permeability = {direction: 0.0 for direction in requested}
        if field_dir is None:
            return permeability
        paths = {}
        for direction in requested:
            path = field_dir / f"velocity_{direction}.npy"
            field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=domain.shape + (3,))
            field[:] = 0
            del field
            paths[direction] = str(path)
        return {"permeability": permeability, "velocity_field_paths": paths}

    permeability: dict[str, float] = {}
    paths: dict[str, str] = {}
    for direction in requested:
        with TemporaryDirectory(prefix=f"chfem_{direction}_") as work_dir:
            prefix = str(Path(work_dir) / "fields") if field_dir is not None else None
            try:
                raw = chfem.compute_property(
                    "permeability", domain, voxel_size=float(voxel_size),
                    solver=solver, solver_tolerance=float(solver_tolerance),
                    solver_maxiter=int(solver_maxiter), precondition=bool(precondition),
                    direction=direction, output_fields=prefix,
                )
            except Exception as exc:
                raise RuntimeError(f"CHFEM {direction.upper()}-direction solve failed") from exc

            if field_dir is not None:
                # CHFEM writes fields_velocity_{0,1,2}.bin for x, y, z;
                # pressure files use the same suffix. The official importer
                # returns input spatial shape + (ux, uy, uz).
                velocity_file = Path(f"{prefix}_velocity_{'xyz'.index(direction)}.bin")
                try:
                    if not velocity_file.is_file():
                        raise FileNotFoundError(velocity_file)
                    velocity = chfem.import_vector_field_from_chfem(
                        str(velocity_file), domain.shape, correct_direction=direction
                    )
                    if velocity.shape != domain.shape + (3,):
                        raise ValueError(f"unexpected velocity shape {velocity.shape}")
                    path = field_dir / f"velocity_{direction}.npy"
                    np.save(path, velocity)
                    paths[direction] = str(path)
                    del velocity
                except Exception as exc:
                    raise RuntimeError(
                        f"CHFEM {direction.upper()} solve succeeded, but velocity field export/import failed"
                    ) from exc

        # CHFEM versions/builds may return a scalar, a 3-vector, or a 3x3
        # tensor even when only one direction was requested.
        index = ("x", "y", "z").index(direction)
        value = np.asarray(raw, dtype=float).squeeze()
        if value.ndim == 0:
            directional_value = float(value)
        elif value.shape == (3, 3):
            directional_value = float(value[index, index])
        elif value.size == 9:
            directional_value = float(value.reshape(3, 3)[index, index])
        elif value.size == 3:
            directional_value = float(value.reshape(-1)[index])
        else:
            raise ValueError(
                f"CHFEM returned shape {value.shape} for direction "
                f"{direction!r}; expected a scalar, 3-vector, or 3x3 tensor"
            )

        if not np.isfinite(directional_value):
            raise ValueError(
                f"CHFEM returned non-finite permeability for {direction}: "
                f"{directional_value}"
            )
        permeability[direction] = directional_value

    if field_dir is not None:
        return {"permeability": permeability, "velocity_field_paths": paths}
    return permeability


def compute_directional_log_metrics(
    computed: Mapping[str, ArrayLike],
    ground_truth: Mapping[str, ArrayLike],
    *,
    directions: Sequence[str] = ("x", "y", "z"),
) -> dict[str, dict[str, float | int]]:
    """Calculate R2, MAE, and RMSE per direction after a log10 transform.

    Parameters
    ----------
    computed, ground_truth
        Mappings whose keys are directions and whose values are equally sized
        one-dimensional collections of permeability values in square metres.
        All values must be finite and strictly positive so that ``log10`` is
        defined.
    directions
        Ordered subset of ``("x", "y", "z")`` to evaluate.

    Returns
    -------
    dict[str, dict[str, float | int]]
        For each direction, returns ``n``, ``r2_log10``, ``mae_log10``, and
        ``rmse_log10``. Errors are measured in log10 units (decades).

    Notes
    -----
    R2 requires at least two samples and non-constant log10 ground truth. A
    ``ValueError`` is raised when either condition is not satisfied.
    """
    requested = tuple(str(direction).lower() for direction in directions)
    if not requested:
        raise ValueError("directions must contain at least one direction")
    invalid = [
        direction for direction in requested if direction not in ("x", "y", "z")
    ]
    if invalid:
        raise ValueError(f"directions must be selected from x, y, z; got {invalid}")
    if len(set(requested)) != len(requested):
        raise ValueError("directions must not contain duplicates")

    metrics: dict[str, dict[str, float | int]] = {}
    for direction in requested:
        if direction not in computed:
            raise KeyError(f"computed is missing direction {direction!r}")
        if direction not in ground_truth:
            raise KeyError(f"ground_truth is missing direction {direction!r}")

        predicted = np.asarray(computed[direction], dtype=float)
        expected = np.asarray(ground_truth[direction], dtype=float)
        if predicted.ndim != 1 or expected.ndim != 1:
            raise ValueError(
                f"{direction}: computed and ground_truth values must be 1-D"
            )
        if predicted.shape != expected.shape:
            raise ValueError(
                f"{direction}: computed shape {predicted.shape} does not match "
                f"ground_truth shape {expected.shape}"
            )
        if predicted.size < 2:
            raise ValueError(
                f"{direction}: at least two samples are required to compute R2"
            )
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
            raise ValueError(f"{direction}: permeability values must be finite")
        if np.any(predicted <= 0) or np.any(expected <= 0):
            raise ValueError(
                f"{direction}: permeability values must be greater than zero "
                "before applying log10"
            )

        log_predicted = np.log10(predicted)
        log_expected = np.log10(expected)
        residual = log_predicted - log_expected
        sum_squared_residual = float(np.sum(residual**2))
        sum_squared_total = float(
            np.sum((log_expected - np.mean(log_expected)) ** 2)
        )
        if sum_squared_total == 0.0:
            raise ValueError(
                f"{direction}: R2 is undefined because log10 ground truth is constant"
            )

        metrics[direction] = {
            "n": int(predicted.size),
            "r2_log10": float(1.0 - sum_squared_residual / sum_squared_total),
            "mae_log10": float(np.mean(np.abs(residual))),
            "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        }

    return metrics


def process_permeability_folder(
    image_folder: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    *,
    voxel_size: float = 120e-6,
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Solve sorted segmented TIFFs and save permeability, velocity, and log10 metrics.

    Saved velocities have TIFF spatial axis order plus ``(ux, uy, uz)``.
    The TIFF spatial order is normally ``(z, y, x)``. Zero-pore samples get
    zero fields and are explicitly excluded from log10 evaluation.
    """
    import pandas as pd
    import tifffile

    folder = Path(image_folder)
    files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")),
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No TIFF images found in {folder}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("TIFF files must have unique stems for velocity output directories")
    truth = pd.read_excel(ground_truth_path)
    columns = [f"permeability_{d}_m2" for d in "xyz"]
    missing = set(columns) - set(truth.columns)
    if missing:
        raise ValueError(f"Ground truth lacks columns: {sorted(missing)}")

    identifiers = next(
        (col for col in ("filename", "file", "image_name", "image", "sample_id", "sample")
         if col in truth.columns and truth[col].notna().all()),
        None,
    )
    if identifiers is None:
        if len(files) != len(truth):
            raise ValueError("Row-order matching requires equal TIFF and ground-truth row counts")
        matching = "sorted TIFF filenames to original Excel row order"
        lookup = dict(zip((p.name for p in files), truth.index))
    else:
        names = truth[identifiers].astype(str).map(lambda value: Path(value).name)
        if names.duplicated().any():
            raise ValueError(f"Ground-truth identifier column {identifiers!r} has duplicate filenames")
        lookup = dict(zip(names, truth.index))
        unmatched = [p.name for p in files if p.name not in lookup]
        if unmatched:
            raise ValueError(f"TIFF files missing from ground truth: {unmatched}")
        matching = f"filename column {identifiers}; {len(truth) - len(files)} unused ground-truth rows"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in files:
        domain = tifffile.imread(image)
        try:
            result = compute_directional_permeability(
                domain, voxel_size=voxel_size, solver=solver,
                solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
                precondition=precondition,
                velocity_output_dir=output / "velocity_fields" / image.stem,
            )
        except Exception as exc:
            raise RuntimeError(f"{image.name}: {exc}") from exc
        row = {"file": image.name}
        for direction in "xyz":
            predicted = result["permeability"][direction]
            expected = float(truth.loc[lookup[image.name], f"permeability_{direction}_m2"])
            if not np.isfinite(expected) or expected <= 0:
                raise ValueError(f"{image.name}: invalid ground-truth {direction} permeability: {expected}")
            if predicted < 0:
                raise ValueError(f"{image.name}: negative CHFEM {direction} permeability: {predicted}")
            row[f"permeability_{direction}_m2"] = predicted
            row[f"ground_truth_{direction}_m2"] = expected
            row[f"log10_permeability_{direction}"] = np.log10(predicted) if predicted > 0 else np.nan
            row[f"log10_ground_truth_{direction}"] = np.log10(expected)
            row[f"velocity_{direction}_path"] = result["velocity_field_paths"][direction]
        row["log10_exclusion"] = "zero CHFEM permeability" if any(
            row[f"permeability_{d}_m2"] == 0 for d in "xyz"
        ) else ""
        rows.append(row)
        del domain

    results = pd.DataFrame(rows)
    metrics = []
    for direction in "xyz":
        subset = results.dropna(subset=[f"log10_permeability_{direction}"])
        predicted = subset[f"log10_permeability_{direction}"].to_numpy()
        expected = subset[f"log10_ground_truth_{direction}"].to_numpy()
        residual = predicted - expected
        denominator = float(np.sum((expected - expected.mean()) ** 2)) if len(expected) else 0.0
        metrics.append({
            "direction": direction, "n": len(subset),
            "r2_log10": 1 - float(np.sum(residual**2)) / denominator
            if len(subset) >= 2 and denominator > 0 else np.nan,
            "mae_log10": float(np.mean(np.abs(residual))) if len(subset) else np.nan,
            "rmse_log10": float(np.sqrt(np.mean(residual**2))) if len(subset) else np.nan,
        })
    metric_table = pd.DataFrame(metrics)
    with pd.ExcelWriter(output / "permeability_results.xlsx") as writer:
        results.to_excel(writer, sheet_name="Results", index=False)
        pd.DataFrame({"matching_method": [matching], "tiff_count": [len(files)],
                      "ground_truth_rows": [len(truth)]}).to_excel(
                          writer, sheet_name="Run_Info", index=False)
    metric_table.to_excel(output / "permeability_metrics.xlsx", index=False)
    return results, metric_table


__all__ = [
    "compute_directional_permeability",
    "compute_directional_log_metrics",
    "process_permeability_folder",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CHFEM permeability and velocity export for TIFF images")
    parser.add_argument("image_folder")
    parser.add_argument("ground_truth_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process_permeability_folder(args.image_folder, args.ground_truth_path, args.output_dir)
