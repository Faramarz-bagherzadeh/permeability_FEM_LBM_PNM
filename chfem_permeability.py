"""Directional permeability computation with CHFEM.

The input phase convention is 0 = pore/fluid and 1 = solid.  The public
function deliberately runs one CHFEM solve for each of x, y, and z instead of
using ``direction="all"``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
import tifffile

def compute_directional_permeability(
    binary: ArrayLike,
    *,
    voxel_size: float = 120e-6,
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
    velocity_output_dir: str | Path | None = None,
) -> dict[str, float] | dict[str, dict[str, float] | dict[str, str]]:
    """Compute CHFEM permeability independently in the X, Y, and Z directions.

    Parameters
    ----------
    binary
        Three-dimensional binary array. CHFEM's phase convention used here is
        ``0 = pore/fluid`` and ``1 = solid``.
    voxel_size
        Isotropic voxel edge length in metres.
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
    import chfem

    domain = np.asarray(binary)
    directions = ("x", "y", "z")

    domain = np.ascontiguousarray(domain, dtype=np.uint8)
    field_dir = Path(velocity_output_dir) if velocity_output_dir is not None else None
    if field_dir is not None:
        field_dir.mkdir(parents=True, exist_ok=True)
    if not np.any(domain == 0):
        permeability = {direction: 0.0 for direction in directions}
        if field_dir is None:
            return permeability
        paths = {}
        for direction in directions:
            path = field_dir / f"velocity_{direction}.npy"
            field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=domain.shape + (3,))
            field[:] = 0
            del field
            paths[direction] = str(path)
        return {"permeability": permeability, "velocity_field_paths": paths}

    permeability: dict[str, float] = {}
    paths: dict[str, str] = {}
    for direction in directions:
        with TemporaryDirectory(prefix=f"chfem_{direction}_") as work_dir:
            prefix = str(Path(work_dir) / "fields") if field_dir is not None else None
            raw = chfem.compute_property(
                "permeability", domain, voxel_size=float(voxel_size),
                solver=solver, solver_tolerance=float(solver_tolerance),
                solver_maxiter=int(solver_maxiter), precondition=bool(precondition),
                direction=direction, output_fields=prefix,
            )

            if field_dir is not None:
                # CHFEM writes fields_velocity_{0,1,2}.bin for x, y, z;
                # pressure files use the same suffix. The official importer
                # returns input spatial shape + (ux, uy, uz).
                velocity_file = Path(f"{prefix}_velocity_{'xyz'.index(direction)}.bin")
                velocity = chfem.import_vector_field_from_chfem(
                    str(velocity_file), domain.shape)
                path = field_dir / f"velocity_{direction}.npy"
                np.save(path, velocity)
                paths[direction] = str(path)
                del velocity

        # CHFEM versions/builds may return a scalar, a 3-vector, or a 3x3
        # tensor even when only one direction was solved.
        index = ("x", "y", "z").index(direction)
        value = np.asarray(raw, dtype=float).squeeze()
        if value.ndim == 0:
            directional_value = float(value)
        elif value.size == 3:
            directional_value = float(value.reshape(-1)[index])
        else:
            directional_value = float(value.reshape(3, 3)[index, index])
        permeability[direction] = directional_value

    if field_dir is not None:
        return {"permeability": permeability, "velocity_field_paths": paths}
    return permeability


def process_permeability_folder(
    image_folder: str | Path,
    output_dir: str | Path,
    *,
    voxel_size: float = 120e-6,
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
) -> "pd.DataFrame":
    """Solve sorted segmented TIFFs and save permeability and velocity fields.

    Saved velocities have TIFF spatial axis order plus ``(ux, uy, uz)``.
    The TIFF spatial order is normally ``(z, y, x)``.
    """
    folder = Path(image_folder)
    files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")),
        key=lambda p: p.name,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in files[:100]:
        print(f"Processing {image.name}...")
        domain = tifffile.imread(image)
        result = compute_directional_permeability(
            domain, voxel_size=voxel_size, solver=solver,
            solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
            precondition=precondition,
            velocity_output_dir=output / "velocity_fields" / image.stem,
        )
        row = {"file": image.name}
        for direction in "xyz":
            row[f"permeability_{direction}_m2"] = result["permeability"][direction]
            row[f"velocity_{direction}_path"] = result["velocity_field_paths"][direction]
        rows.append(row)
        del domain

    results = pd.DataFrame(rows)
    results.to_excel(output / "permeability_results.xlsx", index=False)
    return results


__all__ = [
    "compute_directional_permeability",
    "process_permeability_folder",
]


if __name__ == "__main__":
    import argparse
    import time
    t1= time.time()
    parser = argparse.ArgumentParser(description="Run CHFEM permeability and velocity export for TIFF images")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    print("Input directory:", args.input_dir)
    print("Output directory:", args.output_dir)

    process_permeability_folder(args.input_dir, args.output_dir)
    t2= time.time()
    print(f"Total time taken: {(t2-t1)/60:.2f} minutes")
