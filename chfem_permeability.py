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



def _write_float32_binary(
    file_object,
    array: np.ndarray,
    *,
    chunk_size: int = 1_048_576,
) -> None:
    """Write an array to a Legacy VTK binary block as big-endian float32."""
    flat = np.asarray(array).reshape(-1)
    for start in range(0, flat.size, chunk_size):
        values = np.asarray(flat[start : start + chunk_size], dtype=">f4")
        file_object.write(values.tobytes(order="C"))


def _write_zero_float32_binary(
    file_object,
    value_count: int,
    *,
    chunk_size: int = 1_048_576,
) -> None:
    """Write a zero-filled Legacy VTK binary block without a full-size array."""
    zero_chunk = np.zeros(min(value_count, chunk_size), dtype=">f4")
    remaining = value_count
    while remaining:
        count = min(remaining, zero_chunk.size)
        file_object.write(zero_chunk[:count].tobytes(order="C"))
        remaining -= count


def _write_legacy_vtk(
    path: str | Path,
    domain: np.ndarray,
    velocity: np.ndarray | None,
    voxel_size: float,
) -> Path:
    """Write voxel-centred image and velocity data to one binary Legacy VTK."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = domain.shape
    cell_count = int(nx * ny * nz)
    if velocity is not None and velocity.shape != domain.shape + (3,):
        raise ValueError(
            f"velocity shape {velocity.shape} does not match domain shape "
            f"{domain.shape} + (3,)"
        )

    header = (
        "# vtk DataFile Version 3.0\n"
        "CHFEM segmented domain and directional velocity\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n"
        "ORIGIN 0 0 0\n"
        f"SPACING {voxel_size:.17g} {voxel_size:.17g} {voxel_size:.17g}\n"
        f"CELL_DATA {cell_count}\n"
        "SCALARS image float 1\n"
        "LOOKUP_TABLE default\n"
    )

    with output_path.open("wb") as vtk_file:
        vtk_file.write(header.encode("ascii"))
        # C-order flattening of (z, y, x) makes X the fastest-varying VTK axis.
        _write_float32_binary(vtk_file, domain)
        vtk_file.write(b"\nVECTORS velocity float\n")
        if velocity is None:
            _write_zero_float32_binary(vtk_file, cell_count * 3)
        else:
            # The final axis is kept interleaved as (ux, uy, uz) for each cell.
            _write_float32_binary(vtk_file, velocity)
        vtk_file.write(b"\n")

    return output_path


def compute_directional_permeability(
    binary: ArrayLike,
    *,
    voxel_size: float = 120e-6,
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
    velocity_output_dir: str | Path | None = None,
    vtk_filename: str = "fields.vtk",
) -> dict[str, float] | dict[str, dict[str, float] | str]:
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
    velocity_output_dir
        Optional directory for one Legacy VTK file containing the image and
        assembled directional velocity vector.
    vtk_filename
        Name of the Legacy VTK file written inside ``velocity_output_dir``.

    Returns
    -------
    dict
        Direction-to-permeability mapping in square metres. When
        ``velocity_output_dir`` is supplied, returns permeability and the
        single saved ``vtk_path``.

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
    vtk_path = field_dir / vtk_filename if field_dir is not None else None
    if field_dir is not None:
        field_dir.mkdir(parents=True, exist_ok=True)
    if not np.any(domain == 0):
        permeability = {direction: 0.0 for direction in directions}
        if field_dir is None:
            return permeability
        _write_legacy_vtk(vtk_path, domain, None, float(voxel_size))
        return {"permeability": permeability, "vtk_path": str(vtk_path)}

    permeability: dict[str, float] = {}
    with TemporaryDirectory(prefix="chfem_vtk_") as vtk_work_dir:
        assembled_velocity = None
        if field_dir is not None:
            assembled_velocity = np.memmap(
                Path(vtk_work_dir) / "velocity.float32",
                mode="w+",
                dtype=np.float32,
                shape=domain.shape + (3,),
            )
            assembled_velocity[:] = 0

        for direction in directions:
            with TemporaryDirectory(prefix=f"chfem_{direction}_") as work_dir:
                prefix = (
                    str(Path(work_dir) / "fields")
                    if field_dir is not None
                    else None
                )
                raw = chfem.compute_property(
                    "permeability", domain, voxel_size=float(voxel_size),
                    solver=solver, solver_tolerance=float(solver_tolerance),
                    solver_maxiter=int(solver_maxiter), precondition=bool(precondition),
                    direction=direction, output_fields=prefix,
                )

                if assembled_velocity is not None:
                    # CHFEM's importer returns (z, y, x, [ux, uy, uz]). Keep
                    # the component parallel to each directional solve so the
                    # single VTK vector is (ux_from_x, uy_from_y, uz_from_z).
                    index = directions.index(direction)
                    velocity_file = Path(f"{prefix}_velocity_{index}.bin")
                    velocity = chfem.import_vector_field_from_chfem(
                        str(velocity_file), domain.shape, correct_direction=direction
                    )
                    if velocity.shape != domain.shape + (3,):
                        raise ValueError(
                            f"CHFEM {direction}-velocity shape {velocity.shape} "
                            "does not "
                            f"match {domain.shape + (3,)}"
                        )
                    np.copyto(
                        assembled_velocity[..., index],
                        velocity[..., index],
                        casting="unsafe",
                    )
                    del velocity

            # CHFEM versions/builds may return a scalar, a 3-vector, or a 3x3
            # tensor even when only one direction was solved.
            index = directions.index(direction)
            value = np.asarray(raw, dtype=float).squeeze()
            if value.ndim == 0:
                directional_value = float(value)
            elif value.size == 3:
                directional_value = float(value.reshape(-1)[index])
            else:
                directional_value = float(value.reshape(3, 3)[index, index])
            permeability[direction] = directional_value

        if assembled_velocity is not None:
            assembled_velocity.flush()
            _write_legacy_vtk(
                vtk_path, domain, assembled_velocity, float(voxel_size)
            )
            del assembled_velocity

    if field_dir is not None:
        return {"permeability": permeability, "vtk_path": str(vtk_path)}
    return permeability


def process_permeability_folder(
    image_folder: str | Path,
    output_dir: str | Path,
    *,
    metadata_file: str | Path | None = None,
    porosity_threshold: float = 0.02,
    voxel_size: float = 120e-6,
    solver: str = "minres",
    solver_tolerance: float = 1e-6,
    solver_maxiter: int = 2000,
    precondition: bool = False,
) -> "pd.DataFrame":
    """Solve selected segmented TIFFs and save permeability and VTK fields.

    Each VTK stores the voxel-centred image and assembled directional velocity
    as float32 cell data on the TIFF's physical ``(x, y, z)`` grid.
    When ``metadata_file`` is provided, only images listed in its ``filename``
    column with ``porosity_open`` above ``porosity_threshold`` are solved.
    """
    folder = Path(image_folder)
    selected_names = None
    if metadata_file is not None:
        metadata = pd.read_excel(metadata_file)
        required_columns = {"filename", "porosity_open"}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Metadata file {metadata_file} is missing required columns: {missing}"
            )
        open_porosity = pd.to_numeric(metadata["porosity_open"], errors="coerce")
        selected_names = set(
            metadata.loc[open_porosity > porosity_threshold, "filename"]
            .dropna()
            .astype(str)
            .str.strip()
        )

    files = sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
            and (selected_names is None or p.name in selected_names)
        ),
        key=lambda p: p.name,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in files:
        domain = tifffile.imread(image)
        result = compute_directional_permeability(
            domain, voxel_size=voxel_size, solver=solver,
            solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
            precondition=precondition,
            velocity_output_dir=output / "vtk_fields",
            vtk_filename=f"{image.stem}.vtk",
        )
        row = {"file": image.name}
        for direction in "xyz":
            row[f"permeability_{direction}_m2"] = result["permeability"][direction]
        row["vtk_path"] = result["vtk_path"]
        rows.append(row)
        del domain

    results = pd.DataFrame(rows)
    results.to_excel(output / "permeability_results.xlsx", index=False)
    return results


__all__ = [
    "compute_directional_permeability",
    "process_permeability_folder",]

if __name__ == "__main__":
    import argparse
    import time
    t1= time.time()
    parser = argparse.ArgumentParser(description="Run CHFEM permeability and velocity export for TIFF images")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--metadata_file", type=Path, default=Path(__file__).with_name("B51.xlsx")
    )
    parser.add_argument("--porosity_threshold", type=float, default=0.02)
    args = parser.parse_args()
    print("Input directory:", args.input_dir)
    print("Output directory:", args.output_dir)

    process_permeability_folder(
        args.input_dir,
        args.output_dir,
        metadata_file=args.metadata_file,
        porosity_threshold=args.porosity_threshold,
    )
    t2= time.time()
    print(f"Total time taken: {(t2-t1)/60:.2f} minutes")
