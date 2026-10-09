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
) -> None:
    """Write an array as big-endian float32 one z-slice at a time."""
    for z_slice in array:
        values = np.asarray(z_slice, dtype=">f4")
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


def _write_vti(
    path: str | Path,
    domain: np.ndarray,
    velocity_fields: np.ndarray | None,
    voxel_size: float,
) -> Path:
    """Write the image and three directional velocity fields to a binary VTI."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = domain.shape
    cell_count = int(nx * ny * nz)
    expected_velocity_shape = domain.shape + (3, 3)
    if velocity_fields is not None and velocity_fields.shape != expected_velocity_shape:
        raise ValueError(
            f"velocity fields shape {velocity_fields.shape} does not match "
            f"expected shape {expected_velocity_shape}"
        )

    array_specs = [("image", 1, cell_count)]
    array_specs.extend(
        (f"velocity_{direction}", 3, cell_count * 3)
        for direction in "xyz"
    )
    offsets = []
    offset = 0
    for _, _, value_count in array_specs:
        offsets.append(offset)
        offset += 8 + value_count * 4

    header = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="ImageData" version="1.0" byte_order="BigEndian" '
        'header_type="UInt64">\n',
        f'<ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" '
        f'Origin="0 0 0" Spacing="{voxel_size:.17g} '
        f'{voxel_size:.17g} {voxel_size:.17g}">\n',
        f'<Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n',
        '<CellData Scalars="image">\n',
    ]
    for (name, components, _), data_offset in zip(array_specs, offsets):
        component_attribute = (
            f' NumberOfComponents="{components}"' if components > 1 else ""
        )
        header.append(
            f'<DataArray type="Float32" Name="{name}"{component_attribute} '
            f'format="appended" offset="{data_offset}"/>\n'
        )
    header.extend([
        '</CellData>\n',
        '<PointData/>\n',
        '</Piece>\n',
        '</ImageData>\n',
        '<AppendedData encoding="raw">_',
    ])

    with output_path.open("wb") as vti_file:
        vti_file.write("".join(header).encode("ascii"))
        vti_file.write((cell_count * 4).to_bytes(8, byteorder="big"))
        _write_float32_binary(vti_file, domain)
        for index in range(3):
            vti_file.write((cell_count * 3 * 4).to_bytes(8, byteorder="big"))
            if velocity_fields is None:
                _write_zero_float32_binary(vti_file, cell_count * 3)
            else:
                _write_float32_binary(vti_file, velocity_fields[..., index, :])
        vti_file.write(b'</AppendedData>\n</VTKFile>\n')

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
    vtk_filename: str = "fields.vti",
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
        Optional directory for one VTI file containing the image and all three
        directional velocity vector fields.
    vtk_filename
        Name of the VTI file written inside ``velocity_output_dir``.

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
        _write_vti(vtk_path, domain, None, float(voxel_size))
        return {"permeability": permeability, "vtk_path": str(vtk_path)}

    permeability: dict[str, float] = {}
    with TemporaryDirectory(prefix="chfem_vtk_") as vtk_work_dir:
        assembled_velocity = None
        if field_dir is not None:
            assembled_velocity = np.memmap(
                Path(vtk_work_dir) / "velocity.float32",
                mode="w+",
                dtype=np.float32,
                shape=domain.shape + (3, 3),
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
                    # Preserve the full vector field for each directional solve.
                    index = directions.index(direction)
                    velocity_file = Path(f"{prefix}_velocity_{index}.bin")
                    velocity = chfem.import_vector_field_from_chfem(
                        str(velocity_file), domain.shape)
                    if velocity.shape == (3, *domain.shape):
                        velocity = np.moveaxis(velocity, 0, -1)
                    if velocity.shape != domain.shape + (3,):
                        raise ValueError(
                            f"CHFEM {direction}-velocity shape {velocity.shape} "
                            "does not "
                            f"match {domain.shape + (3,)}"
                        )
                    np.copyto(
                        assembled_velocity[..., index, :],
                        velocity,
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
            _write_vti(
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
            vtk_filename=f"{image.stem}.vti",
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
