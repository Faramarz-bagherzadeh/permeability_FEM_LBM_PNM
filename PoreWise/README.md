<h1 align="center">PoreWise</h1>

<p align="center">
  <a href="https://github.com/SalehMohammadrezaei/PoreWise/actions/workflows/cpu.yml"><img src="https://github.com/SalehMohammadrezaei/PoreWise/actions/workflows/cpu.yml/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licence"></a>
  <a href="https://doi.org/10.5281/zenodo.22896541"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22896541.svg" alt="DOI"></a>
</p>

<p align="center">
  <img src="docs/figures/bentheimer_streamlines.png" width="80%" alt="Flow through Bentheimer sandstone computed with PoreWise">
</p>

## Overview

**PoreWise** computes the absolute permeability of porous materials directly from 2D and 3D
images, such as segmented micro-CT scans of rocks, foams, filters and microfluidic devices.
It solves creeping flow in the pore space with the lattice Boltzmann method and returns the
permeability in each direction or the full permeability tensor.

PoreWise runs on a single GPU or on CPU cores, and is designed so that images of a billion
voxels fit on one workstation.

Formerly LBM-Permeability. Version 0.1.0 is archived on Zenodo, see Citing below.

## Capabilities

- **Permeability tensor** of a 3D image (or 2x2 in 2D) from three independent flow directions,
  with principal values, principal directions and a symmetry check
- **Sealed-sample mode** (`mirror=True`): solves on the mirrored image, which removes the
  artificial resistance a periodic wrap adds to a non-periodic sample
- **Two collision models**: BGK, and two-relaxation-time (TRT), which gives a permeability that
  does not depend on the chosen relaxation time
- **Pore-only solvers** for GPU (`cuda-sparse`) and CPU (`numba-sparse`): solid voxels take no
  memory and no computing time
- **Large images**: a 1024-cube sandstone runs on one 48 GB GPU in a few hours
- **Default output**: x, y, z permeability in m² and one Float32 VTK ImageData file
  containing the input image and three vector velocity fields per 3D image
- **Fixed-step runs**: the default command line performs no convergence or numerical quality
  checks, and applies no permeability acceptance gate
- **Pore-space tools**: porosity, connectivity report, local-thickness distribution, synthetic
  test geometries
- **Command line and Python interfaces**

## Installation

```bash
git clone https://github.com/SalehMohammadrezaei/PoreWise.git
cd PoreWise
python -m venv .venv && source .venv/bin/activate
pip install .            # core, NumPy reference solvers
pip install '.[gpu]'     # CuPy, for NVIDIA GPUs (CUDA 12)
pip install '.[cpu]'     # Numba, for the multi-core CPU solver
pip install '.[viz]'     # Matplotlib and PyVista, for the examples
```

## Quick start

Inputs and solver settings are used without explicit runtime validation.
Python and dependencies can still raise errors during loading, computation or
export. Empty input selections perform no simulations.

From the command line, for a segmented 3D image stored as a TIFF volume:

```bash
python -m porewise rock.tif --backend cuda-sparse --collision trt \
    --solid-value 255 --pore-value 0 --dx 5e-6 --steps 20000 --output results/rock
```

TIFF inputs (`.tif` / `.tiff`) are loaded directly into NumPy arrays in memory,
one volume at a time. No intermediate `.npy` files are written. Existing `.npy`
inputs also work, and formats may be mixed in one batch. Each TIFF must contain
a complete 3D volume in `(z, y, x)` order; separate slice files are not stacked.
Input labels and dtype are preserved. After all three directions and exports
finish, the image arrays are released before the next input is read. Original
input files are retained; existing `.npy` copies are not automatically deleted.

`--dx` is the voxel size in metres. Use `--backend numba-sparse` for CPU execution.
Every image runs three independent simulations from rest, driven along x, y and z.
Each simulation performs exactly `--steps` updates. There are no convergence, mass,
density, finite-state, Mach, Reynolds or permeability acceptance checks in this workflow.
Basic input and backend requirements still apply (a segmented 3D image, valid voxel size
and supported solver settings). No diagnostic, provenance or error-report files are written.

The output folder contains only:

- `permeability.json`: `k_x_m2`, `k_y_m2`, `k_z_m2`, rounded to float32 precision.
- `fields.vti`: one VTK ImageData file containing the input image and all three
  vector velocity fields as **Float32 point-data arrays**.

| Array in `fields.vti` | Meaning | Components |
|---|---|---|
| `MicroCT` | Original input image values, converted to float32 | One scalar |
| `Velocity_X` | Vector field for x-driven flow | `(ux, uy, uz)` |
| `Velocity_Y` | Vector field for y-driven flow | `(ux, uy, uz)` |
| `Velocity_Z` | Vector field for z-driven flow | `(ux, uy, uz)` |

The image and vectors share the same grid. Input array axes are `(z, y, x)`;
VTK dimensions are `(nx, ny, nz)` with x varying fastest. Spacing is `--dx` in
metres, and origin `(0, 0, 0)` is the first voxel centre. `MicroCT` is the active
scalar and `Velocity_X` the initial active vector. Labels are preserved: an input
encoded as pore=0, solid=255 retains those values in `MicroCT`.

Velocities remain in lattice units (`dx/dt`), not m/s: a physical time scale has
not been supplied. Solid voxels have zero velocity. Permeability uses the velocity
averaged over the full sample volume and the conversion `k_m2 = k_lu * dx**2`.
The exported arrays and Python results use float32; internal computation keeps
the solver precision selected by `--precision` (default float64). JSON numbers
have no dtype, so the permeability values are rounded to float32 before writing.
The exporter streams one load at a time and does not require VTK to be installed.

Process several images with the same voxel size and settings:

```bash
python -m porewise rock1.tif rock2.tiff --backend numba-sparse --collision trt \
    --solid-value 255 --pore-value 0 --dx 5e-6 --output results
```

This creates `results/001_rock1/` and `results/002_rock2/`, each with the two files above.
Use a new output folder to avoid retaining files from an earlier run, including old `.npz` files.

From Python:

```python
import numpy as np
from porewise import compute_directional_permeability, load_image

solid = load_image("rock.tif") == 255  # True = solid; array axes (z, y, x)
result = compute_directional_permeability(
    solid, voxel_size=5e-6, backend="numba-sparse", collision="trt",
    tau=0.6, n_steps_max=20000, verbose=False,
)
print(result["k_x_m2"], result["k_y_m2"], result["k_z_m2"])  # np.float32 values
x_flow = result["velocity_fields"]["x"]
velocity_vector = np.stack([x_flow["ux"], x_flow["uy"], x_flow["uz"]], axis=-1)  # float32
```

For a non-periodic scanned sample, `--mirror` (Python: `mirror=True`) computes the
sealed-sample directional permeabilities. Its velocity fields cover the mirrored domain,
which is twice as large along every axis. The default fields match the original image.

The existing 2D solvers, full tensor API and visualization/export utilities remain available
for library use; their existing diagnostic behavior is independent of this minimal workflow.
See [numerical method, conventions and units](docs/numerical_limits.md).

## Choosing a solver

| Backend | Images | Runs on | Use it for |
|---|---|---|---|
| `cuda-sparse` | 3D | NVIDIA GPU | real samples; fastest and most memory-efficient |
| `numba-sparse` | 3D | CPU cores | the same method without a GPU; set `NUMBA_NUM_THREADS` |
| `cuda` | 2D, 3D | NVIDIA GPU | 2D images; dense 3D reference |
| `numpy`, `cupy-array` | 2D, 3D | CPU, GPU | small cases, teaching, cross-checks |

We recommend `collision="trt"` with `tau=0.6` for rock images.

On a 384-cube sandstone crop (porosity 0.25) `cuda-sparse` processes 4,800 million voxels
per second in 4 GB on an RTX 6000 Ada, and `numba-sparse` reaches 1,000 million on 96 cores.

## Documentation

- [Numerical method, conventions and units](docs/numerical_limits.md)
- [Verification and performance](docs/verification.md): analytical channels and pipes, sphere
  arrays, a published micromodel benchmark and published 1024-cube rock benchmarks
- [Reproducing the benchmarks](docs/benchmark_reproduction.md)

## Citing

If you use PoreWise in your work, please cite the archived release:

> Mohammadrezaei, S. (2026). PoreWise: permeability tensors of pore-scale images with the
> lattice Boltzmann method on GPU and CPU (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.22896541

```bibtex
@software{porewise_0_1_0,
  author  = {Mohammadrezaei, Saleh},
  title   = {{PoreWise}: permeability tensors of pore-scale images with the lattice {B}oltzmann method on {GPU} and {CPU}},
  version = {0.1.0},
  year    = {2026},
  doi     = {10.5281/zenodo.22896541},
  url     = {https://github.com/SalehMohammadrezaei/PoreWise}
}
```

The concept DOI [10.5281/zenodo.22896540](https://doi.org/10.5281/zenodo.22896540) always
resolves to the latest release. GitHub also offers "Cite this repository" from
[`CITATION.cff`](CITATION.cff). A paper describing the software is in preparation.

## Contributing

Bug reports, questions and pull requests are welcome through the
[issue tracker](https://github.com/SalehMohammadrezaei/PoreWise/issues). Please run
`python -m pytest` before opening a pull request.

## Licence

PoreWise is released under the [MIT licence](LICENSE). Copyright (c) 2026 Saleh Mohammadrezaei.
