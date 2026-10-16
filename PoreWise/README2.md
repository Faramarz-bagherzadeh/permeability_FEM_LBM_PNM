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
pip install '.[excel]'   # openpyxl, for optional Excel permeability summaries
```

## GPU workflow using a Python script

If you receive a copy of this repository and want to run it entirely through a
Python script, use [`run_gpu.py`](run_gpu.py). It loads your segmented 3D TIFF
volumes directly into memory, runs the three flow directions on an NVIDIA GPU, and saves each image's
results. No notebook or simulation command-line arguments are needed.

### 1. Set up the Python environment once

Open a terminal in the received repository folder. For the repository's CUDA 12
GPU dependency, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[gpu]'
```

An NVIDIA GPU, a working NVIDIA driver and a compatible CuPy/CUDA installation
are required. The local CUDA 13 setup recorded later in this document instead
uses `python -m pip install . cupy-cuda13x`; choose the CuPy package matching your
environment. Install only one CuPy distribution in that environment.

The setup commands install dependencies; all simulation work is controlled by
the Python script. In an editor or IDE, select this virtual environment as the
Python interpreter.

### 2. Put your images in the input folder

The script reads complete 3D TIFF volumes with axes `(z, y, x)` directly
into memory, one image at a time. By default it searches `porewise/data/tif/`
for `B51_bag*` files with `.tif` or `.tiff` extensions (case insensitive).
Place your independent segmented volumes there, or edit the input folder and
pattern below. Separate 2D slice files are not automatically stacked.

No conversion script or intermediate `.npy` files are needed. After all three
directions and result exports finish, the image arrays are released before the
next input is loaded. Original dtypes and labels are preserved; no `uint8` cast
is applied. `tifffile` is installed with PoreWise. Existing `.npy` inputs remain
supported by changing the input folder and pattern.

### 3. Edit the settings in run_gpu.py

```python
INPUT_FOLDER = ROOT / "porewise/data/tif"
INPUT_PATTERN = "B51_bag*"
OUTPUT_FOLDER = ROOT / "results_gpu"
SOLID_VALUE = 1
PORE_VALUE = 0
VOXEL_SIZE_M = 1.2e-4
STEPS = 20000
TAU = 0.6
PRECISION = "float64"
MIRROR = False
SAVE_EXCEL = False
```

These labels and voxel size are B51 examples, not values to assume for other
datasets. Set them from your own image metadata. For pore=0 and solid=255,
change `SOLID_VALUE` to `255`. Set `INPUT_PATTERN = "rock.tif"` for one image
or `"*.tif*"` for all TIFF images in the input folder. All selected images use
the same voxel size and simulation settings.

Paths using `ROOT` are relative to the script's location, so another person
does not need your absolute workspace path. Absolute paths also work, for
example `INPUT_FOLDER = Path("/path/to/my/images")`.

### 4. Run the Python script

With the environment active, run from the repository folder:

```bash
python run_gpu.py
```

You can also run the file from your editor using the same interpreter. The
script explicitly selects `cuda-sparse` and never switches the simulation to a
CPU solver. It performs no GPU preflight or input validation. Image loading,
input preparation and file export still use the CPU. An empty input match runs
no simulations. Python, file I/O and CUDA/CuPy can still raise their own errors.

Progress is printed every 1,000 updates for the current direction. `STEPS`
is the number of updates **per direction**, and each direction starts from rest.
The workflow does not check numerical convergence or solution quality; finishing
the requested updates does not establish convergence. For a longer run, change
`STEPS`, choose a new output folder and run the script again.

### 5. Read the saved results

```text
results_gpu/
└── <image_filename_without_extension>/
    ├── permeability.json
    └── fields.vti
```

The terminal and JSON report `k_x_m2`, `k_y_m2` and `k_z_m2` in m².
`fields.vti` contains `MicroCT` and the three vector fields `Velocity_X`,
`Velocity_Y` and `Velocity_Z` as float32 arrays. Velocities are in lattice units,
not m/s. Open this file in ParaView, or use the Python reader under
[Read the outputs](#read-the-outputs).

For an optional shared Excel summary, install
`python -m pip install '.[excel]'` and set `SAVE_EXCEL = True`. The script writes
`results_gpu/permeability_results.xlsx` after each completed image, storing only
the scalar results between images. Each new run replaces that workbook.
Reusing an output folder also overwrites matching simulation files and retains
unrelated old files; choose a fresh folder for separate runs.

## Quick start

From the command line, for a segmented 3D image stored as a TIFF volume:

```bash
python -m porewise rock.tif --backend cuda-sparse --collision trt \
    --solid-value 255 --pore-value 0 --dx 5e-6 --steps 20000 --output results/rock
```

`--dx` is the voxel size in metres. Use `--backend numba-sparse` for CPU execution.
Every image runs three independent simulations from rest, driven along x, y and z.
Each simulation performs exactly `--steps` updates. There are no convergence, mass,
density, finite-state, Mach, Reynolds or permeability acceptance checks in this workflow.
Supply a segmented 3D image, voxel size and supported solver settings; these are
used without explicit validation. No diagnostic, provenance or error-report files are written.

Without `--excel`, the output folder contains only:

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

### Save permeability results to Excel

Install the optional Excel dependency in the same environment:

```bash
python -m pip install '.[excel]'
```

For a fresh CPU installation, install both extras together with
`python -m pip install '.[cpu,excel]'`. Run the following commands from the
repository root with that environment active.

For one image:

```bash
python -m porewise rock.tif \
    --backend numba-sparse --collision trt --tau 0.6 \
    --solid-value 255 --pore-value 0 --dx 5e-6 --steps 20000 \
    --output results/rock --excel results/rock/permeability_results.xlsx
```

For multiple B51 images, save all images in one workbook, with one row per image:

```bash
python -m porewise porewise/data/tif/B51_bag*.tif* \
    --backend numba-sparse --collision trt --tau 0.6 \
    --solid-value 1 --pore-value 0 --dx 1.2e-4 --steps 20000 \
    --output results --excel results/permeability_results.xlsx
```

The `Permeability` sheet contains **Image filename**, **Image path**, **kx (m²)**,
**ky (m²)** and **kz (m²)**. The paths distinguish images with identical filenames.
Each row associates all three permeabilities with the image that produced them.
The following values illustrate the layout; they are not measured results:

| Image filename | Image path | kx (m²) | ky (m²) | kz (m²) |
|---|---|---:|---:|---:|
| image_1.tif | data/image_1.tif | 1.200000E-12 | 8.000000E-13 | 5.000000E-13 |
| image_2.tiff | data/image_2.tiff | 2.400000E-12 | 1.100000E-12 | 9.000000E-13 |

Permeabilities are stored as numeric cells formatted in scientific notation.
Image paths are recorded as supplied, so relative paths remain relative.
The workbook is updated after each completed image, so completed rows remain
available if a later simulation fails. A new invocation replaces the workbook;
it does not append previous runs. JSON and VTI exports remain available. The
use an `.xlsx` workbook path; missing parent directories are created.
For a shared batch workbook, pass all images in one command as above. Separate
commands using the same workbook path replace one another's summaries.

For Python scripts, collect `(image_path, result)` pairs and export them:

```python
from pathlib import Path
from porewise import (
    compute_directional_permeability,
    decode_mask,
    load_image,
    save_permeability_excel,
)

rows = []
for image_path in sorted(Path("porewise/data/tif").glob("B51_bag*.tif*")):
    image = load_image(image_path)
    solid = decode_mask(image, solid_value=1, pore_value=0)
    result = compute_directional_permeability(
        solid, voxel_size=1.2e-4, backend="numba-sparse",
        collision="trt", tau=0.6, n_steps_max=20000, verbose=True,
    )
    # Keep only the three scalars, rather than retaining every image's fields.
    rows.append((image_path, {f"k_{axis}_m2": result[f"k_{axis}_m2"] for axis in "xyz"}))
    del result, image, solid
    save_permeability_excel("results/permeability_results.xlsx", rows)
```

Replace the labels, input paths and voxel size with those of your own images.
To also save JSON and VTI from Python, use
`porewise.directional.save_directional_results` in place of
`compute_directional_permeability`; it returns the three scalar permeabilities
and streams the velocity fields to disk.

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

## Local B51 workflow

The current workflow produces x, y and z permeability in m² and three vector
velocity fields for every 3D image. Each direction starts independently from rest
and uses a fixed number of updates. The default is 20,000 updates per direction.
Numerical convergence, mass conservation, finite-state, density, Mach and Reynolds
checks and permeability acceptance gating are disabled in this workflow.
Explicit image, parameter, backend and file validation is also removed.

The local case collection contains 23 binary TIFF volumes and matching NumPy
arrays: 5 from bag02, 10 from bag31 and 8 from bag70. The local paths and environment
notes below were recorded on October 13, 2026. The archived simulations later in
this document were run on October 10–11, 2026 with the previous tensor workflow;
these documentation updates do not rerun or convert those results.

### Workflow

1. Activate the virtual environment and install the required GPU or CPU dependencies.
2. Place the binary TIFF volumes in `porewise/data/tif/`.
3. Select the TIFF inputs directly; no conversion or intermediate files are needed.
4. Run each image for the chosen fixed number of updates along x, y and z.
5. Read its three permeabilities from `permeability.json` and its three vector
   velocity fields together with the image from `fields.vti`.
6. Optionally add `--excel <summary.xlsx>` to collect the image filenames and
   directional permeabilities in an Excel workbook. See
   [Save permeability results to Excel](#save-permeability-results-to-excel).

### Local environment setup

Run from the repository root, which in this workspace is
`/home/drill/Projects/FB/Codes/PoreWise/PoreWise`.
The following successful setup commands are reconstructed from shell history:

```bash
cd /home/drill/Projects/FB/Codes/PoreWise/PoreWise
python3 -m venv .venv
source .venv/bin/activate
pip install .
python -m pip install --upgrade pip
python -m pip install cupy-cuda13x
pip install '.[cpu]'
python -m pip install tifffile

nvidia-smi
nvcc --version
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('GPU:', cp.cuda.runtime.getDeviceCount()); print('Test:', (cp.arange(5)**2).get())"
```

The inspected environment contains Python 3.12.3, PoreWise 0.1.0,
`cupy-cuda13x` 14.2.0, `tifffile` 2026.9.20, and Numba 0.68.0.
The general installation instructions above use the repository's CUDA 12 GPU extra;
this local run instead installed `cupy-cuda13x` directly.

### Data and output layout

Use `porewise/results_directional/` for new runs to keep the two-file outputs
separate from existing tensor results in `porewise/results/`.

```text
PoreWise/
├── README.md
├── README2.md
├── porewise/
│   ├── data/
│   │   └── tif/                  # independent binary TIFF volumes; loaded directly
│   ├── results_directional/     # current per-image output
│   │   └── <sample>/
│   │       ├── permeability.json
│   │       └── fields.vti         # MicroCT + Velocity_X/Y/Z, all float32
│   └── results/                 # archived tensor outputs and ground_truth.xlsx
└── reports/permeability_ground_truth_log10/  # archived comparison artifacts
```

By default, only the two files listed under `<sample>/` are produced by the current
solver workflow. With `--excel`, an additional summary workbook is written at the
specified path; it can be shared by all samples in one batch invocation.
No density arrays, convergence CSV, diagnostic results, provenance or
error-report files are exported. Existing files in a reused output directory are
not deleted; use a fresh directory for the two-file layout.

### Read and inspect TIFF data directly

Run from the repository root:

```python
import numpy as np
from porewise import load_image

data = load_image(
    "porewise/data/tif/B51_bag70_hxfb101ms_block_0_69.792_.tif"
)
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Min:", data.min())
print("Max:", data.max())
print("Unique values:", np.unique(data))
del data
```

The B51 inputs have shape `(400, 400, 400)`. Loading creates only an in-memory
NumPy array; it writes no input copies to disk. The original TIFFs are retained.
Existing `.npy` copies are not automatically deleted. Simulation output files
still require disk space, independently of input storage.

### Current simulation settings

The following command examples use the local B51 labels and voxel size recorded
in the archived results. Each of the three directional simulations runs for the
requested fixed number of updates, without convergence-based early termination
or a wall-clock timeout. All-solid images return zero permeability and velocity
without simulation updates.

| Setting | Value used in the B51 examples |
|---|---|
| Method | 3D D3Q19 lattice Boltzmann method |
| Backend | `cuda-sparse`; use `numba-sparse` for CPU |
| Collision model | TRT |
| Relaxation time | `tau = 0.6` |
| TRT magic parameter | `3/16 = 0.1875` |
| Lattice kinematic viscosity | `(tau - 0.5) / 3 = 0.0333333333` |
| Solver storage and computation | `float64` |
| Exported image, velocity arrays and permeability precision | `float32` |
| Input shape | `400 × 400 × 400` |
| Array axes | `(z, y, x)` |
| Velocity components | `(ux, uy, uz)` |
| Pore / solid labels | `0 / 1` |
| Voxel size | `1.2e-4 m`, or `120 µm` |
| Force magnitude | `1e-6` lattice units, one axis per load |
| Boundary conditions | Periodic in all array axes |
| Mirror option | Disabled |
| Updates | 20,000 per direction in the initial example; 60,000 in the longer example |
| Numerical quality and convergence checks | Disabled |
| Permeability acceptance gate | Disabled |
| Velocity-field export | Always enabled for all three directions |

For driving direction `j`, the reported permeability is
`k_j_m2 = nu * mean(u_j) / F_j * dx**2`, where the mean includes the full sample
volume and solid voxels have zero velocity. For these examples, `dx**2 = 1.44e-8 m²`.
The JSON keys are `k_x_m2`, `k_y_m2` and `k_z_m2`; no full tensor, eigenvalues or
principal directions are exported.

One `fields.vti` stores the image and all three complete velocity vectors:

| Array | Meaning | Export dtype |
|---|---|---|
| `MicroCT` | Original input image labels | `float32` |
| `Velocity_X` | `(ux, uy, uz)` for x-driven flow | `float32` |
| `Velocity_Y` | `(ux, uy, uz)` for y-driven flow | `float32` |
| `Velocity_Z` | `(ux, uy, uz)` for z-driven flow | `float32` |

These are VTK point-data arrays, with dimensions `(nx, ny, nz)`, voxel spacing
`dx` in metres and origin at the first voxel centre `(0, 0, 0)`. The B51 image
retains its original labels (0=pore, 1=solid), converted to float32. The binary
VTK XML export is written sequentially by slab and load, without a VTK dependency.
The three JSON permeability numbers are rounded to float32 precision before
writing; JSON itself has no dtype.

For the unmirrored B51 images, each component has shape `(400, 400, 400)`.
Velocities are in lattice units (`dx/dt`); voxel size alone does not supply the
physical time scale needed to convert them to m/s. Adding `--mirror` doubles
field dimensions along every array axis and computes sealed-sample directional
permeabilities on that mirrored domain.

### Run one B51 image

Run from the repository root with the virtual environment active:

```bash
source .venv/bin/activate
python -m porewise porewise/data/tif/B51_bag70_hxfb101ms_block_0_69.792_.tif \
    --backend cuda-sparse \
    --collision trt \
    --tau 0.6 \
    --solid-value 1 \
    --pore-value 0 \
    --dx 1.2e-4 \
    --steps 20000 \
    --output porewise/results_directional/B51_bag70_hxfb101ms_block_0_69.792_
```

The current CLI always runs all three directions and exports all three fields.
The former `--tensor`, `--fields`, `--direction`, `--dimension`, `--tol`,
`--check-every` and `--timeout` options are no longer part of this CLI.

### Run all B51 images

This Bash loop runs from the repository root and gives each image a directory
named after its filename stem. It does not use validity flags or skip cases based
on old results. Every selected image starts a new fixed-step run.

```bash
source .venv/bin/activate
shopt -s nullglob
files=(porewise/data/tif/B51_bag*.tif*)

for file in "${files[@]}"; do
    name=$(basename "$file")
    name=${name%.*}
    python -m porewise "$file" \
        --backend cuda-sparse \
        --collision trt \
        --tau 0.6 \
        --solid-value 1 \
        --pore-value 0 \
        --dx 1.2e-4 \
        --steps 20000 \
        --output "porewise/results_directional/$name"
done
```

Alternatively, pass multiple images to one invocation:

```bash
python -m porewise porewise/data/tif/B51_bag*.tif* \
    --backend cuda-sparse --collision trt --tau 0.6 \
    --solid-value 1 --pore-value 0 --dx 1.2e-4 --steps 20000 \
    --output porewise/results_directional_batch \
    --excel porewise/results_directional_batch/permeability_results.xlsx
```

For multiple inputs, the CLI creates indexed subdirectories such as
`001_<sample>/` and `002_<sample>/`. All inputs in that invocation share voxel
size and solver settings.
This example requires `python -m pip install '.[excel]'` and also writes
`porewise/results_directional_batch/permeability_results.xlsx`, with one row per
input image. Omit `--excel` if a workbook is not needed.

### Run a longer simulation and display progress

To recompute a selected sample for 60,000 updates per direction, use a new output
folder and change `--steps`. This starts all three directions from rest; it does
not resume a checkpoint or depend on an accepted/rejected status.

```bash
python -m porewise porewise/data/tif/B51_bag70_hxfb101ms_block_0_69.792_.tif \
    --backend cuda-sparse --collision trt --tau 0.6 \
    --solid-value 1 --pore-value 0 --dx 1.2e-4 \
    --steps 60000 --verbose --heartbeat 1000 \
    --output porewise/results_directional_60000/B51_bag70_hxfb101ms_block_0_69.792_
```

`--verbose` prints update count, percent complete, elapsed time and ETA at the
`--heartbeat` interval. The ETA refers to the remaining fixed updates for the
current directional simulation. Progress goes to the terminal and adds no files
to the output folder.

The existing [`rerun_failed_bags.sh`](rerun_failed_bags.sh) and historical commands
in [`behrad_points.txt`](behrad_points.txt) use the previous tensor layout and CLI
options. Use the commands above for the current workflow; those scripts have not
been updated as part of this documentation change.

### Read the outputs

Open `fields.vti` in ParaView, select `MicroCT` for the image and choose
`Velocity_X`, `Velocity_Y` or `Velocity_Z` for the desired driving direction.
The file stores all four arrays together on the same grid.

To load the arrays in Python, install the optional VTK reader:

```bash
python -m pip install vtk
```

```python
import json
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy

folder = Path(
    "porewise/results_directional/B51_bag70_hxfb101ms_block_0_69.792_"
)
permeabilities = json.loads((folder / "permeability.json").read_text())
print("kx (m²):", permeabilities["k_x_m2"])
print("ky (m²):", permeabilities["k_y_m2"])
print("kz (m²):", permeabilities["k_z_m2"])

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(str(folder / "fields.vti"))
reader.Update()
image = reader.GetOutput()
nx, ny, nz = image.GetDimensions()
data = image.GetPointData()

microct = vtk_to_numpy(data.GetArray("MicroCT")).reshape(nz, ny, nx)
velocities = {
    axis: vtk_to_numpy(data.GetArray("Velocity_" + axis)).reshape(nz, ny, nx, 3)
    for axis in "XYZ"
}
x_flow = velocities["X"]  # dtype float32; final axis contains (ux, uy, uz)
vector_at_voxel = x_flow[0, 0, 0]
# Component-first layout, matching the supplied reference function:
kx = x_flow.transpose(3, 0, 1, 2)  # shape (3, nz, ny, nx)
```

### Running in tmux

From the repository root:

```bash
tmux new -s porewise_directional
source .venv/bin/activate
# Run a single-image or batch command above in this session.
```

Detach with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux ls
tmux attach -t porewise_directional
```

With `--verbose`, progress appears inside the session. `nvidia-smi` can be run in
another terminal to view GPU usage.

## Archived B51 tensor results and comparison

The following inventory and metrics describe saved outputs from the **previous
workflow**, run on October 10–11, 2026. That workflow exported `result.json` and
`load_0/fields.npz`, `load_1/fields.npz`, `load_2/fields.npz`, with numerical checks
and convergence-based acceptance. Its accepted/rejected statuses and step caps
are historical records and do not describe the new fixed-step output contract.
These artifacts have not been replaced or regenerated by the current update.

### Complete case inventory and saved outcomes

There are **23 cases**: 5 from bag02, 10 from bag31, and 8 from bag70.
Each listed `.npy` input has a matching `.tif` input with the same filename stem.
The table reports the archived tensor JSON, with update counts and termination
reasons ordered **X / Y / Z**. Porosity is the pore fraction expressed as a percentage.

| Input filename | Porosity | Saved step cap | Updates X / Y / Z | Termination X / Y / Z | Saved tensor status |
|---|---|---:|---|---|---|
| `B51_bag02_hxfb101ms_block_0_1.012_.npy` | 58.542% | 20,000 | 4700 / 4000 / 4200 | converged / converged / converged | Accepted |
| `B51_bag02_hxfb101ms_block_0_1.036_.npy` | 58.652% | 20,000 | 4200 / 3500 / 3600 | converged / converged / converged | Accepted |
| `B51_bag02_hxfb101ms_block_0_1.06_.npy` | 59.365% | 20,000 | 4200 / 3600 / 3800 | converged / converged / converged | Accepted |
| `B51_bag02_hxfb101ms_block_0_1.35_.npy` | 63.033% | 20,000 | 17900 / 18400 / 15200 | converged / converged / converged | Accepted |
| `B51_bag02_hxfb101ms_block_0_1.47_.npy` | 66.316% | 20,000 | 3100 / 3100 / 3200 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.034_.npy` | 40.295% | 20,000 | 8900 / 8200 / 9900 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.057_.npy` | 41.377% | 20,000 | 9100 / 7800 / 6600 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.33_.npy` | 41.836% | 20,000 | 8000 / 7300 / 7400 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.353_.npy` | 41.567% | 20,000 | 7900 / 7300 / 7300 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.376_.npy` | 41.396% | 20,000 | 8300 / 7500 / 6800 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.399_.npy` | 41.059% | 20,000 | 9000 / 7800 / 7100 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.421_.npy` | 40.808% | 20,000 | 8700 / 8100 / 6000 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.444_.npy` | 40.803% | 20,000 | 8600 / 7800 / 6900 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.49_.npy` | 41.429% | 20,000 | 9000 / 7900 / 7000 | converged / converged / converged | Accepted |
| `B51_bag31_hxfb101ms_block_0_30.74_.npy` | 41.564% | 20,000 | 8300 / 7500 / 6800 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.769_.npy` | 25.786% | 60,000 | 36400 / 36300 / 34600 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.792_.npy` | 26.087% | 60,000 | 41700 / 36400 / 32900 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.815_.npy` | 25.791% | 60,000 | 45200 / 37900 / 39900 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.838_.npy` | 24.948% | 60,000 | 48100 / 42500 / 40300 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.861_.npy` | 24.682% | 60,000 | 53400 / 50000 / 44700 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.884_.npy` | 25.306% | 60,000 | 49000 / 41900 / 43600 | converged / converged / converged | Accepted |
| `B51_bag70_hxfb101ms_block_0_69.907_.npy` | 25.349% | 20,000 | 20000 / 20000 / 20000 | max_steps / max_steps / max_steps | Rejected full tensor |
| `B51_bag70_hxfb101ms_block_0_69.952_.npy` | 32.418% | 60,000 | 58400 / 60000 / 48700 | converged / max_steps / converged | Rejected full tensor |

**21 full tensors are accepted.** The 15 bag02/bag31 cases have accepted results
from the 20,000-update configuration. Six bag70 cases have accepted results from
the 60,000-update configuration.

Two cases require special interpretation:

- **`69.907`**: the saved tensor JSON is from the initial run, where X, Y, and Z
  each reached 20,000 updates without acceptance. A later 60,000-update retry log
  records X and Y converging and then ends partway through Z. The retry did not
  replace the saved tensor JSON with a completed accepted result. An empty
  `error.json` exists; the available records do not establish the cause of interruption.
- **`69.952`**: the saved retry converged in X at 58,400 updates and Z at 48,700;
  Y reached 60,000 updates. X and Z are accepted directional results, but the full
  tensor is rejected. Its saved accepted values are `K_xx = 2.1686887863e-7 m²`
  and `K_zz = 3.3483707253e-10 m²`; `K_yy` is null.

Saved fields are present for all three directions in all 23 sample directories.
Their presence alone does not establish convergence or tensor acceptance.

### Archived ground-truth comparison

Source: [`compare.py`](reports/permeability_ground_truth_log10/compare.py).
It reads `porewise/results/ground_truth.xlsx`, sheet `Sheet1`, matches each sample
by exact filename stem. The current script assumes matching voxel resolution and
tensor component order without explicit validation.
X, Y, and Z correspond to `K_m2[0][0]`, `K_m2[1][1]`, and `K_m2[2][2]`.

The script compares `log10(k / 1 m²)`, computes R² without fitting a regression,
and reports MAE and RMSE in dex. It excludes invalid directional solves and
nonpositive or nonfinite values, with no arbitrary outlier removal or imputation.
The two cohorts are accepted complete tensors and all accepted directional pairs.

| Coverage | Count |
|---|---:|
| Reference samples | 287 |
| Matched computed samples | 23 |
| Reference samples without computed results | 264 |
| Accepted complete tensors | 21 |
| Accepted directional pairs: X / Y / Z | 22 / 21 / 22 |

Saved metrics:

| Cohort | Direction | Samples | R² | MAE (dex) | RMSE (dex) |
|---|---|---:|---:|---:|---:|
| Complete tensors | X | 21 | 0.649739 | 0.347375 | 0.350051 |
| Complete tensors | Y | 21 | 0.886213 | 0.205552 | 0.209367 |
| Complete tensors | Z | 21 | 0.978273 | 0.084627 | 0.093061 |
| Valid directions | X | 22 | -0.219855 | 0.455865 | 0.675841 |
| Valid directions | Y | 21 | 0.886213 | 0.205552 | 0.209367 |
| Valid directions | Z | 22 | 0.165093 | 0.239291 | 0.749022 |

The valid-direction cohort additionally includes the accepted X and Z values of
`69.952`; these materially affect the comparison metrics.

The comparison script reads the archived `result.json` tensor layout and does not
read the new `permeability.json` files. To reproduce the archived comparison from
the repository root, install its additional
analysis dependencies and run:

```bash
python -m pip install pandas openpyxl matplotlib
python reports/permeability_ground_truth_log10/compare.py
```

These are reproduction instructions. Those three analysis packages were not present
in the inspected project virtual environment, although saved comparison artifacts exist.
The script rewrites its comparison CSV, Excel, JSON, and plot outputs.

- [Excel comparison](reports/permeability_ground_truth_log10/permeability_comparison.xlsx)
- [Saved metrics](reports/permeability_ground_truth_log10/metrics.csv)
- [Methodology and coverage](reports/permeability_ground_truth_log10/methodology.json)
- [Excluded directional solves](reports/permeability_ground_truth_log10/excluded_directions.csv)
- [Log10 comparison plot](reports/permeability_ground_truth_log10/log10_comparison.png)

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
