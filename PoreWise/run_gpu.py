"""Edit the settings below, then run this file to process 3D images on an NVIDIA GPU.

Run inside tmux (terminal commands, not Python code):

1. Start a session from your repository folder:

       cd /path/to/PoreWise
       tmux new -s porewise_gpu

2. Inside that session, activate the environment and start the script:

       source .venv/bin/activate
       python -u run_gpu.py

   Alternatively, also save progress and errors to a uniquely named log:

       set -o pipefail
       python -u run_gpu.py 2>&1 | tee "gpu_run_$(date +%Y%m%d_%H%M%S).log"

   Choose one of these run commands. The -u option prints output immediately.
   Edit INPUT_PATTERN below for one image or a batch; both use the same command.
   TIFF volumes are loaded directly into memory; no .npy copies are created.

3. Detach while the simulation continues: press Ctrl-b, release, then press d.
   You can then close the terminal or disconnect SSH. The host must stay running.

4. Reconnect later (SSH into the same host first if running remotely):

       tmux ls
       tmux attach -t porewise_gpu

   To inspect an optional log from another terminal:

       tail -f gpu_run_<timestamp>.log

5. After completion, the shell stays open for reviewing output. To run again,
   edit settings, choose a fresh OUTPUT_FOLDER, and repeat the run command.
   Detaching preserves a running process; rerunning starts simulations from rest.
   Ctrl-c interrupts a run. Type exit after completion to close the tmux session.

tmux must be installed on the machine running the simulation. These commands
use Bash and the .venv environment described in README2.md.
"""
from pathlib import Path

from porewise import decode_mask, load_image, save_permeability_excel
from porewise.directional import save_directional_results


# Paths are relative to this script, so the repository can be moved or shared.
ROOT = Path(__file__).resolve().parent
INPUT_FOLDER = ROOT / "porewise/data/tif"
INPUT_PATTERN = "B51_bag*"  # Selects .tif/.tiff; use a filename for one image, or *.npy for legacy inputs.
OUTPUT_FOLDER = ROOT / "results_gpu"

# Replace these example values with the values for your images.
SOLID_VALUE = 1
PORE_VALUE = 0
VOXEL_SIZE_M = 1.2e-4
STEPS = 20000  # Updates per direction; each image runs x, y, and z from rest.
TAU = 0.6
PRECISION = "float64"
MIRROR = False
SAVE_EXCEL = False  # Requires the optional Excel dependency.


def main():
    files = sorted(path for path in INPUT_FOLDER.glob(INPUT_PATTERN)
                   if path.is_file() and path.suffix.lower() in ('.tif', '.tiff', '.npy'))
    rows = []
    for index, image_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {image_path.name}", flush=True)
        image = load_image(image_path)
        solid = decode_mask(image, solid_value=SOLID_VALUE, pore_value=PORE_VALUE)

        folder = OUTPUT_FOLDER / image_path.stem
        result = save_directional_results(
            folder,
            solid,
            voxel_size=VOXEL_SIZE_M,
            microct=image,
            backend="cuda-sparse",
            collision="trt",
            tau=TAU,
            precision=PRECISION,
            n_steps_max=STEPS,
            mirror=MIRROR,
            verbose=True,
            heartbeat=1000,
        )
        for axis in "xyz":
            print(f"k_{axis} = {result[f'k_{axis}_m2']:.6e} m²")
        print(f"Saved: {folder}", flush=True)

        if SAVE_EXCEL:
            rows.append((image_path, result))
            save_permeability_excel(OUTPUT_FOLDER / "permeability_results.xlsx", rows)
        del image, solid, result


if __name__ == "__main__":
    main()
