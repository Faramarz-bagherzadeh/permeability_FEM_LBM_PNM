from pathlib import Path
import argparse

import pandas as pd

# run with the results directory as an argument, or run in the results directory to combine all task_*/permeability_results.xlsx files
# python combine__results.py /albedo/scratch/user/fabagh001/lbm
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "porewise_permeability_results_complete.xlsx"


def main():
    parser = argparse.ArgumentParser(
        description="Combine PoreWise Excel results from task_* folders."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR,
        help="Folder containing task_* directories (defaults to this script's folder).",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {args.results_dir}")

    files = sorted(
        args.results_dir.glob("task_*/permeability_results.xlsx"),
        key=lambda path: int(path.parent.name.removeprefix("task_")),
    )
    if not files:
        raise FileNotFoundError(
            f"No task_*/permeability_results.xlsx files found in {args.results_dir}"
        )

    frames = [pd.read_excel(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"Found Excel files, but they contain no result rows: {args.results_dir}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "Image path" in combined.columns:
        combined = combined.sort_values("Image path", kind="stable", ignore_index=True)

    combined.to_excel(OUTPUT_FILE, index=False)
    print(f"Combined {len(combined)} rows from {len(files)} task files.")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()