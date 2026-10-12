"""Combine per-task CHFEM permeability workbooks into one results file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_DIR = Path("/albedo/scratch/user/fabagh001/chfem_results")
OUTPUT_DIR = Path("/albedo/scratch/user/fabagh001/chfed_excel_files")


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob("task_*/permeability_results.xlsx"))
    results = [pd.read_excel(path) for path in files]
    results = [frame for frame in results if not frame.empty]
    if not results:
        raise FileNotFoundError(f"No task result rows found in {INPUT_DIR}")

    combined = pd.concat(results, ignore_index=True, sort=False)
    if "file" in combined.columns:
        combined = combined.sort_values("file", kind="stable", ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "permeability_results_complete.xlsx"
    combined.to_excel(output_file, index=False)
    print(f"Saved {len(combined)} rows to {output_file}")
