"""
Parse week-5/inputs.txt and week-5/outputs.txt (cumulative 4-round data),
then save per-function .npy files into week-4/data/function_X/.

Each line in inputs.txt is one round; each element in the list is one function.
Same structure for outputs.txt.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_inputs_line(line: str) -> list[np.ndarray]:
    """Parse one line like [array([...]), array([...]), ...] into a list of arrays."""
    line = line.strip()
    arrays = []
    for match in re.finditer(r"array\(\[([\d\., \n-]+)\]\)", line):
        nums = [float(v) for v in re.findall(r"[\d.eE+-]+", match.group(1))]
        arrays.append(np.array(nums))
    return arrays


def parse_outputs_line(line: str) -> list[float]:
    """Parse one line like [np.float64(...), np.float64(...), ...] into floats."""
    return [float(v) for v in re.findall(r"np\.float64\(([-+eE\d.]+)\)", line)]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    week5_dir = repo_root / "week-5"
    week4_data_root = repo_root / "week-4" / "data"

    raw_inputs = (week5_dir / "inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs_lines = (week5_dir / "outputs.txt").read_text(encoding="utf-8").splitlines()
    outputs_lines = [l for l in outputs_lines if l.strip()]

    # Join continuation lines (those not starting with '[') to the previous line
    inputs_lines: list[str] = []
    for raw in raw_inputs:
        if not raw.strip():
            continue
        if raw.startswith("["):
            inputs_lines.append(raw)
        else:
            inputs_lines[-1] += " " + raw.strip()

    assert len(inputs_lines) == len(outputs_lines), "Mismatch between input and output rows"

    n_functions = 8

    # week-3/data already contains initial data + rounds 1 & 2 (12 points for 2D functions).
    # Rows 2 and 3 of inputs.txt are rounds 3 and 4 — both must be appended to reach 14.
    new_rounds_inputs = [parse_inputs_line(inputs_lines[i]) for i in [-2, -1]]
    new_rounds_outputs = [parse_outputs_line(outputs_lines[i]) for i in [-2, -1]]

    for fid in range(1, n_functions + 1):
        idx = fid - 1

        # Load existing cumulative data from week-3
        week3_dir = repo_root / "week-3" / "data" / f"function_{fid}"
        x_existing = np.load(week3_dir / "inputs.npy")
        y_existing = np.load(week3_dir / "outputs.npy")

        # Append rounds 3 and 4
        new_x = np.array([new_rounds_inputs[r][idx] for r in range(2)])
        new_y = np.array([new_rounds_outputs[r][idx] for r in range(2)])

        x_arr = np.vstack([x_existing, new_x])
        y_arr = np.concatenate([y_existing, new_y])

        out_dir = week4_data_root / f"function_{fid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "inputs.npy", x_arr)
        np.save(out_dir / "outputs.npy", y_arr)

        print(f"function_{fid}: shape={x_arr.shape}, y={y_arr}")

    print(f"\nSaved {n_functions} function datasets to week-4/data/")


if __name__ == "__main__":
    main()
