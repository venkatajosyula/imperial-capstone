"""
Parse week-13/inputs.txt and week-13/outputs.txt (cumulative 12-round data),
then append only the round-12 result to week-11 cumulative .npy files,
saving the result into week-12/data/function_X/.

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
    week13_dir = repo_root / "week-13"
    week12_data_root = repo_root / "week-12" / "data"

    raw_inputs = (week13_dir / "inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs_lines = (week13_dir / "outputs.txt").read_text(encoding="utf-8").splitlines()
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

    assert len(inputs_lines) == len(outputs_lines), (
        f"Mismatch: {len(inputs_lines)} input rows vs {len(outputs_lines)} output rows"
    )

    n_functions = 8

    # week-12/data already contains rounds 1-11 (21 points for 2D functions).
    # Only the last row (round 12) is new and must be appended.
    new_input = parse_inputs_line(inputs_lines[-1])
    new_output = parse_outputs_line(outputs_lines[-1])

    for fid in range(1, n_functions + 1):
        idx = fid - 1

        # Load existing cumulative data from week-12
        week12_dir = repo_root / "week-12" / "data" / f"function_{fid}"
        x_existing = np.load(week12_dir / "inputs.npy")
        y_existing = np.load(week12_dir / "outputs.npy")

        # Append round 12
        new_x = new_input[idx].reshape(1, -1)
        new_y = np.array([new_output[idx]])

        x_arr = np.vstack([x_existing, new_x])
        y_arr = np.concatenate([y_existing, new_y])

        out_dir = week12_data_root / f"function_{fid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "inputs.npy", x_arr)
        np.save(out_dir / "outputs.npy", y_arr)

        print(f"function_{fid}: shape={x_arr.shape}, y[-1]={y_arr[-1]:.6g}")


if __name__ == "__main__":
    main()
