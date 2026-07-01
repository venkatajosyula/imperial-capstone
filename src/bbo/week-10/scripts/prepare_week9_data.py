"""
Parse week-10/inputs.txt and week-10/outputs.txt (cumulative 9-round data),
then append only the round-9 result to week-8's cumulative .npy files,
saving the result into week-9/data/function_X/.

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
    week10_dir = repo_root / "week-10"
    week9_data_root = repo_root / "week-9" / "data"

    raw_inputs = (week10_dir / "inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs_lines = (week10_dir / "outputs.txt").read_text(encoding="utf-8").splitlines()
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

    # week-8/data already contains rounds 1-8 (18 points for 2D functions).
    # Only the last row (round 9) is new and must be appended.
    new_input = parse_inputs_line(inputs_lines[-1])
    new_output = parse_outputs_line(outputs_lines[-1])

    for fid in range(1, n_functions + 1):
        idx = fid - 1

        # Load existing cumulative data from week-8
        week8_dir = repo_root / "week-8" / "data" / f"function_{fid}"
        x_existing = np.load(week8_dir / "inputs.npy")
        y_existing = np.load(week8_dir / "outputs.npy")

        # Append round 9
        new_x = new_input[idx].reshape(1, -1)
        new_y = np.array([new_output[idx]])

        x_arr = np.vstack([x_existing, new_x])
        y_arr = np.concatenate([y_existing, new_y])

        out_dir = week9_data_root / f"function_{fid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "inputs.npy", x_arr)
        np.save(out_dir / "outputs.npy", y_arr)

        print(f"function_{fid}: shape={x_arr.shape}, y[-1]={y_arr[-1]:.6g}")


if __name__ == "__main__":
    main()
