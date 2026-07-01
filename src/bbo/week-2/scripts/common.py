from pathlib import Path

import numpy as np


WEIGHTS = np.array([0.6, 0.3, 0.1])


def load_function_data(function_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the best available dataset for one function.

    Priority order:
    1) week-2/data/function_N with common cumulative naming
    2) week-2/data/function_N with initial naming
    3) fallback to week-1/initial_data/function_N
    """
    repo_root = Path(__file__).resolve().parents[2]
    week2_dir = repo_root / "week-2" / "data" / f"function_{function_id}"

    candidate_pairs = [
        ("inputs.npy", "outputs.npy"),
        ("round2_inputs.npy", "round2_outputs.npy"),
        ("cumulative_inputs.npy", "cumulative_outputs.npy"),
        ("initial_inputs.npy", "initial_outputs.npy"),
    ]

    for x_name, y_name in candidate_pairs:
        x_path = week2_dir / x_name
        y_path = week2_dir / y_name
        if x_path.exists() and y_path.exists():
            return np.load(x_path), np.load(y_path), f"week-2/data/function_{function_id} ({x_name}, {y_name})"

    fallback_dir = repo_root / "week-1" / "initial_data" / f"function_{function_id}"
    fallback_x = fallback_dir / "initial_inputs.npy"
    fallback_y = fallback_dir / "initial_outputs.npy"
    if fallback_x.exists() and fallback_y.exists():
        return np.load(fallback_x), np.load(fallback_y), "week-1/initial_data fallback"

    raise FileNotFoundError(
        f"Could not find input/output files for function_{function_id}. "
        "Expected files under week-2/data/function_N or week-1/initial_data/function_N."
    )


def build_query(function_id: int) -> dict:
    x, y, source = load_function_data(function_id)

    top_idx = np.argsort(y)[::-1][:3]
    top_x = x[top_idx]
    top_y = y[top_idx]

    query = np.clip((WEIGHTS[:, None] * top_x).sum(axis=0), 0.0, 0.999999)
    query_str = "-".join(f"{value:.6f}" for value in query)
    y_est = float(np.dot(WEIGHTS, top_y))
    min_distance = float(np.sqrt(((x - query) ** 2).sum(axis=1)).min())

    return {
        "function_id": function_id,
        "data_source": source,
        "query_str": query_str,
        "estimated_output": y_est,
        "top_idx": top_idx.tolist(),
        "top_y": [float(value) for value in top_y],
        "min_distance": min_distance,
    }
