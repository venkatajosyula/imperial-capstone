from pathlib import Path

import numpy as np


WEIGHTS = np.array([0.6, 0.3, 0.1])


def build_query(function_id: int) -> dict:
    base = Path(__file__).resolve().parents[1] / "initial_data" / f"function_{function_id}"
    x = np.load(base / "initial_inputs.npy")
    y = np.load(base / "initial_outputs.npy")

    top_idx = np.argsort(y)[::-1][:3]
    top_x = x[top_idx]
    top_y = y[top_idx]

    query = np.clip((WEIGHTS[:, None] * top_x).sum(axis=0), 0.0, 0.999999)
    query_str = "-".join(f"{value:.6f}" for value in query)
    y_est = float(np.dot(WEIGHTS, top_y))
    min_distance = float(np.sqrt(((x - query) ** 2).sum(axis=1)).min())

    return {
        "function_id": function_id,
        "query_str": query_str,
        "estimated_output": y_est,
        "top_idx": top_idx.tolist(),
        "top_y": [float(value) for value in top_y],
        "min_distance": min_distance,
    }


def main() -> None:
    results = [build_query(function_id) for function_id in range(1, 9)]

    print("Week 1 query summary")
    print("=" * 60)
    for result in results:
        print(f"Function {result['function_id']}")
        print(f"  Query: {result['query_str']}")
        print(f"  Top-3 indices: {result['top_idx']}")
        print(f"  Top-3 outputs: {result['top_y']}")
        print(f"  Estimated output proxy: {result['estimated_output']:.12g}")
        print(f"  Closest existing-point distance: {result['min_distance']:.6f}")
        print()

    print("Submission copy block")
    print("=" * 60)
    for result in results:
        print(result["query_str"])


if __name__ == "__main__":
    main()