from pathlib import Path

import numpy as np


FUNCTION_ID = 2
WEIGHTS = np.array([0.6, 0.3, 0.1])


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "initial_data" / f"function_{FUNCTION_ID}"
    x = np.load(base / "initial_inputs.npy")
    y = np.load(base / "initial_outputs.npy")

    top_idx = np.argsort(y)[::-1][:3]
    top_x = x[top_idx]
    top_y = y[top_idx]

    query = np.clip((WEIGHTS[:, None] * top_x).sum(axis=0), 0.0, 0.999999)
    query_str = "-".join(f"{v:.6f}" for v in query)
    y_est = float(np.dot(WEIGHTS, top_y))

    distances = np.sqrt(((x - query) ** 2).sum(axis=1))

    print(f"Function {FUNCTION_ID} query derivation")
    print("=" * 50)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Top-3 row indices by output: {top_idx.tolist()}")
    print(f"Top-3 output values: {[float(v) for v in top_y]}")
    print(f"Weights used: {WEIGHTS.tolist()}")
    print(f"Submitted query string: {query_str}")
    print(f"Estimated output proxy (weighted top-3 y): {y_est:.12g}")
    print(f"Closest existing-point distance: {distances.min():.6f}")
    print("Explanation: blend strongest known points, then keep bounds valid.")
    print("Note: true output is unknown until evaluated in the evaluation system.")


if __name__ == "__main__":
    main()
