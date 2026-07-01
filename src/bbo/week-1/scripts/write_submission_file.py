from pathlib import Path

import numpy as np


WEIGHTS = np.array([0.6, 0.3, 0.1])


def build_query(function_id: int) -> str:
    base = Path(__file__).resolve().parents[1] / "initial_data" / f"function_{function_id}"
    x = np.load(base / "initial_inputs.npy")
    y = np.load(base / "initial_outputs.npy")

    top_idx = np.argsort(y)[::-1][:3]
    top_x = x[top_idx]
    query = np.clip((WEIGHTS[:, None] * top_x).sum(axis=0), 0.0, 0.999999)
    return "-".join(f"{value:.6f}" for value in query)


def main() -> None:
    canonical_path = Path(__file__).resolve().parents[1] / "submission_round1.txt"
    queries = [build_query(function_id) for function_id in range(1, 9)]
    payload = "\n".join(queries) + "\n"
    canonical_path.write_text(payload, encoding="ascii")

    print(f"Wrote {len(queries)} query lines to: {canonical_path}")
    print()
    print(canonical_path.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()