"""SVM-guided query generation for the BBO capstone (Week 3 / Module 14.1).

Strategy
--------
For each black-box function we have N evaluated points {x_i, y_i}.
We binary-label them by whether y_i is above the 50th-percentile
(label 1 = "high-performing region", label 0 = "low-performing").

A soft-margin RBF SVC is fitted on this labelled dataset.  A dense grid
of candidate inputs is scored by the SVM decision function; the candidate
with the highest score is the suggested next query.

If the SVM cannot distinguish the two classes (e.g. all examples are in
the same class), we fall back to the heuristic blend from common.py so
the pipeline never produces an invalid submission.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import build_query, load_function_data

# Total number of random candidate points to evaluate (constant regardless of dimension)
N_CANDIDATES = 10_000
# Percentile threshold for "high-performing" label
HIGH_PERCENTILE = 50
# SVM hyper-parameters (soft-margin C; RBF for non-linear boundaries)
SVM_C = 1.0
SVM_KERNEL = "rbf"
# Random seed for reproducibility
RNG_SEED = 42


def _make_candidates(n_dims: int, n: int = N_CANDIDATES, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return *n* random candidate points uniformly drawn from [0, 1)^n_dims."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    return rng.uniform(0.0, 0.999999, size=(n, n_dims))


def svm_build_query(function_id: int) -> dict:
    """Return an SVM-guided query for *function_id*.

    Returns a dict with the same keys as ``common.build_query`` plus:
    - ``method``       : "svm" or "heuristic_fallback"
    - ``svm_score``    : decision-function score of the chosen candidate
    - ``n_train``      : number of training points
    - ``n_high``       : number of high-performing training points
    """
    x, y, source = load_function_data(function_id)
    n_dims = x.shape[1]

    threshold = np.percentile(y, HIGH_PERCENTILE)
    labels = (y >= threshold).astype(int)

    n_high = int(labels.sum())
    n_low = int((labels == 0).sum())

    # Need at least one example from each class to fit the SVM
    if n_high == 0 or n_low == 0:
        fallback = build_query(function_id)
        fallback["method"] = "heuristic_fallback"
        fallback["svm_score"] = float("nan")
        fallback["n_train"] = len(y)
        fallback["n_high"] = n_high
        return fallback

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RNG_SEED)
    clf.fit(x_scaled, labels)

    candidates = _make_candidates(n_dims)
    candidates_scaled = scaler.transform(candidates)
    scores = clf.decision_function(candidates_scaled)

    best_idx = int(np.argmax(scores))
    best_candidate = candidates[best_idx]
    svm_score = float(scores[best_idx])

    query_str = "-".join(f"{v:.6f}" for v in best_candidate)

    # Estimated output: weighted average of top-3 observed outputs (reused for reporting)
    top_idx = np.argsort(y)[::-1][:3]
    top_y = y[top_idx]
    weights = np.array([0.6, 0.3, 0.1])
    y_est = float(np.dot(weights, top_y))

    min_distance = float(np.sqrt(((x - best_candidate) ** 2).sum(axis=1)).min())

    return {
        "function_id": function_id,
        "data_source": source,
        "query_str": query_str,
        "estimated_output": y_est,
        "top_idx": top_idx.tolist(),
        "top_y": [float(v) for v in top_y],
        "min_distance": min_distance,
        "method": "svm",
        "svm_score": svm_score,
        "n_train": len(y),
        "n_high": n_high,
    }


def run_all_svm_queries(verbose: bool = True) -> list[dict]:
    results = []
    for fid in range(1, 9):
        r = svm_build_query(fid)
        results.append(r)
        if verbose:
            method_tag = f"[{r['method']}]"
            svm_tag = (
                f"  svm_score={r['svm_score']:.4f}  n_train={r['n_train']}  n_high={r['n_high']}"
                if r["method"] == "svm"
                else "  (heuristic fallback)"
            )
            print(
                f"Function {fid:2d}  {method_tag:<22}  query={r['query_str']}"
                f"  y_est={r['estimated_output']:.4f}  dist={r['min_distance']:.4f}"
                f"{svm_tag}"
            )
    return results


if __name__ == "__main__":
    print("=== Week 3 SVM-guided query summary ===\n")
    results = run_all_svm_queries(verbose=True)

    lines = [r["query_str"] for r in results]
    out_path = (
        __file__.replace("svm_query.py", "submission_round3_svm.txt")
    )
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nSVM submission written to: {out_path}")

    # Side-by-side comparison with heuristic
    print("\n=== Comparison: SVM vs Heuristic ===")
    from common import build_query as heuristic_query  # noqa: E402 – local import for comparison
    for r in results:
        fid = r["function_id"]
        h = heuristic_query(fid)
        match = "SAME" if r["query_str"] == h["query_str"] else "DIFFERENT"
        print(f"  F{fid}: {match}")
        if match == "DIFFERENT":
            print(f"       SVM      : {r['query_str']}")
            print(f"       Heuristic: {h['query_str']}")
