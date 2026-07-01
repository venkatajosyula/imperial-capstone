from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


RNG_SEED = 42
MAX_VAL = 0.999999
N_CANDIDATES = 5000
N_TOP_STARTS = 10


@dataclass
class SurrogateBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    mlp: MLPRegressor


def load_function_data(function_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    repo_root = Path(__file__).resolve().parents[2]

    # Search priority: week-5 > week-4 > week-3 > week-2 fallback
    search_dirs = [
        repo_root / "week-5" / "data" / f"function_{function_id}",
        repo_root / "week-4" / "data" / f"function_{function_id}",
        repo_root / "week-3" / "data" / f"function_{function_id}",
        repo_root / "week-2" / "data" / f"function_{function_id}",
    ]
    candidate_pairs = [
        ("inputs.npy", "outputs.npy"),
        ("round2_inputs.npy", "round2_outputs.npy"),
        ("cumulative_inputs.npy", "cumulative_outputs.npy"),
        ("initial_inputs.npy", "initial_outputs.npy"),
    ]

    for search_dir in search_dirs:
        for x_name, y_name in candidate_pairs:
            x_path = search_dir / x_name
            y_path = search_dir / y_name
            if x_path.exists() and y_path.exists():
                week = search_dir.parents[1].name
                return (
                    np.load(x_path),
                    np.load(y_path),
                    f"{week}/data/function_{function_id} ({x_name}, {y_name})",
                )

    fallback_dir = repo_root / "week-1" / "initial_data" / f"function_{function_id}"
    fallback_x = fallback_dir / "initial_inputs.npy"
    fallback_y = fallback_dir / "initial_outputs.npy"
    if fallback_x.exists() and fallback_y.exists():
        return np.load(fallback_x), np.load(fallback_y), "week-1/initial_data fallback"

    raise FileNotFoundError(f"Could not find data for function_{function_id}")


def train_nn_surrogate(x: np.ndarray, y: np.ndarray, function_id: int) -> SurrogateBundle:
    d = x.shape[1]
    hidden = (max(16, 4 * d), max(8, 2 * d))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="tanh",
        alpha=1e-3,
        learning_rate_init=1e-2,
        max_iter=5000,
        early_stopping=False,
        n_iter_no_change=100,
        random_state=RNG_SEED + function_id,
    )
    mlp.fit(x_scaled, y_scaled)

    return SurrogateBundle(x_scaler=x_scaler, y_scaler=y_scaler, mlp=mlp)


def predict_surrogate(bundle: SurrogateBundle, x: np.ndarray) -> np.ndarray:
    x_scaled = bundle.x_scaler.transform(x)
    y_scaled = bundle.mlp.predict(x_scaled)
    return bundle.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def nearest_distances(candidates: np.ndarray, observed: np.ndarray) -> np.ndarray:
    diff = candidates[:, None, :] - observed[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def numerical_gradient(bundle: SurrogateBundle, x_point: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x_point)
    for j in range(x_point.shape[0]):
        plus = x_point.copy()
        minus = x_point.copy()
        plus[j] = min(MAX_VAL, plus[j] + eps)
        minus[j] = max(0.0, minus[j] - eps)
        y_plus = float(predict_surrogate(bundle, plus.reshape(1, -1))[0])
        y_minus = float(predict_surrogate(bundle, minus.reshape(1, -1))[0])
        step = plus[j] - minus[j]
        grad[j] = (y_plus - y_minus) / step if step > 0 else 0.0
    return grad


def refine_with_gradient_ascent(
    bundle: SurrogateBundle, start: np.ndarray, steps: int = 35, lr: float = 0.08
) -> np.ndarray:
    x_curr = start.copy()
    for _ in range(steps):
        grad = numerical_gradient(bundle, x_curr)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-12:
            break
        x_curr = np.clip(x_curr + lr * (grad / (grad_norm + 1e-12)), 0.0, MAX_VAL)
    return x_curr


def evaluate_nn_model(x: np.ndarray, y: np.ndarray, function_id: int) -> dict[str, float]:
    d = x.shape[1]
    mlp_hidden = (max(16, 4 * d), max(8, 2 * d))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    nn = MLPRegressor(
        hidden_layer_sizes=mlp_hidden,
        activation="tanh",
        alpha=1e-3,
        learning_rate_init=1e-2,
        max_iter=5000,
        early_stopping=False,
        n_iter_no_change=100,
        random_state=RNG_SEED + function_id,
    )

    n_splits = min(5, len(y))
    if n_splits < 3:
        nn.fit(x_scaled, y_scaled)
        pred_scaled = nn.predict(x_scaled)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        mse = float(np.mean((pred - y) ** 2))
        return {"nn_train_mse": mse, "nn_cv_mse": mse}

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    vals = cross_val_score(nn, x_scaled, y_scaled, scoring="neg_mean_squared_error", cv=cv)
    nn_cv_mse_scaled = float(-vals.mean())

    nn.fit(x_scaled, y_scaled)
    pred_scaled = nn.predict(x_scaled)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    nn_train_mse = float(np.mean((pred - y) ** 2))

    y_std = float(np.std(y))
    nn_cv_mse = nn_cv_mse_scaled * (y_std ** 2)
    return {"nn_train_mse": nn_train_mse, "nn_cv_mse": nn_cv_mse}


def propose_query(function_id: int) -> dict:
    x, y, source = load_function_data(function_id)
    bundle = train_nn_surrogate(x, y, function_id)

    rng = np.random.default_rng(RNG_SEED + 100 * function_id)
    candidates = rng.uniform(0.0, MAX_VAL, size=(N_CANDIDATES, x.shape[1]))

    preds = predict_surrogate(bundle, candidates)
    dists = nearest_distances(candidates, x)
    explore_weight = max(1e-6, float(0.1 * np.std(y)))
    candidate_scores = preds + explore_weight * dists

    start_idx = np.argsort(candidate_scores)[::-1][:N_TOP_STARTS]

    best_score = -np.inf
    best_point = candidates[start_idx[0]].copy()
    best_pred = float(preds[start_idx[0]])

    for idx in start_idx:
        start = candidates[idx]
        refined = refine_with_gradient_ascent(bundle, start)
        pred_val = float(predict_surrogate(bundle, refined.reshape(1, -1))[0])
        min_dist = float(nearest_distances(refined.reshape(1, -1), x)[0])
        score = pred_val + explore_weight * min_dist
        if score > best_score:
            best_score = score
            best_point = refined
            best_pred = pred_val

    min_dist = float(nearest_distances(best_point.reshape(1, -1), x)[0])
    if min_dist < 1e-5:
        top_candidates = np.argsort(candidate_scores)[::-1][:500]
        top_points = candidates[top_candidates]
        top_dists = nearest_distances(top_points, x)
        best_point = top_points[int(np.argmax(top_dists))]
        best_pred = float(predict_surrogate(bundle, best_point.reshape(1, -1))[0])
        min_dist = float(nearest_distances(best_point.reshape(1, -1), x)[0])

    grad = numerical_gradient(bundle, best_point)
    grad_abs = np.abs(grad)
    grad_order = np.argsort(grad_abs)[::-1]

    nn_metrics = evaluate_nn_model(x, y, function_id)
    query_str = "-".join(f"{v:.6f}" for v in best_point)

    return {
        "function_id": function_id,
        "data_source": source,
        "n_samples": int(len(y)),
        "dim": int(x.shape[1]),
        "query": best_point,
        "query_str": query_str,
        "nn_pred": best_pred,
        "distance_to_nearest": min_dist,
        "gradient": grad,
        "top_gradient_dims": [int(i + 1) for i in grad_order[: min(3, len(grad_order))]],
        "top_gradient_magnitudes": [float(grad_abs[i]) for i in grad_order[: min(3, len(grad_order))]],
        "nn_metrics": nn_metrics,
    }


def build_all_queries() -> list[dict]:
    return [propose_query(fid) for fid in range(1, 9)]


def write_outputs(results: list[dict]) -> tuple[Path, Path]:
    week6_root = Path(__file__).resolve().parents[1]
    submission_path = week6_root / "submission_round6_nn.txt"
    diagnostics_path = week6_root / "doc" / "nn_strategy_diagnostics.md"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    submission_lines = [r["query_str"] for r in results]
    submission_path.write_text("\n".join(submission_lines) + "\n", encoding="ascii")

    lines: list[str] = ["# Week 6 NN Strategy Diagnostics", ""]
    for r in results:
        lines.append(f"## Function {r['function_id']}")
        lines.append(f"- Data source: {r['data_source']}")
        lines.append(f"- Samples: {r['n_samples']} | Dimension: {r['dim']}")
        lines.append(f"- Suggested query: {r['query_str']}")
        lines.append(f"- NN predicted output at query: {r['nn_pred']:.10g}")
        lines.append(f"- Distance from nearest observed point: {r['distance_to_nearest']:.6f}")
        lines.append(
            "- Steepest gradient dimensions (1-based): "
            + ", ".join(str(v) for v in r["top_gradient_dims"])
        )
        lines.append(
            "- Gradient magnitudes on those dims: "
            + ", ".join(f"{v:.6g}" for v in r["top_gradient_magnitudes"])
        )
        mse = r["nn_metrics"]
        lines.append(
            f"- NN error diagnostics (lower is better): "
            f"train_mse={mse['nn_train_mse']:.6g}, cv_mse={mse['nn_cv_mse']:.6g}"
        )
        lines.append("")

    diagnostics_path.write_text("\n".join(lines), encoding="ascii")
    return submission_path, diagnostics_path


def main() -> None:
    results = build_all_queries()
    submission_path, diagnostics_path = write_outputs(results)

    print("Week 6 NN-guided query summary")
    print("=" * 60)
    for r in results:
        mse = r["nn_metrics"]
        print(f"F{r['function_id']}: {r['query_str']}")
        print(
            f"  pred={r['nn_pred']:.8g}  dist={r['distance_to_nearest']:.6f}  "
            f"top_grad_dims={r['top_gradient_dims']}  "
            f"nn_mse(train/cv)=({mse['nn_train_mse']:.6g}/{mse['nn_cv_mse']:.6g})"
        )
    print()
    print(f"Wrote submission: {submission_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
