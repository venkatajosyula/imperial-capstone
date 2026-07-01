from common import build_query
from svm_query import svm_build_query


def main() -> None:
    heuristic_results = [build_query(function_id) for function_id in range(1, 9)]
    svm_results = [svm_build_query(function_id) for function_id in range(1, 9)]

    print("Week 3 query summary")
    print("=" * 60)
    for idx, result in enumerate(heuristic_results):
        svm_result = svm_results[idx]
        print(f"Function {result['function_id']}")
        print(f"  Data source: {result['data_source']}")
        print(f"  Heuristic query: {result['query_str']}")
        print(f"  SVM query: {svm_result['query_str']}")
        print(f"  SVM method: {svm_result.get('method', 'svm')}")
        print(f"  SVM score: {svm_result['svm_score']:.6f}")
        print(f"  Top-3 indices: {result['top_idx']}")
        print(f"  Top-3 outputs: {result['top_y']}")
        print(f"  Estimated output proxy: {result['estimated_output']:.12g}")
        print(f"  Closest existing-point distance: {result['min_distance']:.6f}")
        print()

    print("Submission copy block (heuristic)")
    print("=" * 60)
    for result in heuristic_results:
        print(result["query_str"])

    print("\nSubmission copy block (svm)")
    print("=" * 60)
    for result in svm_results:
        print(result["query_str"])


if __name__ == "__main__":
    main()
