from common import build_query


def main() -> None:
    results = [build_query(function_id) for function_id in range(1, 9)]

    print("Week 2 query summary")
    print("=" * 60)
    for result in results:
        print(f"Function {result['function_id']}")
        print(f"  Data source: {result['data_source']}")
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
