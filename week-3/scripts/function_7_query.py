from common import build_query


FUNCTION_ID = 7


def main() -> None:
    result = build_query(FUNCTION_ID)

    print(f"Function {FUNCTION_ID} query derivation")
    print("=" * 50)
    print(f"Data source: {result['data_source']}")
    print(f"Top-3 row indices by output: {result['top_idx']}")
    print(f"Top-3 output values: {result['top_y']}")
    print(f"Submitted query string: {result['query_str']}")
    print(f"Estimated output proxy (weighted top-3 y): {result['estimated_output']:.12g}")
    print(f"Closest existing-point distance: {result['min_distance']:.6f}")
    print("Explanation: blend strongest known points, then keep bounds valid.")
    print("Note: true output is unknown until evaluated in the evaluation system.")


if __name__ == "__main__":
    main()
