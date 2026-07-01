from pathlib import Path

from common import build_query


def main() -> None:
    canonical_path = Path(__file__).resolve().parents[1] / "submission_round3.txt"
    queries = [build_query(function_id)["query_str"] for function_id in range(1, 9)]
    payload = "\n".join(queries) + "\n"
    canonical_path.write_text(payload, encoding="ascii")

    print(f"Wrote {len(queries)} query lines to: {canonical_path}")
    print()
    print(canonical_path.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
