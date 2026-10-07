"""Build the V1 AI experimental dataset once into a fresh directory."""

import argparse
from pathlib import Path

from mcp_route_reliability.ai_dataset import build_ai_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "v1-ai-experimental",
        help="Fresh output directory (must not exist)",
    )
    output = parser.parse_args().output
    summary = build_ai_dataset(
        root / "data" / "v1-candidate",
        root / "data" / "v1-ai-inputs",
        output,
    )
    print(f"Created {output}: {summary.base_query_count} bases, {summary.variant_count} variants")


if __name__ == "__main__":
    main()
