from __future__ import annotations

import argparse
from pathlib import Path

from urban_heat_risk.data.profiles import read_and_build_profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical district vulnerability profile from public-data extracts."
    )
    parser.add_argument("--demographics", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--source-quality", required=True)
    parser.add_argument("--output", default="data/reference/district_profiles.csv")
    args = parser.parse_args()

    profiles = read_and_build_profiles(
        demographics_path=args.demographics,
        environment_path=args.environment,
        coordinates_path=args.coordinates,
        source_quality=args.source_quality,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output, index=False)
    print(f"Wrote {len(profiles)} district profiles to {output}")


if __name__ == "__main__":
    main()
