from __future__ import annotations

import argparse

import pandas as pd

from urban_heat_risk.data.profiles import validate_district_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate district vulnerability profiles.")
    parser.add_argument("path", nargs="?", default="data/reference/district_profiles.csv")
    args = parser.parse_args()
    errors = validate_district_profiles(pd.read_csv(args.path))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Profile validation passed: {args.path}")


if __name__ == "__main__":
    main()
