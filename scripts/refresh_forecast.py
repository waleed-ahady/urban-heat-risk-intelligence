from __future__ import annotations

import argparse
import json

from urban_heat_risk.pipelines.refresh import run_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh forecast risk outputs.")
    parser.add_argument("--mode", choices=["demo", "live"], default="live")
    args = parser.parse_args()
    print(json.dumps(run_refresh(mode=args.mode), indent=2, default=str))


if __name__ == "__main__":
    main()
