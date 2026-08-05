from __future__ import annotations

import argparse
import json

from urban_heat_risk.pipelines.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heat anomaly models.")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    args = parser.parse_args()
    print(json.dumps(run_training(mode=args.mode), indent=2, default=str))


if __name__ == "__main__":
    main()
