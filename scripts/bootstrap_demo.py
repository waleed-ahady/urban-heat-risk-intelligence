from __future__ import annotations

import json

from urban_heat_risk.pipelines.refresh import run_refresh
from urban_heat_risk.pipelines.train import run_training


def main() -> None:
    metrics = run_training(mode="demo")
    health = run_refresh(mode="demo")
    print(json.dumps({"training": metrics["evaluation"], "refresh": health}, indent=2))


if __name__ == "__main__":
    main()
