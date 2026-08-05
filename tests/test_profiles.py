from __future__ import annotations

import pandas as pd

from urban_heat_risk.data.profiles import build_district_profiles


def test_build_district_profiles_computes_exposure_and_age_share() -> None:
    demographics = pd.DataFrame(
        {
            "district": ["A", "B"],
            "population": [100_000, 50_000],
            "area_km2": [20.0, 25.0],
            "population_age_65_plus": [20_000, 5_000],
        }
    )
    environment = pd.DataFrame(
        {
            "district": ["A", "B"],
            "green_space_pct": [20.0, 50.0],
            "impervious_surface_pct": [70.0, 30.0],
        }
    )
    coordinates = pd.DataFrame(
        {"district": ["A", "B"], "latitude": [52.5, 52.6], "longitude": [13.4, 13.3]}
    )
    result = build_district_profiles(
        demographics, environment, coordinates, source_quality="official_test_release"
    )
    district_a = result[result["district"] == "A"].iloc[0]
    assert district_a["population_density_per_km2"] == 5000
    assert district_a["share_age_65_plus_pct"] == 20
