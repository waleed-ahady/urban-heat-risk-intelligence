# Reference data

`district_profiles.csv` is an explicitly marked **demo seed** used to make the repository run offline and in CI. Coordinates are representative district centroids. Vulnerability fields are plausible starter values, not an official analytical release.

For a production or research use case, replace these values through an aggregation pipeline using:

- Berlin Open Data / Amt für Statistik Berlin-Brandenburg population and age data.
- Berlin Environmental Atlas population-density, green-volume, land-use, or impervious-surface WFS layers.

The application exposes `source_quality`, and the model card prohibits treating the demo profile as official evidence.

## Canonical profile builder

Map official downloads to the three CSV contracts in `templates/`, then run:

```bash
python scripts/build_district_profiles.py \
  --demographics <file> \
  --environment <file> \
  --coordinates <file> \
  --source-quality <versioned-source-label>
```

The raw source files should be retained outside the committed demo repository or added with explicit licence and provenance documentation.
