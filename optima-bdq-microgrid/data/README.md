# Data

By default the project generates **synthetic** hourly data (`data.csv_path: null`).

To use real campus data, place a CSV here and set `data.csv_path` in `configs/default.yaml`:

| column    | unit  | notes                         |
|-----------|-------|-------------------------------|
| timestamp | ISO   | hourly                        |
| load      | MW    | campus demand                 |
| solar     | MW    | PV generation                 |
| wind      | MW    | wind generation               |
| price     | $/MWh | grid purchase price           |

The pipeline reindexes to a full hourly grid, interpolates gaps of up to
`max_gap_hours`, clips negatives, and drops every day that still has missing values.
