# PathSnap: preprocessing + candidate generation

This ZIP combines the two work packages you asked for. It has two clearly named folders:

- `preprocessing/` — raw-data inventory/validation, GPS quality analysis, road-network connectivity results, and the two Folium maps.
- `candidate_generation/` — candidate-search and feature-generation code, the labeled training candidate table, candidate coverage, and feature schema.

The training table uses only the 80 supplied training trips. The 10 student-test trips are used only in the coverage summary. No evaluator-test trips were included in the supplied data.

## Reproduce

Extract `Student folder.zip` so `--data` points to the inner folder containing `train/` and `student_test/`. From this package root:

```powershell
python -m pip install -r requirements.txt
python preprocessing/src/pathsnap/audit.py --data "C:/path/to/data-root" --out preprocessing/results/reproduced/audit
python candidate_generation/src/pathsnap/preprocess.py --data "C:/path/to/data-root" --out candidate_generation/results/reproduced
```

To regenerate one of the interactive maps:

```powershell
python preprocessing/src/pathsnap/map_trip.py --data "C:/path/to/data-root" --partition train --trip 00000001 --out preprocessing/results/maps/train_00000001.html
```

## Candidate table notes

The current search uses a 60 m radius and keeps up to 12 road-segment candidates per GPS observation. Route labels are assigned with a monotone, progress-aware spatial heuristic because the input files do not provide a row-wise GPS-to-route mapping. Read `candidate_generation/results/feature_schema.json` before training: the table includes identifier and label-metadata columns that must not be included in model input. Split cross-validation by `trip_id`.

Coverage is 82.02% on training observations and 74.67% on student-test observations under this candidate/label procedure. The table contains only observations with a reference arc among their candidates; all misses are counted in `candidate_coverage.csv`. These are candidate coverage figures, not model accuracy.
