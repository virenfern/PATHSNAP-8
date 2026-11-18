# PathSnap model training and selection

This archive is the ML-training work package. It is separate from the earlier preprocessing and feature-engineering ZIP. It includes the exact candidate tables used for the reported run so the results can be reproduced without re-running preprocessing.

## What is included

- `src/train_models.py` — five-model trip-grouped cross-validation, student-test evaluation, timing, confusion matrices, full-data refits and saved artifacts.
- `src/render_model_plots.py` — reproducible comparison plots using a headless backend.
- `data/train_candidate_features.csv.gz` — candidate rows from the 80 training trips.
- `data/student_test_candidate_features.csv.gz` — separate student-test candidate rows; never used to fit models.
- `results/model_comparison.csv` — CV and student-test Accuracy, Macro-F1, Precision, Recall, Log Loss, Top-1, and timing.
- `results/cv_fold_metrics.csv` — metrics for each of the five trip-grouped validation folds.
- `results/confusion_matrices/` and `results/cv_confusion_matrices.png` — candidate-level OOF confusion matrices.
- `results/model_comparison.png` and `results/model_loss_and_efficiency.png` — measured comparison plots.
- `results/models/` — all five full-training model artifacts plus `selected_final_model.joblib`.
- `results/experiment_log.json`, `results/selected_model.txt`, and `results/model_selection_report.md` — configuration, selection rule and report text.

## Models trained

Logistic Regression, Decision Tree, Random Forest, Extra Trees, and Hist Gradient Boosting. Five-fold `GroupKFold` splits by `trip_id`, so candidates from a trip do not cross between training and validation.

## Selection and results

The selected artifact is **Hist Gradient Boosting**. To reflect the five CV metrics in the comparison, selection gives one equal vote to each: candidate Accuracy, Macro-F1, Precision, Recall, and Log Loss (lower is better). Hist Gradient Boosting leads three metrics (Accuracy, Precision, Log Loss); Random Forest leads Macro-F1; Logistic Regression leads Recall. Ties are broken by lower CV Log Loss. Student-test metrics are held-out evaluation and did not determine selection.

The underlying measured values remain in `results/model_comparison.csv`; `cv_metric_wins` records the count. Route continuity, shown in the supplied comparison template, has not been computed, so comparison against that full template remains incomplete. Candidate coverage is a separate limit: about 82.0% on training observations and 74.7% on student-test. These are not evaluator-set results; the supplied data lacked the evaluator partition.

## Reproduce

From this folder:

```powershell
python -m pip install -r requirements.txt
python src/train_models.py --train data/train_candidate_features.csv.gz --test data/student_test_candidate_features.csv.gz --out results --folds 5
python src/render_model_plots.py --results results
```

Feature columns are defined in the companion preprocessing package's `feature_schema.json`. Identifiers and reference-label metadata must not be passed into model inputs. The split must remain grouped by trip.
