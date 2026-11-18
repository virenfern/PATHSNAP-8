# Model experimentation and selection

Five models were trained and compared using 5-fold GroupKFold, grouping rows by trip ID. The student-test partition was never used for fitting. Model selection gave one equal vote to each of the five displayed grouped-CV metrics: candidate Accuracy, Macro-F1, Precision, Recall, and Log Loss (lower is better for Log Loss). The model leading on the most metrics wins; ties are broken by lower CV Log Loss. Student-test numbers are held-out evaluation and did not select the winner.

**Selected model: Hist Gradient Boosting.**

| Model | CV Top-1 (covered) | CV candidate Macro-F1 | CV Log Loss | Student-test Top-1 (labeled) | Student-test candidate Macro-F1 | Mean CV fit (s) | Test inference (ms/candidate) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5429 | 0.6899 | 0.4444 | 0.5088 | 0.6647 | 0.98 | 0.0001 |
| Extra Trees | 0.5428 | 0.7039 | 0.4189 | 0.5064 | 0.6902 | 21.77 | 0.0012 |
| Hist Gradient Boosting | 0.5364 | 0.7087 | 0.2501 | 0.5155 | 0.7473 | 3.72 | 0.0007 |
| Random Forest | 0.5357 | 0.7177 | 0.3698 | 0.5169 | 0.7071 | 27.54 | 0.0013 |
| Decision Tree | 0.5211 | 0.6983 | 0.4371 | 0.5019 | 0.6847 | 16.24 | 0.0001 |

## Interpretation and limits

The training candidate table contains 1,308,050 candidate rows across 80 training trips. The chosen model is the measured grouped-CV leader under the stated rule; this is not a claim of evaluator performance. Candidate misses cap end-to-end accuracy: recorded candidate coverage was about 82.0% for training observations and 74.7% for student-test observations. Student-test Top-1 is computed only where a reference arc label could be assigned; candidate misses among those labeled observations count as wrong. No evaluator partition was supplied.

Candidate-level Precision, Recall, Accuracy, Macro-F1, Log Loss and confusion matrices are saved alongside observation-level Top-1. All result graphs and model artifacts are generated from these actual runs.


## Metric-win summary

| Model | CV metrics led | Wins |
|---|---|---:|
| Hist Gradient Boosting | Accuracy, Precision, Log Loss | 3/5 |
| Random Forest | Macro-F1 | 1/5 |
| Logistic Regression | Recall | 1/5 |
| Extra Trees | None | 0/5 |
| Decision Tree | None | 0/5 |

## Selection limitation

Route continuity, requested in the supplied comparison template, has not been measured in these existing experiments. This selection is therefore provisional relative to that full comparison. The selection rule is based only on grouped cross-validation; the student-test metrics are shown for evaluation and were not used to select the model.
