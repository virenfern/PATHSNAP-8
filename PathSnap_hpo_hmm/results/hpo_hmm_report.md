# Bayesian optimization and HMM route decoding

## Bayesian optimization

The selected model family remains Hist Gradient Boosting. Six hyperparameter configurations were evaluated. Three seeded random initial trials trained a Gaussian-process surrogate; the remaining three configurations were proposed by Expected Improvement. The objective was mean candidate-level Log Loss across three GroupKFold folds grouped by trip ID. The held-out student-test set was not read by the optimizer.

| Run | CV candidate Log Loss |
|---|---:|
| Untuned baseline | 0.247615 |
| Best tuned trial (2) | 0.245513 |

The selected trial reduced mean CV Log Loss by 0.002102 (0.85% relative) compared with the baseline under the same three folds. This is a small measured improvement, not a guarantee of better performance on new trips.

Best parameters: `{"l2_regularization": 3.26392401342566, "learning_rate": 0.16438810179505506, "max_iter": 38, "max_leaf_nodes": 29, "min_samples_leaf": 85}`

All individual trial settings and fold scores are in `optimization_trials.csv` and `optimization_summary.json`. The tuned and untuned estimators refit on all supplied training trips are saved under `models/`.

## HMM and Viterbi

Each candidate road arc is an HMM state. The classifier probability that a candidate is correct supplies the emission score; candidate probabilities are normalized within each GPS observation. Transition weights favor staying on the same arc (weight 2), moving to a directly connected directed arc (weight 1), and penalize disconnected transitions. The weights are normalized per previous state to form transition probabilities. These are explicit heuristic weights, not learned or calibrated transition probabilities.

Viterbi uses log probabilities to find the most likely entire state sequence. For gaps longer than 5 seconds between represented GPS observations, the disconnected-transition floor is relaxed from 0.00001 to 0.05. If the candidate sets have no same-arc or directly connected pair, the decoder can still continue through the penalized fallback instead of crashing. The measured student-test run had 44 relaxed-gap steps and zero steps with no possible linked candidate pair.

## Held-out student-test decoder comparison

| Model | Decoder | Reference-arc accuracy | Route continuity (gap ≤ 5 s) | Continuous / eligible transitions |
|---|---|---:|---:|---:|
| Untuned Hist Gradient Boosting | Classifier Only | 51.55% | 86.89% | 16,601 / 19,106 |
| Untuned Hist Gradient Boosting | Hmm Viterbi | 53.56% | 99.47% | 19,005 / 19,106 |
| Tuned Hist Gradient Boosting | Classifier Only | 51.06% | 86.15% | 16,460 / 19,106 |
| Tuned Hist Gradient Boosting | Hmm Viterbi | 53.80% | 99.45% | 19,000 / 19,106 |

Reference-arc accuracy is evaluated on all 16,300 student-test observations with an assigned reference arc. If the correct arc is absent from that observation’s candidate set, it is necessarily counted wrong; this occurred for 1,994 labeled observations. The remaining 2,860 observations have no assigned reference label and are excluded from accuracy. Route continuity is the share of consecutive represented GPS observation pairs (gap at most 5 seconds) whose selected arcs are identical or directly connected in the directed road graph. Candidate coverage remains a separate preprocessing limit: approximately 74.7% of all student-test observations had their reference arc among generated candidates.

For the tuned model, Viterbi changed student-test reference-arc accuracy from 51.06% to 53.80% and short-gap continuity from 86.15% to 99.45%. This is a held-out comparison, not a new model-selection step. The run covers the supplied student-test partition (10 trips), not an evaluator partition.

## Limitations

- Bayesian search was intentionally compact (six trials, three grouped folds); it is a real optimization run but not an exhaustive search.
- CV tuning objective is candidate-level Log Loss, while decoder accuracy and continuity are sequence-level metrics. They capture different behavior.
- The transition weights and 5-second gap threshold are explicit heuristic settings. The search did not optimize them.
- Route continuity uses direct directed-arc adjacency or the same arc; it does not search multi-arc paths between GPS samples.
- Student-test observations without a reference label are excluded from reference-arc accuracy; labeled observations whose correct arc is absent from candidates are counted wrong. Report candidate coverage separately.
