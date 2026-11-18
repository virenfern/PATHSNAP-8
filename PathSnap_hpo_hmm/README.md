# PathSnap Bayesian optimization and HMM/Viterbi package

This is a separate package for tuning the already selected **Hist Gradient Boosting** model and decoding temporally consistent road-arc routes. It uses the previous candidate-feature data; it does not recreate or fabricate training labels.

## Included files

- `data/train_candidate_features.csv.gz` — the same supplied 80-trip training candidate table used by the previous model package.
- `data/student_test_candidate_features.csv.gz` — separate 10-trip student-test candidate table, for evaluation only.
- `data/student_test_arcs/` — actual directed road-arc endpoint tables for the student-test trips, used to check road adjacency.
- `src/optimize_hgb.py` — group-aware Gaussian-process Bayesian optimization with Expected Improvement.
- `src/hmm_decode.py` — probability-emission HMM and log-space Viterbi decoder, with gap relaxation and disconnected-transition fallback.
- `src/build_report.py` — creates tables, plots, and report from actual run outputs.
- `results/optimization_trials.csv` and `optimization_summary.json` — six measured configurations and fold-level scores.
- `results/models/` — full-training untuned and tuned Hist Gradient Boosting estimators.
- `results/untuned/` and `results/tuned/` — per-trip decoder predictions and continuity results.
- `results/decoder_comparison.csv`, `decoder_comparison.png`, `optimization_trials.png`, and `hpo_hmm_report.md` — comparison deliverables.

## Run it (PowerShell)

From this package folder:

```powershell
python -m pip install -r requirements.txt
python src/optimize_hgb.py --train data/train_candidate_features.csv.gz --out results --trials 6 --folds 3
python src/hmm_decode.py --candidates data/student_test_candidate_features.csv.gz --arcs data/student_test_arcs --model results/models/untuned_hist_gradient_boosting.joblib --out results/untuned
python src/hmm_decode.py --candidates data/student_test_candidate_features.csv.gz --arcs data/student_test_arcs --model results/models/tuned_hist_gradient_boosting.joblib --out results/tuned
python src/build_report.py
```

The optimization objective is candidate-level Log Loss on GroupKFold validation folds grouped by trip. The student-test table is not used by the optimizer. The decoder compares independent per-observation top candidate against Viterbi output for both the untuned and tuned Hist Gradient Boosting models.

## Interpretation limits

The run evaluates six trial configurations over three grouped folds, so the optimization is compact rather than exhaustive. Transition weights and the 5-second gap threshold are explicit heuristics, not learned transition probabilities. Route continuity means consecutive represented arcs are identical or directly connected in the directed graph, for observation gaps no greater than five seconds. Candidate-generation misses remain outside decoder accuracy/continuity denominators; the existing student-test coverage is about 74.7%. This evaluation uses the supplied student-test partition, not an evaluator partition.
