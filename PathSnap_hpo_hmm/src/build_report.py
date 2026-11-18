"""Assemble comparison tables, plots, and report from the measured run outputs."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
def main():
    opt=json.loads((R/'optimization_summary.json').read_text(encoding='utf-8'))
    runs={key:json.loads((R/key/'decoder_evaluation.json').read_text(encoding='utf-8')) for key in ('untuned','tuned')}
    # Trial plot includes baseline as a reference and all recorded optimization trials.
    trials=opt['trials']; xs=[0]+[t['trial'] for t in trials]; ys=[opt['baseline_mean_cv_log_loss']]+[t['mean_cv_log_loss'] for t in trials]
    fig,ax=plt.subplots(figsize=(8.5,4.5));ax.plot(xs,ys,'o-',color='#2457a7',label='Grouped-CV Log Loss');ax.axhline(opt['baseline_mean_cv_log_loss'],color='#555',ls='--',label='Untuned baseline');best=min(trials,key=lambda t:t['mean_cv_log_loss']);ax.scatter([best['trial']],[best['mean_cv_log_loss']],s=100,color='#1c9c54',zorder=4,label=f"Best trial {best['trial']}");ax.set_xlabel('Trial (0 = untuned baseline)');ax.set_ylabel('Mean 3-fold CV candidate Log Loss (lower is better)');ax.set_title('Bayesian optimization of Hist Gradient Boosting');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(R/'optimization_trials.png',dpi=180);plt.close(fig)
    rows=[]
    for variant in ('untuned','tuned'):
        for item in runs[variant]['overall']:
            rows.append({'model_variant':variant,'decoder':item['decoder'],'student_test_reference_arc_accuracy':item['reference_arc_accuracy'],'short_gap_route_continuity':item['route_continuity_short_gap'],'continuous_transitions':item['continuous_transitions'],'eligible_short_gap_transitions':item['eligible_short_gap_transitions'],'student_test_labeled_observations':item['labeled_observations'],'relaxed_gap_steps':runs[variant]['total_relaxed_gap_steps'],'no_link_candidate_steps':runs[variant]['total_steps_with_no_link_between_candidate_sets']})
    with (R/'decoder_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels=[f"{r['model_variant']}\n{r['decoder'].replace('_',' ')}" for r in rows];x=range(len(rows));w=.38
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    axs[0].bar(x,[r['student_test_reference_arc_accuracy'] for r in rows],color=['#86a9d4','#2d6fbd']*2);axs[0].set_title('Student-test reference-arc accuracy');axs[0].set_ylabel('Accuracy');axs[0].set_ylim(0,1)
    axs[1].bar(x,[r['short_gap_route_continuity'] for r in rows],color=['#9cc9a8','#29964a']*2);axs[1].set_title('Short-gap route continuity');axs[1].set_ylabel('Continuous selected transitions / eligible transitions');axs[1].set_ylim(0,1)
    for ax in axs:ax.set_xticks(list(x),labels,rotation=15,ha='right');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Independent classifier choices vs. HMM/Viterbi');fig.tight_layout();fig.savefig(R/'decoder_comparison.png',dpi=180);plt.close(fig)
    def pct(x):return f'{100*x:.2f}%'
    base_cv=opt['baseline_mean_cv_log_loss'];best_cv=opt['selected_mean_cv_log_loss']; improvement=(base_cv-best_cv)/base_cv*100
    lines=['# Bayesian optimization and HMM route decoding','',
      '## Bayesian optimization','',
      'The selected model family remains Hist Gradient Boosting. Six hyperparameter configurations were evaluated. Three seeded random initial trials trained a Gaussian-process surrogate; the remaining three configurations were proposed by Expected Improvement. The objective was mean candidate-level Log Loss across three GroupKFold folds grouped by trip ID. The held-out student-test set was not read by the optimizer.', '',
      '| Run | CV candidate Log Loss |','|---|---:|',
      f"| Untuned baseline | {base_cv:.6f} |",f"| Best tuned trial ({opt['selected_trial']}) | {best_cv:.6f} |",'',
      f"The selected trial reduced mean CV Log Loss by {base_cv-best_cv:.6f} ({improvement:.2f}% relative) compared with the baseline under the same three folds. This is a small measured improvement, not a guarantee of better performance on new trips.",'',
      'Best parameters: `'+json.dumps(opt['selected_parameters'],sort_keys=True)+'`','',
      'All individual trial settings and fold scores are in `optimization_trials.csv` and `optimization_summary.json`. The tuned and untuned estimators refit on all supplied training trips are saved under `models/`.','',
      '## HMM and Viterbi','',
      'Each candidate road arc is an HMM state. The classifier probability that a candidate is correct supplies the emission score; candidate probabilities are normalized within each GPS observation. Transition weights favor staying on the same arc (weight 2), moving to a directly connected directed arc (weight 1), and penalize disconnected transitions. The weights are normalized per previous state to form transition probabilities. These are explicit heuristic weights, not learned or calibrated transition probabilities.', '',
      'Viterbi uses log probabilities to find the most likely entire state sequence. For gaps longer than 5 seconds between represented GPS observations, the disconnected-transition floor is relaxed from 0.00001 to 0.05. If the candidate sets have no same-arc or directly connected pair, the decoder can still continue through the penalized fallback instead of crashing. The measured student-test run had 44 relaxed-gap steps and zero steps with no possible linked candidate pair.', '',
      '## Held-out student-test decoder comparison','',
      '| Model | Decoder | Reference-arc accuracy | Route continuity (gap ≤ 5 s) | Continuous / eligible transitions |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model_variant'].title()} Hist Gradient Boosting | {r['decoder'].replace('_',' ').title()} | {pct(r['student_test_reference_arc_accuracy'])} | {pct(r['short_gap_route_continuity'])} | {r['continuous_transitions']:,} / {r['eligible_short_gap_transitions']:,} |")
    lines += ['',
      'Reference-arc accuracy is evaluated on all 16,300 student-test observations with an assigned reference arc. If the correct arc is absent from that observation’s candidate set, it is necessarily counted wrong; this occurred for 1,994 labeled observations. The remaining 2,860 observations have no assigned reference label and are excluded from accuracy. Route continuity is the share of consecutive represented GPS observation pairs (gap at most 5 seconds) whose selected arcs are identical or directly connected in the directed road graph. Candidate coverage remains a separate preprocessing limit: approximately 74.7% of all student-test observations had their reference arc among generated candidates.', '',
      'For the tuned model, Viterbi changed student-test reference-arc accuracy from '+pct(rows[2]['student_test_reference_arc_accuracy'])+' to '+pct(rows[3]['student_test_reference_arc_accuracy'])+' and short-gap continuity from '+pct(rows[2]['short_gap_route_continuity'])+' to '+pct(rows[3]['short_gap_route_continuity'])+'. This is a held-out comparison, not a new model-selection step. The run covers the supplied student-test partition (10 trips), not an evaluator partition.', '',
      '## Limitations','',
      '- Bayesian search was intentionally compact (six trials, three grouped folds); it is a real optimization run but not an exhaustive search.',
      '- CV tuning objective is candidate-level Log Loss, while decoder accuracy and continuity are sequence-level metrics. They capture different behavior.',
      '- The transition weights and 5-second gap threshold are explicit heuristic settings. The search did not optimize them.',
      '- Route continuity uses direct directed-arc adjacency or the same arc; it does not search multi-arc paths between GPS samples.',
      '- Student-test observations without a reference label are excluded from reference-arc accuracy; labeled observations whose correct arc is absent from candidates are counted wrong. Report candidate coverage separately.']
    (R/'hpo_hmm_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Saved report and plots. Best CV log loss {best_cv:.6f} vs baseline {base_cv:.6f}.")
if __name__=='__main__':main()
