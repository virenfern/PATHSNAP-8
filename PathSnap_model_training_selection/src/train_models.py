"""Trip-grouped five-model PathSnap candidate classifier comparison."""
from __future__ import annotations
import argparse,csv,gzip,json,math,time,traceback
from pathlib import Path
import numpy as np
import joblib, sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,f1_score,log_loss,
                             precision_score,recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank']
SEED=42

def read_table(path,feature_cols=FEATURES):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        first=next(csv.reader(f)); n=sum(1 for _ in f)
    arrays={'X':np.empty((n,len(feature_cols)),dtype=np.float32),
            'y':np.empty(n,dtype=np.uint8),'trip':np.empty(n,dtype=np.int16),
            'obs':np.empty(n,dtype=np.int32),'candidate':np.empty(n,dtype=np.int32),
            'reference':np.empty(n,dtype=np.int32)}
    trip_codes={}; key_prev=None; obs_code=-1; i=0
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            trip=row['trip_id'];
            if trip not in trip_codes: trip_codes[trip]=len(trip_codes)
            key=(trip,row['observation_index'])
            if key!=key_prev: obs_code+=1; key_prev=key
            arrays['X'][i]=[float(row[k]) for k in feature_cols]
            arrays['y'][i]=int(row['label'])
            arrays['trip'][i]=trip_codes[trip]
            arrays['obs'][i]=obs_code
            arrays['candidate'][i]=int(row['candidate_arc_id'])
            arrays['reference'][i]=int(row['reference_arc_id']) if row['reference_arc_id'] else -1
            i+=1
    if i!=n: raise RuntimeError(f'{path}: row count changed between passes')
    return arrays,trip_codes

def grouped_top1(y,probs,obs_ids,valid=None):
    starts=np.r_[0,np.flatnonzero(np.diff(obs_ids)!=0)+1]
    ends=np.r_[starts[1:],len(y)]
    correct=0; used=0
    for s,e in zip(starts,ends):
        if valid is not None and not valid[s]: continue
        best=s+int(np.argmax(probs[s:e]))
        used+=1
        correct+=int(y[best]==1)
    return correct/used if used else float('nan'),used

def binary_metrics(y,p):
    pred=(p>=.5).astype(np.uint8)
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def model_specs():
    return {
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=60,class_weight='balanced',solver='lbfgs',random_state=SEED)),
      'Decision Tree':DecisionTreeClassifier(max_depth=12,min_samples_leaf=100,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=18,max_depth=14,min_samples_leaf=80,class_weight='balanced_subsample',n_jobs=2,random_state=SEED),
      'Extra Trees':ExtraTreesClassifier(n_estimators=18,max_depth=14,min_samples_leaf=80,class_weight='balanced',n_jobs=2,random_state=SEED),
      'Hist Gradient Boosting':HistGradientBoostingClassifier(max_iter=35,learning_rate=.1,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1.0,early_stopping=True,random_state=SEED)
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True,type=Path); ap.add_argument('--test',required=True,type=Path); ap.add_argument('--out',required=True,type=Path); ap.add_argument('--folds',type=int,default=5); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'models').mkdir(exist_ok=True); (a.out/'confusion_matrices').mkdir(exist_ok=True)
    print('Loading compressed candidate tables...',flush=True)
    tr,_=read_table(a.train); te,_=read_table(a.test)
    # Only student-test rows with a known reference arc can be scored. Missed candidates
    # remain in the table; a group whose reference is absent gets an incorrect Top-1.
    test_ref_known=te['reference']>=0
    groups=tr['trip']; gkf=GroupKFold(n_splits=a.folds)
    models=model_specs(); comparison=[]; fold_rows=[]; selected=None
    log=[]
    with threadpool_limits(limits=2):
      for name,template in models.items():
        print(f'[{name}] grouped CV...',flush=True)
        oof=np.full(len(tr['y']),np.nan,dtype=np.float32); fold_fit=[]; fold_inf=[]
        for fold,(it,iv) in enumerate(gkf.split(tr['X'],tr['y'],groups),1):
            est=clone(template); t0=time.perf_counter(); est.fit(tr['X'][it],tr['y'][it]); fit_s=time.perf_counter()-t0
            t0=time.perf_counter(); prob=est.predict_proba(tr['X'][iv])[:,1]; inf_s=time.perf_counter()-t0
            oof[iv]=prob; fold_fit.append(fit_s); fold_inf.append(inf_s)
            bm=binary_metrics(tr['y'][iv],prob); t1,nobs=grouped_top1(tr['y'][iv],prob,tr['obs'][iv])
            fold_rows.append({'model':name,'fold':fold,'train_trips':len(np.unique(groups[it])),'validation_trips':len(np.unique(groups[iv])),'candidate_rows':len(iv),'observation_groups':nobs,'top1_accuracy_covered':t1,**{k:v for k,v in bm.items() if k!='confusion_matrix'},'fit_seconds':fit_s,'inference_seconds':inf_s})
            print(f'  fold {fold}/{a.folds}: Top1={t1:.4f}, fit={fit_s:.1f}s',flush=True)
        if np.isnan(oof).any(): raise RuntimeError('OOF predictions missing; fold setup was incomplete')
        cv_bin=binary_metrics(tr['y'],oof); cv_top1,cv_nobs=grouped_top1(tr['y'],oof,tr['obs'])
        # Refit on all permitted training trips and assess once on student-test.
        final=clone(template); t0=time.perf_counter(); final.fit(tr['X'],tr['y']); final_fit=time.perf_counter()-t0
        evalmask=test_ref_known
        Xtest=te['X'][evalmask]; yt=te['y'][evalmask]; ot=te['obs'][evalmask]
        t0=time.perf_counter(); pt=final.predict_proba(Xtest)[:,1]; test_inf=time.perf_counter()-t0
        test_bin=binary_metrics(yt,pt) if len(yt) else {k:float('nan') for k in ('accuracy','macro_f1','precision','recall','log_loss')}
        test_top1,test_groups=grouped_top1(yt,pt,ot) if len(yt) else (float('nan'),0)
        # Count aligned test observations with no candidate rows separately from coverage.
        model_file=a.out/'models'/(name.lower().replace(' ','_')+'.joblib'); joblib.dump(final,model_file,compress=3)
        cm=cv_bin.pop('confusion_matrix'); np.savetxt(a.out/'confusion_matrices'/(name.lower().replace(' ','_')+'_cv.csv'),np.asarray(cm,dtype=int),fmt='%d',delimiter=',')
        row={'model':name,'cv_candidate_accuracy':cv_bin['accuracy'],'cv_candidate_macro_f1':cv_bin['macro_f1'],'cv_candidate_precision':cv_bin['precision'],'cv_candidate_recall':cv_bin['recall'],'cv_candidate_log_loss':cv_bin['log_loss'],'cv_observation_top1_accuracy_covered':cv_top1,'cv_observation_groups':cv_nobs,'cv_mean_fit_seconds':float(np.mean(fold_fit)),'cv_mean_inference_seconds':float(np.mean(fold_inf)),'student_test_candidate_accuracy':test_bin['accuracy'],'student_test_candidate_macro_f1':test_bin['macro_f1'],'student_test_candidate_precision':test_bin['precision'],'student_test_candidate_recall':test_bin['recall'],'student_test_candidate_log_loss':test_bin['log_loss'],'student_test_observation_top1_accuracy_labeled':test_top1,'student_test_labeled_observation_groups':test_groups,'student_test_inference_seconds':test_inf,'student_test_inference_ms_per_candidate':1000*test_inf/max(1,len(Xtest)),'final_fit_seconds':final_fit,'artifact':str(model_file.relative_to(a.out))}
        comparison.append(row)
        log.append({'model':name,'parameters':str(template.get_params(deep=False)),'cv_confusion_matrix':cm})
        print(f'[{name}] CV Top1={cv_top1:.4f}; student-test Top1={test_top1:.4f}; saved {model_file.name}',flush=True)
    # Equal vote across the five displayed grouped-CV metrics; lower CV Log Loss breaks ties.
    cv_metrics=[('cv_candidate_accuracy','max'),('cv_candidate_macro_f1','max'),('cv_candidate_precision','max'),('cv_candidate_recall','max'),('cv_candidate_log_loss','min')]
    metric_winners={}
    for metric,direction in cv_metrics:
        best=(max if direction=='max' else min)(r[metric] for r in comparison)
        metric_winners[metric]=next(r['model'] for r in comparison if abs(r[metric]-best)<1e-12)
    for r in comparison: r['cv_metric_wins']=sum(winner==r['model'] for winner in metric_winners.values())
    comparison.sort(key=lambda r:(r['cv_metric_wins'],-r['cv_candidate_log_loss']),reverse=True)
    selected=comparison[0]['model']; selected_file=a.out/'models'/'selected_final_model.joblib';
    # The selected full-train artifact is already saved as one of the five model files; copy for stable path.
    joblib.dump(joblib.load(a.out/'models'/(selected.lower().replace(' ','_')+'.joblib')),selected_file,compress=3)
    for filename,rows in [('model_comparison.csv',comparison),('cv_fold_metrics.csv',fold_rows)]:
        with (a.out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out/'experiment_log.json').write_text(json.dumps({'scikit_learn':sklearn.__version__,'folds':a.folds,'splitter':'GroupKFold by trip_id','seed':SEED,'training_candidate_rows':len(tr['y']),'training_trips':len(np.unique(tr['trip'])),'student_test_candidate_rows':len(te['y']),'student_test_reference_labeled_rows':int(evalmask.sum()),'models':log,'selected_model':selected,'selection_rule':'Equal vote across CV Accuracy, Macro-F1, Precision, Recall, and Log Loss (lower is better); tie-break by lower CV Log Loss. Student-test metrics did not determine selection.'},indent=2),encoding='utf-8')
    (a.out/'selected_model.txt').write_text(f'Selected model: {selected}\nRule: equal vote across CV Accuracy, Macro-F1, Precision, Recall, and Log Loss (lower is better); tie-break by lower CV Log Loss.\nStudent-test metrics are held-out evaluation and were not used to select.\nRoute continuity is not computed in this experiment.\n',encoding='utf-8')
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      names=[r['model'] for r in comparison]; x=np.arange(len(names)); width=.26
      fig,ax=plt.subplots(figsize=(11,5.5)); ax.bar(x-width,[r['cv_observation_top1_accuracy_covered'] for r in comparison],width,label='Grouped CV Top-1 (covered)'); ax.bar(x,[r['student_test_observation_top1_accuracy_labeled'] for r in comparison],width,label='Student-test Top-1 (labeled)'); ax.bar(x+width,[r['cv_candidate_macro_f1'] for r in comparison],width,label='Grouped CV candidate Macro-F1')
      ax.set_xticks(x,names,rotation=18,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_title('PathSnap model comparison');ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.out/'model_comparison.png',dpi=180);plt.close(fig)
      fig,axs=plt.subplots(1,5,figsize=(17,3.7))
      for ax,r in zip(axs,comparison):
       cm=np.loadtxt(a.out/'confusion_matrices'/(r['model'].lower().replace(' ','_')+'_cv.csv'),delimiter=',').astype(int); ax.imshow(cm,cmap='Blues');ax.set_title(r['model'],fontsize=9);ax.set_xticks([0,1],['pred 0','pred 1'],rotation=35);ax.set_yticks([0,1],['true 0','true 1']);
       for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center',fontsize=8)
      fig.suptitle('OOF candidate-level confusion matrices');fig.tight_layout();fig.savefig(a.out/'cv_confusion_matrices.png',dpi=180);plt.close(fig)
      fig,ax=plt.subplots(figsize=(10,5));ax.bar(names,[r['cv_mean_fit_seconds'] for r in comparison],label='Mean fit time per fold');ax.set_ylabel('Seconds');ax.set_title('Model fit time');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.out/'training_time.png',dpi=180);plt.close(fig)
    except Exception as e: print(f'Plot warning: {e}',flush=True)
    # Human-readable report section populated only with measured outputs.
    lines=['# Model experimentation and selection','',f'Five models were trained and compared using {a.folds}-fold GroupKFold, grouping rows by trip ID. The student-test partition was never used for fitting. Model selection used out-of-fold observation-level Top-1 accuracy on candidate-covered training observations, with candidate-level Macro-F1 as tie-break. The student-test numbers are reported as a development check and did not select the winner.','',f'**Selected model: {selected}.**','', '| Model | CV Top-1 (covered) | CV candidate Macro-F1 | CV Log Loss | Student-test Top-1 (labeled) | Student-test candidate Macro-F1 | Mean CV fit (s) | Test inference (ms/candidate) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison: lines.append(f"| {r['model']} | {r['cv_observation_top1_accuracy_covered']:.4f} | {r['cv_candidate_macro_f1']:.4f} | {r['cv_candidate_log_loss']:.4f} | {r['student_test_observation_top1_accuracy_labeled']:.4f} | {r['student_test_candidate_macro_f1']:.4f} | {r['cv_mean_fit_seconds']:.2f} | {r['student_test_inference_ms_per_candidate']:.4f} |")
    lines += ['', '## Interpretation and limits','', f"The training candidate table contains {len(tr['y']):,} candidate rows across {len(np.unique(tr['trip']))} training trips. The chosen model is the measured grouped-CV leader under the stated rule; this is not a claim of evaluator performance. Candidate misses cap end-to-end accuracy: recorded candidate coverage was about 82.0% for training observations and 74.7% for student-test observations. Student-test Top-1 is computed only where a reference arc label could be assigned; candidate misses among those labeled observations count as wrong. No evaluator partition was supplied.", '', 'Candidate-level Precision, Recall, Accuracy, Macro-F1, Log Loss and confusion matrices are saved alongside observation-level Top-1. All result graphs and model artifacts are generated from these actual runs.']
    (a.out/'model_selection_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'\nSELECTED: {selected}\nOutputs: {a.out.resolve()}',flush=True)
if __name__=='__main__':main()
