"""Group-aware Gaussian-process Bayesian optimization for HistGradientBoosting.

Selection objective: mean pooled candidate-level Log Loss over GroupKFold splits.
Only training trips are used here; student-test data is never read by this script.
"""
from __future__ import annotations
import argparse, csv, gzip, json, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

FEATURES=['distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank']
SEED=2026
SPACE=np.array([[.03,.20],[7,31],[50,250],[0.,5.],[20,80]],dtype=float)

def load_train(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        n=sum(1 for _ in f)-1
    X=np.empty((n,len(FEATURES)),np.float32); y=np.empty(n,np.uint8); groups=np.empty(n,dtype='U32')
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f)
        for i,row in enumerate(r):
            X[i]=[float(row[k]) for k in FEATURES]; y[i]=int(row['label']); groups[i]=row['trip_id']
    return X,y,groups

def params(v):
    return {'learning_rate':float(v[0]),'max_leaf_nodes':int(round(v[1])),'min_samples_leaf':int(round(v[2])),'l2_regularization':float(v[3]),'max_iter':int(round(v[4]))}

def model(p):
    return HistGradientBoostingClassifier(**p,early_stopping=True,random_state=42)

def cv_score(X,y,groups,splits,p):
    losses=[]; fit_s=0.; fold=[]
    for fold_id,(it,iv) in enumerate(splits,1):
        est=model(p); t=time.perf_counter(); est.fit(X[it],y[it]); fit_s+=time.perf_counter()-t
        prob=np.clip(est.predict_proba(X[iv])[:,1],1e-7,1-1e-7)
        loss=float(log_loss(y[iv],prob,labels=[0,1])); losses.append(loss)
        fold.append({'fold':fold_id,'rows':int(len(iv)),'validation_trips':int(len(np.unique(groups[iv]))),'log_loss':loss})
    return float(np.mean(losses)),fit_s,fold

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',type=Path,default=Path('data/train_candidate_features.csv.gz')); ap.add_argument('--out',type=Path,default=Path('results')); ap.add_argument('--trials',type=int,default=8); ap.add_argument('--folds',type=int,default=3); a=ap.parse_args()
    if a.trials<4: ap.error('Use at least 4 trials (3 initial plus at least 1 Bayesian proposal).')
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'models').mkdir(exist_ok=True)
    print('Loading the supplied training candidates...',flush=True); X,y,groups=load_train(a.train)
    if len(np.unique(groups))<a.folds: raise ValueError('Fewer distinct trips than CV folds.')
    splits=list(GroupKFold(n_splits=a.folds).split(X,y,groups)); rng=np.random.default_rng(SEED)
    base={'learning_rate':.1,'max_leaf_nodes':15,'min_samples_leaf':100,'l2_regularization':1.,'max_iter':35}
    trials=[]; seen=set(); scaler=MinMaxScaler().fit(SPACE.T)
    # Evaluate the existing untuned HGB settings on identical grouped folds.
    with threadpool_limits(limits=2):
        baseline,baseline_fit,baseline_folds=cv_score(X,y,groups,splits,base)
        print(f'Baseline grouped CV Log Loss={baseline:.6f}',flush=True)
        for i in range(a.trials):
            if i<3 or len(trials)<3:
                v=rng.uniform(SPACE[:,0],SPACE[:,1])
            else:
                observed=np.array([t['vector'] for t in trials]); scores=np.array([t['mean_cv_log_loss'] for t in trials])
                gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0,(.1,10))*Matern(length_scale=np.ones(5),nu=2.5)+WhiteKernel(noise_level=1e-5),normalize_y=True,random_state=SEED,n_restarts_optimizer=1)
                gp.fit(scaler.transform(observed),scores)
                pool=rng.uniform(SPACE[:,0],SPACE[:,1],size=(256,5)); mu,std=gp.predict(scaler.transform(pool),return_std=True)
                best=float(scores.min()); z=(best-mu-0.01)/np.maximum(std,1e-9)
                # Expected improvement for minimization.
                from scipy.stats import norm
                ei=(best-mu-0.01)*norm.cdf(z)+std*norm.pdf(z)
                order=np.argsort(-ei); v=next((pool[j] for j in order if tuple(np.round(pool[j],6)) not in seen),pool[order[0]])
            v[1]=round(v[1]);v[2]=round(v[2]);v[4]=round(v[4]); key=tuple(np.round(v,6))
            while key in seen:
                v=rng.uniform(SPACE[:,0],SPACE[:,1]);v[1]=round(v[1]);v[2]=round(v[2]);v[4]=round(v[4]);key=tuple(np.round(v,6))
            seen.add(key); p=params(v); score,fit_s,folds=cv_score(X,y,groups,splits,p)
            trials.append({'trial':i+1,'method':'random_initialization' if i<3 else 'gaussian_process_expected_improvement','vector':v.tolist(),'parameters':p,'mean_cv_log_loss':score,'fit_seconds_total':fit_s,'folds':folds})
            partial={'method':'Gaussian-process Bayesian optimization with Expected Improvement','objective':'Mean candidate-level Log Loss over group-aware CV folds','splitter':f'{a.folds}-fold GroupKFold by trip_id','student_test_used_for_optimization':False,'seed':SEED,'training_rows':int(len(y)),'training_trips':int(len(np.unique(groups))),'baseline_parameters':base,'baseline_mean_cv_log_loss':baseline,'baseline_fold_log_loss':baseline_folds,'trials':trials}
            (a.out/'optimization_trials_progress.json').write_text(json.dumps(partial,indent=2),encoding='utf-8')
            print(f"Trial {i+1}/{a.trials}: CV Log Loss={score:.6f} params={p}",flush=True)
    best=min(trials,key=lambda t:t['mean_cv_log_loss'])
    # Refit selected tuned HGB on all training candidate rows for downstream decoding.
    final=model(best['parameters'])
    with threadpool_limits(limits=2): final.fit(X,y)
    joblib.dump(final,a.out/'models'/'tuned_hist_gradient_boosting.joblib',compress=3)
    # Baseline model is copied from the package's measured original artifact when present;
    # otherwise refit using the explicit original settings on all supplied training rows.
    joblib.dump(model(base).fit(X,y),a.out/'models'/'untuned_hist_gradient_boosting.joblib',compress=3)
    summary={'method':'Gaussian-process Bayesian optimization with Expected Improvement','objective':'Mean candidate-level Log Loss over group-aware CV folds','splitter':f'{a.folds}-fold GroupKFold by trip_id','student_test_used_for_optimization':False,'seed':SEED,'training_rows':int(len(y)),'training_trips':int(len(np.unique(groups))),'baseline_parameters':base,'baseline_mean_cv_log_loss':baseline,'baseline_fold_log_loss':baseline_folds,'trials':trials,'selected_trial':best['trial'],'selected_parameters':best['parameters'],'selected_mean_cv_log_loss':best['mean_cv_log_loss'],'selected_fold_log_loss':best['folds'],'fit_seconds_baseline_cv':baseline_fit}
    (a.out/'optimization_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (a.out/'optimization_trials.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['trial','method','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','max_iter','mean_cv_log_loss','fit_seconds_total']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t in trials: w.writerow({'trial':t['trial'],'method':t['method'],**t['parameters'],'mean_cv_log_loss':t['mean_cv_log_loss'],'fit_seconds_total':t['fit_seconds_total']})
    (a.out/'optimization_selection.txt').write_text(f"Selected tuned model: Hist Gradient Boosting\nObjective: mean {a.folds}-fold GroupKFold candidate Log Loss, grouped by trip_id.\nUntuned baseline CV Log Loss: {baseline:.6f}\nBest tuned CV Log Loss: {best['mean_cv_log_loss']:.6f}\nSelected trial: {best['trial']}\nParameters: {json.dumps(best['parameters'],sort_keys=True)}\nStudent-test data was not used during optimization.\n",encoding='utf-8')
    print(f"BEST trial={best['trial']} CV log loss={best['mean_cv_log_loss']:.6f}; baseline={baseline:.6f}",flush=True)
if __name__=='__main__': main()
