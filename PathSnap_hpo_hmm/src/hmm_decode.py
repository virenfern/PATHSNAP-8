"""Evaluate Hist Gradient Boosting candidate ranking with a gap-aware HMM/Viterbi decoder."""
from __future__ import annotations
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from threadpoolctl import threadpool_limits

FEATURES=['distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank']

def read_test(path):
    trips=defaultdict(list)
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            tid=r['trip_id']; obs=int(r['observation_index'])
            if not trips[tid] or trips[tid][-1]['observation_index']!=obs:
                trips[tid].append({'observation_index':obs,'timestamp_s':float(r['timestamp_s']),'gps_gap_s':float(r['gps_gap_s']),'reference':int(r['reference_arc_id']) if r['reference_arc_id'] else None,'candidates':[],'X':[],'labels':[]})
            g=trips[tid][-1]
            g['candidates'].append(int(r['candidate_arc_id']));g['X'].append([float(r[k]) for k in FEATURES]);g['labels'].append(int(r['label']))
            if g['reference'] is None and r['reference_arc_id']:g['reference']=int(r['reference_arc_id'])
    return trips

def read_arcs(path):
    with path.open(encoding='utf-8') as f:return [tuple(map(int,line.split())) for line in f if line.strip()]

def connected(a,b,arcs):
    return a==b or (0<=a<len(arcs) and 0<=b<len(arcs) and arcs[a][1]==arcs[b][0])

def best_classifier(prob):return int(np.argmax(prob))

def viterbi(groups,arcs,transition_floor=1e-5,gap_relax_s=5.0):
    if not groups:return [],{'relaxed_gap_steps':0,'unconnected_candidate_steps':0}
    emissions=[]
    for g in groups:
        p=np.clip(g['prob'],1e-12,1.0);p=p/p.sum();emissions.append(np.log(p))
    score=emissions[0].copy();backs=[];relaxed=0;unconnected=0
    for t in range(1,len(groups)):
        prev,curr=groups[t-1],groups[t]; gap=max(0.0,float(curr['timestamp_s']-prev['timestamp_s'])); is_relaxed=gap>gap_relax_s
        if is_relaxed:relaxed+=1
        pa,ca=prev['candidates'],curr['candidates']; trans=np.empty((len(pa),len(ca)),dtype=float);has_link=False
        for i,a in enumerate(pa):
            for j,b in enumerate(ca):
                if a==b: w=2.0;has_link=True
                elif 0<=a<len(arcs) and 0<=b<len(arcs) and arcs[a][1]==arcs[b][0]:w=1.0;has_link=True
                else:w=max(transition_floor,.05 if is_relaxed else transition_floor)
                trans[i,j]=w
            trans[i]/=trans[i].sum()
        if not has_link:unconnected+=1
        logt=np.log(np.maximum(trans,1e-300))
        values=score[:,None]+logt
        back=np.argmax(values,axis=0);score=values[back,np.arange(len(ca))]+emissions[t]
        backs.append(back)
    state=int(np.argmax(score));path=[state]
    for back in reversed(backs):state=int(back[state]);path.append(state)
    path.reverse()
    return path,{'relaxed_gap_steps':relaxed,'unconnected_candidate_steps':unconnected}

def continuity(pred,groups,arcs,gap_limit=5.0):
    okay=total=0
    for i in range(1,len(groups)):
        if groups[i]['observation_index']!=groups[i-1]['observation_index']+1:continue
        if groups[i]['timestamp_s']-groups[i-1]['timestamp_s']>gap_limit:continue
        total+=1;okay+=int(connected(pred[i-1],pred[i],arcs))
    return (okay/total if total else float('nan')),okay,total

def summarize(trips,arc_root,estimator,out_csv):
    pred_c=[];pred_h=[];metrics=[];pred_rows=[];total_relaxed=total_unconnected=0; total_groups=0; total_labeled=0; labeled_reference_misses=0
    with threadpool_limits(limits=2):
        for tid,groups in sorted(trips.items()):
            total_groups+=len(groups)
            total_labeled+=sum(g['reference'] is not None for g in groups)
            labeled_reference_misses+=sum(g['reference'] is not None and g['reference'] not in g['candidates'] for g in groups)
            arcs=read_arcs(arc_root/f'{tid}.arcs')
            allx=np.vstack([g['X'] for g in groups]);allp=estimator.predict_proba(allx)[:,1]
            off=0
            for g in groups:
                n=len(g['candidates']);p=allp[off:off+n];off+=n;g['prob']=p
                g['classifier_arc']=g['candidates'][best_classifier(p)]
            path,diag=viterbi(groups,arcs);total_relaxed+=diag['relaxed_gap_steps'];total_unconnected+=diag['unconnected_candidate_steps']
            for g,j in zip(groups,path):g['hmm_arc']=g['candidates'][j]
            for mode,key in [('classifier','classifier_arc'),('hmm_viterbi','hmm_arc')]:
                pred=[g[key] for g in groups];cont,ok,ntrans=continuity(pred,groups,arcs)
                known=[i for i,g in enumerate(groups) if g['reference'] is not None]
                corr=sum(pred[i]==groups[i]['reference'] for i in known)
                metrics.append({'trip_id':tid,'decoder':mode,'candidate_covered_labeled_observations':len(known),'reference_arc_accuracy':corr/len(known) if known else float('nan'),'route_continuity_short_gap':cont,'continuous_transitions':ok,'eligible_short_gap_transitions':ntrans,'long_gap_steps_relaxed':diag['relaxed_gap_steps'],'steps_with_no_link_between_candidate_sets':diag['unconnected_candidate_steps']})
                if mode=='classifier':pred_c.extend((groups[i]['reference'],pred[i]) for i in known)
                else:pred_h.extend((groups[i]['reference'],pred[i]) for i in known)
            for g in groups:
                pred_rows.append({'trip_id':tid,'observation_index':g['observation_index'],'timestamp_s':g['timestamp_s'],'gps_gap_s':g['gps_gap_s'],'reference_arc_id':g['reference'] if g['reference'] is not None else '', 'classifier_arc_id':g['classifier_arc'],'hmm_viterbi_arc_id':g['hmm_arc'],'classifier_top1_correct':int(g['reference']==g['classifier_arc']) if g['reference'] is not None else '', 'hmm_top1_correct':int(g['reference']==g['hmm_arc']) if g['reference'] is not None else ''})
    def aggregate(name,pairs):
        y=[x[0] for x in pairs];p=[x[1] for x in pairs]
        return {'decoder':name,'labeled_observations':len(y),'reference_arc_accuracy':accuracy_score(y,p) if y else float('nan')}
    overall=[]
    for name,pairs in [('classifier_only',pred_c),('hmm_viterbi',pred_h)]:overall.append(aggregate(name,pairs))
    for name in ('classifier','hmm_viterbi'):
        selected=[m for m in metrics if m['decoder']==name];total=sum(x['eligible_short_gap_transitions'] for x in selected);ok=sum(x['continuous_transitions'] for x in selected)
        row=next(x for x in overall if x['decoder']==('classifier_only' if name=='classifier' else 'hmm_viterbi'))
        row.update({'route_continuity_short_gap':ok/total if total else float('nan'),'continuous_transitions':ok,'eligible_short_gap_transitions':total})
    with (out_csv.parent/'route_continuity_by_trip.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (out_csv.parent/'decoder_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(overall[0]));w.writeheader();w.writerows(overall)
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pred_rows[0]));w.writeheader();w.writerows(pred_rows)
    return {'overall':overall,'represented_observation_groups':total_groups,'labeled_observation_groups':total_labeled,'labeled_reference_arc_missing_from_candidate_set':labeled_reference_misses,'labeled_reference_candidate_coverage':(total_labeled-labeled_reference_misses)/total_labeled if total_labeled else float('nan'),'total_relaxed_gap_steps':total_relaxed,'total_steps_with_no_link_between_candidate_sets':total_unconnected,'continuity_definition':'Fraction of consecutive predicted arcs that are identical or directly connected (end node of prior arc equals start node of next arc), among represented consecutive GPS observations with gap <= 5 seconds.','evaluation_scope':'Accuracy includes all represented student-test observations with an assigned reference arc. If the reference arc is absent from that observation’s candidate set, the prediction is necessarily counted wrong. Unlabeled observations are excluded from accuracy.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=Path,default=Path('data/student_test_candidate_features.csv.gz'));ap.add_argument('--arcs',type=Path,default=Path('data/student_test_arcs'));ap.add_argument('--model',type=Path,default=Path('results/models/tuned_hist_gradient_boosting.joblib'));ap.add_argument('--out',type=Path,default=Path('results'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    trips=read_test(a.candidates);model=joblib.load(a.model)
    summary=summarize(trips,a.arcs,model,a.out/'student_test_decoded_predictions.csv')
    summary['trips_evaluated']=len(trips);summary['model']=a.model.name
    (a.out/'decoder_evaluation.json').write_text(json.dumps(summary,indent=2,allow_nan=True),encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=True))
if __name__=='__main__':main()
