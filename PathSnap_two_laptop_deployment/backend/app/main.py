"""FastAPI inference and route-evaluation API for the two-laptop demo."""
from __future__ import annotations
import logging,os
from pathlib import Path
import numpy as np
import joblib
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from threadpoolctl import threadpool_limits
from .decoder import read_test,viterbi,connected

ROOT=Path(__file__).resolve().parents[1]
TRIP_DIR=ROOT/'data'/'trips'
CANDIDATE_FILE=ROOT/'data'/'candidates'/'student_test_candidate_features.csv.gz'
MODEL_FILE=ROOT/'models'/'tuned_hist_gradient_boosting.joblib'
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'),format='%(asctime)s %(levelname)s %(name)s %(message)s')
log=logging.getLogger('pathsnap.api')

app=FastAPI(title='PathSnap Route API',version='1.0.0',description='Trip listing, Hist Gradient Boosting inference, and HMM/Viterbi route decoding.')
origins=[x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:3000,http://127.0.0.1:3000').split(',') if x.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_origin_regex=r'^http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+):3000$',allow_credentials=False,allow_methods=['GET'],allow_headers=['*'])

log.info('Loading candidate tables and tuned Hist Gradient Boosting artifact')
TRIPS=read_test(CANDIDATE_FILE)
MODEL=joblib.load(MODEL_FILE)
PAYLOAD_CACHE={}
log.info('Loaded model and %d supplied student-test trips',len(TRIPS))

def read_rows(path,cast=float):
    with path.open(encoding='utf-8') as f:return [tuple(cast(v) for v in line.split()) for line in f if line.strip()]

def arc_coords(arc_ids,arcs,nodes):
    coords=[];previous=None
    for aid in arc_ids:
        if aid==previous:continue
        previous=aid
        if aid<0 or aid>=len(arcs):continue
        u,v=arcs[aid]
        for node in (u,v):
            point=[float(nodes[node][0]),float(nodes[node][1])]
            if not coords or coords[-1]!=point:coords.append(point)
    return coords

def build_trip_payload(trip_id:str):
    groups=TRIPS[trip_id];folder=TRIP_DIR/trip_id
    nodes=read_rows(folder/f'{trip_id}.nodes');arcs=read_rows(folder/f'{trip_id}.arcs',int)
    reference_route=[r[0] for r in read_rows(folder/f'{trip_id}.route',int)]
    track=read_rows(folder/f'{trip_id}.track')
    features=np.vstack([g['X'] for g in groups])
    with threadpool_limits(limits=2):probabilities=MODEL.predict_proba(features)[:,1]
    offset=0
    for group in groups:
        n=len(group['candidates']);group['prob']=probabilities[offset:offset+n];offset+=n
        group['classifier_arc']=group['candidates'][int(np.argmax(group['prob']))]
    path,decoder_info=viterbi(groups,arcs)
    for group,index in zip(groups,path):group['hmm_arc']=group['candidates'][index]

    labeled=[g for g in groups if g['reference'] is not None]
    labeled_candidate_misses=sum(g['reference'] not in g['candidates'] for g in labeled)
    decoders={}
    for name,key in [('classifier_only','classifier_arc'),('hmm_viterbi','hmm_arc')]:
        selected=[g[key] for g in groups]
        correct=sum(g[key]==g['reference'] for g in labeled)
        eligible=continuous=0
        for i in range(1,len(groups)):
            prev,curr=groups[i-1],groups[i]
            if curr['observation_index']!=prev['observation_index']+1:continue
            if curr['timestamp_s']-prev['timestamp_s']>5:continue
            eligible+=1;continuous+=int(connected(selected[i-1],selected[i],arcs))
        decoders[name]={'predicted_arc_ids':selected,'predicted_route':arc_coords(selected,arcs,nodes),'reference_arc_accuracy':correct/len(labeled) if labeled else None,'labeled_observations':len(labeled),'labeled_reference_arc_missing_from_candidates':labeled_candidate_misses,'route_continuity':continuous/eligible if eligible else None,'continuous_transitions':continuous,'eligible_short_gap_transitions':eligible,'long_gap_steps_relaxed':decoder_info['relaxed_gap_steps'],'steps_with_no_link_between_candidate_sets':decoder_info['unconnected_candidate_steps']}

    candidate_arc_ids={aid for g in groups for aid in g['candidates']}
    candidate_roads=[]
    for aid in sorted(candidate_arc_ids):
        if 0<=aid<len(arcs):
            u,v=arcs[aid];candidate_roads.append([[float(nodes[u][0]),float(nodes[u][1])],[float(nodes[v][0]),float(nodes[v][1])]])
    payload={'trip_id':trip_id,'partition':'student_test','is_hidden_evaluator_data':False,'gps_point_count':len(track),'network_nodes':len(nodes),'network_arcs':len(arcs),'candidate_observation_groups':len(groups),'reference_route_arc_count':len(reference_route),'track':[[float(x[0]),float(x[1])] for x in track],'road_candidates':candidate_roads,'reference_route':arc_coords(reference_route,arcs,nodes),'decoders':decoders,'data_note':'These are the supplied student-test trips, shown as demo examples. No hidden evaluator partition was supplied.'}
    PAYLOAD_CACHE[trip_id]=payload
    return payload

@app.get('/api/health')
def health():return {'status':'ok','model':'Tuned Hist Gradient Boosting','trips':len(TRIPS)}

@app.get('/api/trips')
def list_trips():
    rows=[]
    for tid,groups in sorted(TRIPS.items()):
        track=read_rows(TRIP_DIR/tid/f'{tid}.track')
        rows.append({'trip_id':tid,'partition':'student_test','gps_points':len(track),'candidate_observations':len(groups)})
    log.info('Listed %d student-test trips',len(rows))
    return {'trips':rows,'note':'Supplied student-test examples; no hidden evaluator partition was supplied.'}

@app.get('/api/trips/{trip_id}')
def get_trip(trip_id:str):
    if trip_id not in TRIPS:raise HTTPException(status_code=404,detail='Unknown trip ID')
    if trip_id not in PAYLOAD_CACHE:
        log.info('Computing prediction and Viterbi route for trip_id=%s',trip_id)
        try:build_trip_payload(trip_id)
        except Exception:
            log.exception('Trip processing failed for trip_id=%s',trip_id)
            raise HTTPException(status_code=500,detail='Could not process this trip; see backend logs.')
    return PAYLOAD_CACHE[trip_id]
