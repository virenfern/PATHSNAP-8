"""Dataset EDA, local metric geometry, network audit, candidate generation and labels.

All labels are derived from the supplied .route reference and are explicitly described as
reference-route labels. No evaluator data is read unless its directory is passed explicitly.
"""
from __future__ import annotations
import argparse, bisect, csv, gzip, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

R_EARTH=6371008.8

def read_rows(path, cast=float):
    out=[]
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            ss=line.split()
            if not ss: continue
            try: out.append(tuple(cast(x) for x in ss))
            except Exception as e: raise ValueError(f'{path}:{line_no}: invalid value') from e
    return out

def bearing(x1,y1,x2,y2): return math.atan2(x2-x1,y2-y1)
def angle_diff(a,b): return abs((a-b+math.pi)%(2*math.pi)-math.pi)
def haversine(lon1,lat1,lon2,lat2):
    p1,p2=math.radians(lat1),math.radians(lat2); dp=p2-p1; dl=math.radians(lon2-lon1)
    z=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R_EARTH*math.asin(min(1,math.sqrt(z)))

def project(lon,lat,lon0,lat0):
    # Local equirectangular/tangent-plane coordinates, suitable for the short trip/network extents here.
    return (R_EARTH*math.radians(lon-lon0)*math.cos(math.radians(lat0)),R_EARTH*math.radians(lat-lat0))

def segment_distance(px,py,x1,y1,x2,y2):
    dx=x2-x1; dy=y2-y1; l2=dx*dx+dy*dy
    t=0.0 if l2==0 else max(0.0,min(1.0,((px-x1)*dx+(py-y1)*dy)/l2))
    qx=x1+t*dx; qy=y1+t*dy
    return math.hypot(px-qx,py-qy),t,math.sqrt(l2)

class UnionFind:
    def __init__(self,n): self.p=list(range(n)); self.sz=[1]*n; self.components=n
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b:return
        if self.sz[a]<self.sz[b]:a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]; self.components-=1

class SegmentIndex:
    def __init__(self,nodes,arcs,cell=100):
        self.cell=cell; self.seg=[]; self.grid=defaultdict(list)
        for aid,(u,v) in enumerate(arcs):
            x1,y1=nodes[u]; x2,y2=nodes[v]
            length=math.hypot(x2-x1,y2-y1); b=bearing(x1,y1,x2,y2)
            self.seg.append((x1,y1,x2,y2,length,b))
            # A segment is inserted in each cell touched by its bounding box, padded slightly.
            ix0=math.floor((min(x1,x2)-5)/cell); ix1=math.floor((max(x1,x2)+5)/cell)
            iy0=math.floor((min(y1,y2)-5)/cell); iy1=math.floor((max(y1,y2)+5)/cell)
            for ix in range(ix0,ix1+1):
                for iy in range(iy0,iy1+1): self.grid[(ix,iy)].append(aid)
    def query(self,x,y,radius,limit):
        ix=math.floor(x/self.cell); iy=math.floor(y/self.cell); k=math.ceil(radius/self.cell)+1; found=set()
        for gx in range(ix-k,ix+k+1):
            for gy in range(iy-k,iy+k+1): found.update(self.grid.get((gx,gy),()))
        scored=[]
        for aid in found:
            x1,y1,x2,y2,length,brg=self.seg[aid]
            d,t,_=segment_distance(x,y,x1,y1,x2,y2)
            if d<=radius: scored.append((d,aid,t,length,brg))
        scored.sort(key=lambda z:(z[0],z[1]))
        return scored[:limit]

def trip_audit(folder,partition,make_features=False,feature_writer=None,config=None):
    tid=folder.name
    track=read_rows(folder/f'{tid}.track',float); node_geo=read_rows(folder/f'{tid}.nodes',float)
    arcs=read_rows(folder/f'{tid}.arcs',int); route=[r[0] for r in read_rows(folder/f'{tid}.route',int)]
    if any(len(x)!=3 for x in track) or any(len(x)!=2 for x in node_geo) or any(len(x)!=2 for x in arcs): raise ValueError(f'{tid}: unexpected row width')
    n=len(node_geo)
    if any(u<0 or v<0 or u>=n or v>=n for u,v in arcs): raise ValueError(f'{tid}: arc endpoint outside node table')
    if any(a<0 or a>=len(arcs) for a in route): raise ValueError(f'{tid}: route arc ID outside arc table')
    lons=[p[0] for p in track]; lats=[p[1] for p in track]; ts=[p[2] for p in track]
    if not track: raise ValueError(f'{tid}: empty trajectory')
    # GPS quality/anomaly summary
    gaps=[ts[i]-ts[i-1] for i in range(1,len(ts))]
    dist=[]; speeds=[]; duplicate_xy=0; duplicate_ts=0
    for i in range(1,len(track)):
        d=haversine(lons[i-1],lats[i-1],lons[i],lats[i]); dt=gaps[i-1]; dist.append(d)
        if (lons[i],lats[i])==(lons[i-1],lats[i-1]): duplicate_xy+=1
        if dt==0: duplicate_ts+=1
        speeds.append(d/dt if dt>0 else math.inf)
    positive=[g for g in gaps if g>0]
    gps={'partition':partition,'trip_id':tid,'gps_rows':len(track),'missing_or_bad_gps_rows':0,'timestamp_non_increasing_steps':sum(g<=0 for g in gaps),'zero_timestamp_steps':duplicate_ts,'gap_gt_1_5s':sum(g>1.5 for g in gaps),'gap_gt_5s':sum(g>5 for g in gaps),'gap_max_s':max(gaps,default=0),'gap_median_positive_s':statistics.median(positive) if positive else 0,'duplicate_consecutive_coordinates':duplicate_xy,'speed_gt_40mps_steps':sum(s>40 for s in speeds),'speed_max_mps':max((s for s in speeds if math.isfinite(s)),default=0),'lon_min':min(lons),'lon_max':max(lons),'lat_min':min(lats),'lat_max':max(lats),'nodes':n,'arcs':len(arcs),'route_arcs':len(route),'file_bytes':sum((folder/f'{tid}{ext}').stat().st_size for ext in ('.track','.nodes','.arcs','.route'))}
    # Weak connectivity and degree analysis; arc direction is preserved in out/in counts.
    uf=UnionFind(n); indeg=[0]*n; outdeg=[0]*n
    for u,v in arcs: uf.union(u,v); outdeg[u]+=1; indeg[v]+=1
    roots=Counter(uf.find(i) for i in range(n)); component_sizes=sorted(roots.values(),reverse=True)
    isolated=sum(indeg[i]+outdeg[i]==0 for i in range(n))
    net={'partition':partition,'trip_id':tid,'nodes':n,'arcs':len(arcs),'weak_components':len(component_sizes),'largest_component_nodes':component_sizes[0] if component_sizes else 0,'largest_component_fraction':component_sizes[0]/n if n else 0,'isolated_nodes':isolated,'zero_outdegree_nodes':sum(x==0 for x in outdeg),'zero_indegree_nodes':sum(x==0 for x in indeg),'self_loop_arcs':sum(u==v for u,v in arcs),'duplicate_arc_rows':len(arcs)-len(set(arcs))}
    # Candidate features and supervision, only train partition. Reference alignment is monotone.
    diagnostics={'candidate_observations':0,'candidate_misses':0,'route_alignment_fallbacks':0,'candidate_rows':0,'training_observations_covered':0,'training_candidate_rows':0,'reference_label_unmatched':0}
    alignment_fallbacks=0
    if make_features:
        lon0=sum(lons)/len(lons); lat0=sum(lats)/len(lats)
        nodes=[project(lon,lat,lon0,lat0) for lon,lat in node_geo]
        gps_xy=[project(lon,lat,lon0,lat0) for lon,lat,_ in track]
        index=SegmentIndex(nodes,arcs,config['cell_m'])
        # Spatial lookup of route geometry reuses road candidates. Greedy monotone occurrence alignment
        # assigns each GPS observation to the closest eligible route arc along the supplied sequence.
        route_positions=defaultdict(list)
        route_progress=[]; total_route_m=0.0
        for j,aid in enumerate(route):
            route_positions[aid].append(j)
            route_progress.append(total_route_m)
            total_route_m += index.seg[aid][4]
        gps_progress=[0.0]
        for j in range(1,len(track)):
            gps_progress.append(gps_progress[-1]+haversine(lons[j-1],lats[j-1],lons[j],lats[j]))
        total_gps_m=gps_progress[-1]
        progress_weight=config['progress_weight']
        last_pos=0; matched=[]; alignment_fallbacks=0
        for i,(x,y) in enumerate(gps_xy):
            target_progress=total_route_m*(gps_progress[i]/total_gps_m) if total_gps_m>0 else 0.0
            near=index.query(x,y,config['label_radius_m'],config['max_label_candidates'])
            eligible=[]
            for d,aid,*_ in near:
                ps=route_positions.get(aid)
                if ps:
                    k=bisect.bisect_left(ps,last_pos)
                    if k<len(ps):
                        occ=ps[k:]
                        q=bisect.bisect_left([route_progress[pos] for pos in occ],target_progress)
                        for oi in {max(0,q-1),min(len(occ)-1,q)}:
                            pos=occ[oi]
                            score=d+progress_weight*abs(route_progress[pos]-target_progress)
                            eligible.append((score,d,aid,pos))
            if not eligible:
                # Explicit fallback allows unavoidable GPS noise/coverage gaps; counted for disclosure.
                alignment_fallbacks+=1
                all_near=index.query(x,y,config['fallback_label_radius_m'],config['max_label_candidates']*4)
                for d,aid,*_ in all_near:
                    ps=route_positions.get(aid)
                    if ps:
                        k=bisect.bisect_left(ps,last_pos)
                        if k<len(ps):
                            occ=ps[k:]
                            q=bisect.bisect_left([route_progress[pos] for pos in occ],target_progress)
                            for oi in {max(0,q-1),min(len(occ)-1,q)}:
                                pos=occ[oi]
                                score=d+progress_weight*abs(route_progress[pos]-target_progress)
                                eligible.append((score,d,aid,pos))
            if eligible:
                _,d,aid,pos=min(eligible,key=lambda z:(z[0],z[3])); last_pos=pos; matched.append((aid,pos,d))
            else: matched.append((None,None,None))
        for i,(x,y) in enumerate(gps_xy):
            candidates=index.query(x,y,config['candidate_radius_m'],config['max_candidates'])
            ref_arc,ref_pos,ref_d=matched[i]
            diagnostics['candidate_observations']+=1
            hit=ref_arc is not None and any(c[1]==ref_arc for c in candidates)
            if not hit: diagnostics['candidate_misses']+=1
            diagnostics['candidate_rows']+=len(candidates)
            if ref_arc is None: diagnostics['reference_label_unmatched']+=1
            if not candidates: continue
            if partition=='train' and not hit: continue
            if feature_writer is None: continue
            if partition=='train' and feature_writer is not None:
                diagnostics['training_observations_covered']+=1
                diagnostics['training_candidate_rows']+=len(candidates)
            if i==0: motion_bearing=None; speed=0.0; dt=0.0; heading_valid=0
            else:
                dx=x-gps_xy[i-1][0]; dy=y-gps_xy[i-1][1]; dt=gaps[i-1]
                step_m=math.hypot(dx,dy); speed=step_m/dt if dt>0 else 0.0
                motion_bearing=math.atan2(dx,dy) if dt>0 and step_m>=1.0 and dt<=5.0 else None
                heading_valid=int(motion_bearing is not None)
            for rank,(dist_m,aid,tproj,length,road_bearing) in enumerate(candidates,1):
                label=int(aid==ref_arc)
                delta=((road_bearing-motion_bearing+math.pi)%(2*math.pi)-math.pi) if motion_bearing is not None else 0.0
                feature_writer.writerow({'partition':partition,'trip_id':tid,'observation_index':i,'timestamp_s':track[i][2],
                    'candidate_arc_id':aid,'reference_arc_id':ref_arc,'reference_route_position':ref_pos,'label':label,
                    'distance_to_segment_m':dist_m,'projection_fraction':tproj,'segment_length_m':length,
                    'road_bearing_sin':math.sin(road_bearing),'road_bearing_cos':math.cos(road_bearing),
                    'heading_difference_sin':math.sin(delta),'heading_difference_cos':math.cos(delta),
                    'heading_valid':heading_valid,'gps_speed_mps':speed,'gps_gap_s':dt,'candidate_rank':rank})
    diagnostics['route_alignment_fallbacks']=alignment_fallbacks
    return gps,net,diagnostics

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path,help='Root containing train/ and student_test/');p.add_argument('--out',required=True,type=Path);p.add_argument('--candidate-radius-m',type=float,default=60);p.add_argument('--label-radius-m',type=float,default=100);p.add_argument('--fallback-label-radius-m',type=float,default=300);p.add_argument('--max-candidates',type=int,default=12);p.add_argument('--cell-m',type=float,default=100);p.add_argument('--skip-features',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg={'candidate_radius_m':a.candidate_radius_m,'label_radius_m':a.label_radius_m,'fallback_label_radius_m':a.fallback_label_radius_m,'max_candidates':a.max_candidates,'max_label_candidates':256,'cell_m':a.cell_m,'progress_weight':0.35}
    gps_rows=[];net_rows=[]; diag_rows=[]; train_path=a.out/'train_candidate_features.csv.gz'; fields=['partition','trip_id','observation_index','timestamp_s','candidate_arc_id','reference_arc_id','reference_route_position','label','distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank']
    writer=None
    try:
        if not a.skip_features:
            writer=gzip.open(train_path,'wt',newline='',encoding='utf-8');csv.DictWriter(writer,fieldnames=fields).writeheader()
        for partition in ('train','student_test'):
            root=a.data/partition
            for folder in sorted(x for x in root.iterdir() if x.is_dir()):
                gp,net,dg=trip_audit(folder,partition,make_features=(not a.skip_features),feature_writer=csv.DictWriter(writer,fieldnames=fields) if writer and partition=='train' else None,config=cfg)
                gps_rows.append(gp);net_rows.append(net);diag_rows.append({'partition':partition,'trip_id':folder.name,**dg})
                print(f"{partition}/{folder.name}: gps={gp['gps_rows']} nodes={net['nodes']} arcs={net['arcs']} weak_components={net['weak_components']} candidates={dg['candidate_rows']} misses={dg['candidate_misses']}",flush=True)
    finally:
        if writer:writer.close()
    write_csv(a.out/'gps_quality.csv',gps_rows);write_csv(a.out/'road_network_connectivity.csv',net_rows);write_csv(a.out/'candidate_coverage.csv',diag_rows)
    candidate_summary={k:sum(r[k] for r in diag_rows) for k in ('candidate_observations','candidate_misses','candidate_rows','route_alignment_fallbacks','training_observations_covered','training_candidate_rows')}
    for part in ('train','student_test'):
        subset=[r for r in diag_rows if r['partition']==part]
        obs=sum(r['candidate_observations'] for r in subset); miss=sum(r['candidate_misses'] for r in subset)
        candidate_summary[part+'_reference_candidate_coverage']=((obs-miss)/obs if obs else None)
    summary={'data_root':'local source path omitted','partitions':{k:sum(r['partition']==k for r in gps_rows) for k in ('train','student_test')},'trip_count':len(gps_rows),'config':cfg,'gps_totals':{k:sum(r[k] for r in gps_rows) for k in ('gps_rows','nodes','arcs','route_arcs','file_bytes','timestamp_non_increasing_steps','gap_gt_1_5s','gap_gt_5s','duplicate_consecutive_coordinates','speed_gt_40mps_steps')},'network_summary':{'mean_weak_components':statistics.mean(r['weak_components'] for r in net_rows),'median_largest_component_fraction':statistics.median(r['largest_component_fraction'] for r in net_rows),'trips_with_isolated_nodes':sum(r['isolated_nodes']>0 for r in net_rows)},'candidate_summary':candidate_summary}
    (a.out/'preprocessing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.out/'feature_schema.json').write_text(json.dumps({'predictor_features':['distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank'],'target':'label','target_meaning':'1 if candidate_arc_id equals the route arc assigned to this GPS observation by monotone progress-aware spatial alignment; 0 otherwise','identifiers_and_target_metadata_to_exclude_from_X':['partition','trip_id','observation_index','timestamp_s','candidate_arc_id','reference_arc_id','reference_route_position'],'split_guidance':'Group by trip_id. Never split candidate rows randomly across train/validation folds.','coverage_note':'Only observations whose assigned reference arc occurs in the candidate set are written to the training table. candidate_coverage.csv records all observations, including misses.'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()



