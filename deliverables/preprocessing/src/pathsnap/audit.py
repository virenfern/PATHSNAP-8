"""Audit raw PathSnap trip files without altering source data."""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path


def rows(path: Path):
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            s=line.split()
            if not s: continue
            yield line_no, s

def audit_trip(folder: Path, partition: str):
    tid=folder.name
    track=folder/f'{tid}.track'; nodes=folder/f'{tid}.nodes'; arcs=folder/f'{tid}.arcs'; route=folder/f'{tid}.route'
    missing=[p.suffix for p in (track,nodes,arcs,route) if not p.is_file()]
    if missing: return {'partition':partition,'trip_id':tid,'missing_files':';'.join(missing)}
    ntrack=nnodes=narcs=nroute=bad_track=bad_nodes=bad_arcs=bad_route=0
    lon_min=lat_min=ts_min=math.inf; lon_max=lat_max=ts_max=-math.inf
    prev=None; gaps=[]
    for _,s in rows(track):
        ntrack+=1
        try:
            if len(s)!=3: raise ValueError
            lon,lat,ts=map(float,s)
            if not all(map(math.isfinite,(lon,lat,ts))): raise ValueError
            lon_min=min(lon_min,lon); lon_max=max(lon_max,lon); lat_min=min(lat_min,lat); lat_max=max(lat_max,lat)
            ts_min=min(ts_min,ts); ts_max=max(ts_max,ts)
            if prev is not None: gaps.append(ts-prev)
            prev=ts
        except (ValueError,TypeError): bad_track+=1
    for _,s in rows(nodes):
        nnodes+=1
        try:
            if len(s)!=2 or not all(map(math.isfinite,map(float,s))): raise ValueError
        except ValueError: bad_nodes+=1
    for _,s in rows(arcs):
        narcs+=1
        try:
            if len(s)!=2 or min(map(int,s))<0: raise ValueError
        except ValueError: bad_arcs+=1
    for _,s in rows(route):
        nroute+=1
        try:
            if len(s)!=1 or not 0<=int(s[0])<narcs: raise ValueError
        except ValueError: bad_route+=1
    gaps.sort()
    size=sum(p.stat().st_size for p in (track,nodes,arcs,route))
    return {'partition':partition,'trip_id':tid,'track_rows':ntrack,'nodes':nnodes,'arcs':narcs,'route_rows':nroute,'missing_files':'','bad_track_rows':bad_track,'bad_node_rows':bad_nodes,'bad_arc_rows':bad_arcs,'bad_route_rows_or_arc_ids':bad_route,'timestamp_min_s':ts_min if ntrack else '', 'timestamp_max_s':ts_max if ntrack else '', 'gap_count':len(gaps),'gap_median_s':gaps[len(gaps)//2] if gaps else '', 'gap_max_s':max(gaps) if gaps else '', 'longitude_min':lon_min if ntrack else '', 'longitude_max':lon_max if ntrack else '', 'latitude_min':lat_min if ntrack else '', 'latitude_max':lat_max if ntrack else '', 'bytes_all_files':size}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True,type=Path,help='Folder containing train/ and student_test/')
    ap.add_argument('--out',required=True,type=Path); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for partition in ('train','student_test'):
        root=a.data/partition
        for trip in sorted(p for p in root.iterdir() if p.is_dir()): records.append(audit_trip(trip,partition))
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (a.out/'dataset_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    lines=['# Dataset audit','',f'- Data root: `{a.data.resolve()}`',f'- Trips inventoried: {len(records)}']
    for part in ('train','student_test'):
        rs=[r for r in records if r['partition']==part]
        lines.append(f'- `{part}` trips: {len(rs)}; track observations: {sum(int(r.get("track_rows",0) or 0) for r in rs):,}; road nodes: {sum(int(r.get("nodes",0) or 0) for r in rs):,}; arcs: {sum(int(r.get("arcs",0) or 0) for r in rs):,}; route rows: {sum(int(r.get("route_rows",0) or 0) for r in rs):,}')
        lines.append(f'- `{part}` trips with missing files: {sum(bool(r.get("missing_files")) for r in rs)}; malformed track/node/arc/route rows: '+', '.join(str(sum(int(r.get(k,0) or 0) for r in rs)) for k in ('bad_track_rows','bad_node_rows','bad_arc_rows','bad_route_rows_or_arc_ids')))
    lines += ['', '## Scope and limitations','', '- The supplied archive contains only `train/` and `student_test/`; no evaluator-test partition was present.', '- This audit checks file presence, row shape, numeric validity, timestamp gaps, coordinate extent and route arc-id bounds. It does not claim map-matching quality or independently verified ground truth.', '- Route rows are reference arc IDs. GPS observations are not assumed to correspond row-for-row to route entries.', '- Evaluator data, final model metrics, optimisation trials, decoder comparisons, API/UI demonstration screenshots and public GitHub publication are not fabricated by this audit.']
    (a.out/'dataset_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
