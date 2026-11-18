"""Create an interactive Folium map from an actual supplied PathSnap trip."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import folium
from folium.plugins import Fullscreen

def read_xy(path):
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        v=line.split()
        if v: out.append((float(v[0]),float(v[1])))
    return out

def read_pairs(path):
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        v=line.split()
        if v: out.append((int(v[0]),int(v[1])))
    return out

def build_map(data:Path,partition:str,trip_id:str,output:Path,network_radius_m:float=350,max_network_edges:int=4000):
    folder=data/partition/trip_id
    track=read_xy(folder/f'{trip_id}.track'); nodes=read_xy(folder/f'{trip_id}.nodes')
    arcs=read_pairs(folder/f'{trip_id}.arcs'); route=[int(x) for x in (folder/f'{trip_id}.route').read_text(encoding='utf-8').split()]
    if not track: raise ValueError(f'No GPS observations in {folder}')
    if any(a<0 or a>=len(arcs) for a in route): raise ValueError('Reference route contains invalid arc IDs')
    if any(u<0 or v<0 or u>=len(nodes) or v>=len(nodes) for u,v in arcs): raise ValueError('Arc endpoint outside node table')
    center=[sum(p[1] for p in track)/len(track),sum(p[0] for p in track)/len(track)]
    m=folium.Map(location=center,zoom_start=14,control_scale=True,tiles='OpenStreetMap')
    gps=folium.FeatureGroup(name='Observed GPS points',show=True)
    for i,(lon,lat) in enumerate(track):
        folium.CircleMarker((lat,lon),radius=3,color='#1769aa',fill=True,fill_opacity=.8,tooltip=f'GPS observation {i+1}').add_to(gps)
    gps.add_to(m)
    ref=folium.FeatureGroup(name='Supplied reference route',show=True)
    coords=[]
    for aid in route:
        u,v=arcs[aid]; coords.extend([(nodes[u][1],nodes[u][0]),(nodes[v][1],nodes[v][0])])
    if coords: folium.PolyLine(coords,color='#20853b',weight=4,opacity=.9,tooltip=f'Reference route ({len(route)} arc IDs from .route)').add_to(ref)
    ref.add_to(m)
    # Show road segments whose endpoints fall in a local bbox around the track. Cap
    # displayed geometry so regional network files do not create unwieldy HTML maps.
    lat_pad=network_radius_m/111000; lon_pad=network_radius_m/(111000*max(.15,math.cos(math.radians(center[0]))))
    lats=[p[1] for p in track]; lons=[p[0] for p in track]
    latlo,lathi=min(lats)-lat_pad,max(lats)+lat_pad; lonlo,lonhi=min(lons)-lon_pad,max(lons)+lon_pad
    near={i for i,(lon,lat) in enumerate(nodes) if latlo<=lat<=lathi and lonlo<=lon<=lonhi}
    edge_ids=[i for i,(u,v) in enumerate(arcs) if u in near and v in near]
    if len(edge_ids)>max_network_edges:
        stride=math.ceil(len(edge_ids)/max_network_edges); edge_ids=edge_ids[::stride][:max_network_edges]
    road=folium.FeatureGroup(name=f'Nearby road network (up to {max_network_edges} arcs)',show=False)
    for i in edge_ids:
        u,v=arcs[i]
        folium.PolyLine([(nodes[u][1],nodes[u][0]),(nodes[v][1],nodes[v][0])],color='#888888',weight=1,opacity=.35).add_to(road)
    road.add_to(m); folium.LayerControl(collapsed=False).add_to(m); Fullscreen().add_to(m)
    m.fit_bounds([[min(lats),min(lons)],[max(lats),max(lons)]])
    output.parent.mkdir(parents=True,exist_ok=True); m.save(str(output))
    return len(track),len(route),len(edge_ids)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True,type=Path); p.add_argument('--partition',choices=['train','student_test'],default='train'); p.add_argument('--trip',required=True); p.add_argument('--out',required=True,type=Path); p.add_argument('--network-radius-m',type=float,default=350); p.add_argument('--max-network-edges',type=int,default=4000)
    a=p.parse_args(); t,r,e=build_map(a.data,a.partition,a.trip,a.out,a.network_radius_m,a.max_network_edges)
    print(f'Saved {a.out}; GPS observations={t}; supplied reference-route arcs={r}; road arcs shown={e}')
if __name__=='__main__': main()
