"""Regenerate the supplied student-test candidate features using the project's pipeline."""
from __future__ import annotations
import argparse,csv,gzip,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BACKEND=HERE.parent
sys.path.insert(0,str(HERE))
from preprocess import trip_audit

FIELDS=['partition','trip_id','observation_index','timestamp_s','candidate_arc_id','reference_arc_id','reference_route_position','label','distance_to_segment_m','projection_fraction','segment_length_m','road_bearing_sin','road_bearing_cos','heading_difference_sin','heading_difference_cos','heading_valid','gps_speed_mps','gps_gap_s','candidate_rank']
CONFIG={'candidate_radius_m':60.0,'label_radius_m':100.0,'fallback_label_radius_m':300.0,'max_candidates':12,'max_label_candidates':256,'cell_m':100.0,'progress_weight':.35}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trips',type=Path,default=BACKEND/'data'/'trips')
    ap.add_argument('--out',type=Path,default=BACKEND/'data'/'candidates'/'student_test_candidate_features.csv.gz')
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.out,'wt',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for folder in sorted(p for p in a.trips.iterdir() if p.is_dir()):
            gps,network,details=trip_audit(folder,'student_test',True,writer,CONFIG)
            print(f"{folder.name}: gps={gps['gps_rows']} arcs={network['arcs']} candidate_rows={details['candidate_rows']} misses={details['candidate_misses']}",flush=True)
    print(f'Saved candidate feature table: {a.out.resolve()}')

if __name__=='__main__':main()
