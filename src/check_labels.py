import json, glob, os
bad = 0
label_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'labels', 'merged_v1')
for f in sorted(glob.glob(os.path.join(label_dir, '*.json'))):
    with open(f) as jf:
        data = json.load(jf)
    segs = data.get('segments', [])
    print(f'\n--- {os.path.basename(f)} ---')
    for i, s in enumerate(segs):
        g = s.get('gesture', '?')
        print(f'  [{i:2d}] {g:15s}  {s["start_frame"]:5d} - {s["end_frame"]:5d}  ({s["end_frame"]-s["start_frame"]:4d} fr)')
        if s['start_frame'] > s['end_frame']:
            print(f'  *** CORRUPTED ***')
            bad += 1
    for i in range(len(segs)-1):
        gap = segs[i+1]['start_frame'] - segs[i]['end_frame']
        if gap < 0:
            print(f'  *** OVERLAP seg[{i}] -> [{i+1}]: {segs[i]["end_frame"]} > {segs[i+1]["start_frame"]} ***')
            bad += 1
print(f'\nTotal corrupted: {bad}')
