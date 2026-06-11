"""
Repair corrupted merged_v1 label JSONs where segments have end_frame < start_frame.
Rebuilds segments sequentially: each end_frame = next segment's start_frame - 1
(or total_frames - 1 for the last segment).
"""
import json, glob, os

label_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'labels', 'merged_v1')

repaired = 0
for fpath in sorted(glob.glob(os.path.join(label_dir, '*.json'))):
    with open(fpath) as f:
        data = json.load(f)
    segs = data.get('segments', [])
    total = data.get('total_frames', 0)

    # Check for corruption
    needs_fix = any(s['start_frame'] > s['end_frame'] for s in segs)
    if not needs_fix:
        continue

    # Rebuild: sort by start_frame, then fix sequential boundaries
    segs.sort(key=lambda s: s['start_frame'])
    fixed = []
    for i, s in enumerate(segs):
        start = s['start_frame']
        # End is next segment's start - 1, or total_frames - 1
        if i + 1 < len(segs):
            nxt_start = segs[i + 1]['start_frame']
            end = nxt_start - 1
        else:
            end = total - 1
        if end > start:
            s['end_frame'] = end
        else:
            s['end_frame'] = start + max(30, (total - start) // 10)
        fixed.append(s)

    data['segments'] = fixed
    data['repaired_by'] = 'repair_labels.py'

    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)

    valid = sum(1 for s in data['segments'] if s['start_frame'] <= s['end_frame'])
    print(f'Repaired: {os.path.basename(fpath)} ({valid} valid segments)')
    repaired += 1

print(f'\nRepaired {repaired} file(s).')

# Re-validate all
bad = 0
for fpath in sorted(glob.glob(os.path.join(label_dir, '*.json'))):
    with open(fpath) as f:
        data = json.load(f)
    for s in data.get('segments', []):
        if s['start_frame'] > s['end_frame']:
            print(f'  STILL CORRUPT: {os.path.basename(fpath)}: {s}')
            bad += 1
print(f'Remaining corrupted: {bad}')
