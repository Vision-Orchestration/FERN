import json, glob, os
label_dir = '../data/labels/merged_v1'
total_corrupt = 0
corrupt_files = []
for f in sorted(glob.glob(os.path.join(label_dir, '*.json'))):
    with open(f) as jf:
        data = json.load(jf)
    segs = data.get('segments', [])
    for s in segs:
        if s['start_frame'] > s['end_frame']:
            total_corrupt += 1
            corrupt_files.append(os.path.basename(f))
            break
print(f'Corrupted files: {len(set(corrupt_files))}')
for cf in sorted(set(corrupt_files)):
    print(f'  {cf}')
print(f'Total corrupted segments: {total_corrupt}')
