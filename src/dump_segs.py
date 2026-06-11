import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
segs = d['segments']
print(f'Total: {len(segs)} segments')
corrupt = 0
for s in segs:
    ok = 'OK' if s['start_frame'] <= s['end_frame'] else 'CORRUPT'
    if ok != 'OK': corrupt += 1
    print(f'  [{ok}] {s["gesture"]:15s}  {s["start_frame"]:5d} - {s["end_frame"]:5d}')
print(f'\nCorrupted: {corrupt}')
