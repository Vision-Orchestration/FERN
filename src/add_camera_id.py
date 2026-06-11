"""Step 3: Add camera_id to all label JSONs."""
import json
from pathlib import Path

# c3 front labels -> camera_id 0
front_dir = Path("data/labels/front")
updated = 0
for f in sorted(front_dir.glob("*.json")):
    with open(f) as fh:
        data = json.load(fh)
    if "camera_id" not in data:
        data["camera_id"] = 0
        with open(f, "w") as fh:
            json.dump(data, fh, indent=2)
        updated += 1
total_front = len(list(front_dir.glob("*.json")))
print(f"  front: {updated}/{total_front} updated (camera_id=0)")

# front_plus_45 labels: c3->0, c2->1
p45_dir = Path("data/labels/front_plus_45")
updated_c3 = 0
updated_c2 = 0
for f in sorted(p45_dir.glob("*.json")):
    with open(f) as fh:
        data = json.load(fh)
    stem = f.stem.replace("_mirror", "")
    if "c2" in stem:
        target = 1
    else:
        target = 0
    if data.get("camera_id") != target:
        data["camera_id"] = target
        with open(f, "w") as fh:
            json.dump(data, fh, indent=2)
        if target == 0:
            updated_c3 += 1
        else:
            updated_c2 += 1

all_jsons = list(p45_dir.glob("*.json"))
c3_count = len([f for f in all_jsons if "c2" not in f.stem.replace("_mirror", "")])
c2_count = len([f for f in all_jsons if "c2" in f.stem.replace("_mirror", "")])
print(f"  front_plus_45 c3: {updated_c3}/{c3_count} updated (camera_id=0)")
print(f"  front_plus_45 c2: {updated_c2}/{c2_count} updated (camera_id=1)")

# Verify samples
print()
for label in ["data/labels/front", "data/labels/front_plus_45"]:
    files = sorted(Path(label).glob("*.json"))
    if files:
        with open(files[0]) as fh:
            d = json.load(fh)
        print(f"  Sample {files[0].name}: camera_id={d.get('camera_id')}")
    c2f = [f for f in files if "c2" in f.stem]
    if c2f:
        with open(c2f[0]) as fh:
            d = json.load(fh)
        print(f"  Sample c2 {c2f[0].name}: camera_id={d.get('camera_id')}")
