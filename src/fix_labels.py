"""
Fix label JSON inconsistencies:
  1. Map neutral -> foot_hold (idle class) in segments + gesture_order
  2. Fix gesture_order names: heeltap->heel_tap, lateral_slide->sideway_kick, flamingo->flamingo_bend
  3. Remove duplicate foot_hold entries in gesture_order (neutral+foot_hold -> both foot_hold)
"""

import json
import os
import glob

OLD_TO_NEW = {
    "heeltap": "heel_tap",
    "lateral_slide": "sideway_kick",
    "flamingo": "flamingo_bend",
    "neutral": "foot_hold",
}

def fix_json(path):
    with open(path) as f:
        data = json.load(f)

    changed = False

    # Fix gesture_order
    if "gesture_order" in data:
        new_order = []
        for g in data["gesture_order"]:
            mapped = OLD_TO_NEW.get(g, g)
            if not new_order or new_order[-1] != mapped:  # deduplicate consecutive
                new_order.append(mapped)
        if new_order != data["gesture_order"]:
            data["gesture_order"] = new_order
            changed = True

    # Fix segments
    if "segments" in data:
        for seg in data["segments"]:
            old = seg.get("gesture")
            if old in OLD_TO_NEW:
                seg["gesture"] = OLD_TO_NEW[old]
                changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Fixed: {os.path.basename(path)}")
        return True
    return False


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for label_dir in ["data/labels/merged_v1", "data/labels/grouped_by_gesture"]:
        full = os.path.join(base, label_dir)
        if not os.path.isdir(full):
            print(f"Skipping {label_dir} (not found)")
            continue

        print(f"\n--- {label_dir} ---")
        count = 0
        for fpath in sorted(glob.glob(os.path.join(full, "*.json"))):
            if fix_json(fpath):
                count += 1
        print(f"  {count} file(s) updated.")
