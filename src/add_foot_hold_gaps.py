"""Insert 60-frame foot_hold gaps at gesture transitions where gap is too small."""

import argparse
import json
import math
from pathlib import Path

MIN_SEGMENT = 20
TARGET_GAP = 60


def total_foot_hold_frames(segments):
    return sum(
        s["end_frame"] - s["start_frame"] + 1
        for s in segments
        if s["gesture"] == "foot_hold"
    )


def add_foot_hold_gaps(label_path, min_segment=MIN_SEGMENT, target_gap=TARGET_GAP):
    with open(label_path, encoding="utf-8") as f:
        label = json.load(f)

    if label.get("foot_hold_gaps_added_by"):
        return -1  # already processed

    segments = label.get("segments", [])
    if not segments:
        return False

    if total_foot_hold_frames(segments) >= target_gap:
        return False

    result = []

    for i, seg in enumerate(segments):
        gesture, start, end = seg["gesture"], seg["start_frame"], seg["end_frame"]

        if i > 0 and segments[i - 1]["gesture"] != gesture:
            prev = result.pop()
            gap = start - prev["end_frame"]

            prev_len = prev["end_frame"] - prev["start_frame"] + 1
            curr_len = end - start + 1
            max_trim_prev = prev_len - min_segment
            max_trim_curr = curr_len - min_segment

            if gap >= target_gap:
                fh_start = prev["end_frame"] + 1
                fh_end = start - 1
                result.append(prev)
                if fh_end >= fh_start:
                    result.append({
                        "gesture": "foot_hold",
                        "start_frame": fh_start,
                        "end_frame": fh_end,
                    })
                continue

            if max_trim_prev > 0 and max_trim_curr > 0:
                needed = target_gap + 1 - gap
                trim_prev = min(int(math.ceil(needed / 2)), max_trim_prev)
                trim_curr = min(needed - trim_prev, max_trim_curr)

                total_trim = gap + trim_prev + trim_curr - 1
                if total_trim >= target_gap:
                    prev["end_frame"] -= trim_prev
                    fh_start = prev["end_frame"] + 1
                    new_start = start + trim_curr
                    fh_end = new_start - 1

                    result.append(prev)
                    if fh_end >= fh_start:
                        result.append({
                            "gesture": "foot_hold",
                            "start_frame": fh_start,
                            "end_frame": fh_end,
                        })
                    start = new_start
                else:
                    result.append(prev)
            else:
                result.append(prev)

        result.append({
            "gesture": gesture,
            "start_frame": start,
            "end_frame": end,
        })

    label["segments"] = result
    label["foot_hold_gaps_added_by"] = "add_foot_hold_gaps.py"

    inserted = total_foot_hold_frames(result) - total_foot_hold_frames(segments)
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2, ensure_ascii=False)

    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True,
                    help="e.g. data/labels/front_plus_45")
    ap.add_argument("--min_segment", type=int, default=20)
    ap.add_argument("--target_gap", type=int, default=60)
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    jsons = sorted(label_dir.glob("*.json"))
    total_inserted = 0
    modified = 0
    skipped = 0

    for p in jsons:
        inserted = add_foot_hold_gaps(p, args.min_segment, args.target_gap)
        if inserted is False:
            continue
        if inserted == -1:
            skipped += 1
            continue
        if inserted > 0:
            print(f"  {p.stem}: +{inserted} foot_hold frames")
            total_inserted += inserted
            modified += 1
        else:
            print(f"  {p.stem}: already has >=60 foot_hold frames")

    print(f"\nModified {modified}/{len(jsons)} files  (skipped {skipped} already processed)")
    print(f"Total foot_hold frames inserted: {total_inserted}")


if __name__ == "__main__":
    main()
