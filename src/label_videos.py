"""
FERN v2 — Step 2: Create timestamp label files for continuous videos.

Since your 1-minute videos follow a fixed structure (each gesture repeated
3 times in a known order), you do NOT need to label every frame by hand.
You only need to mark the START time of each gesture block.

This script provides two modes:

  --mode interactive
      Opens each video in a window.  You press SPACE to mark the current
      timestamp.  The script records 21 marks (7 gestures x 3 repetitions)
      then saves a JSON label file.

  --mode auto
      If all your recordings have the exact same timing structure (e.g.
      gesture 1 starts at t=0, ends at t=9s; gesture 2 starts at t=10s,
      etc.), you can pass a template JSON and it will be copied to all
      videos without any manual work.

Label file format (JSON):
{
    "video_path": "data/raw/session01.mp4",
    "fps": 30.0,
    "gesture_order": ["idle", "tap_left", "tap_right", ...],
    "segments": [
        {"gesture": "idle",     "start_frame": 0,   "end_frame": 89},
        {"gesture": "idle",     "start_frame": 90,  "end_frame": 179},
        ...
    ]
}

Usage — interactive
-------------------
python src/label_videos.py \
    --mode        interactive \
    --video_dir   data/raw \
    --label_dir   data/labels \
    --gestures    idle tap_left tap_right swipe_left swipe_right stomp heel_raise cross_step \
    --reps        3

Usage — auto (fixed timing)
---------------------------
python src/label_videos.py \
    --mode        auto \
    --video_dir   data/raw \
    --label_dir   data/labels \
    --template    data/labels/template.json
"""

import argparse
import json
import os
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Interactive labeling
# ---------------------------------------------------------------------------

def label_interactive(video_path: str, output_json: str,
                      gestures: list, reps: int):
    """
    Open a video, let the user press SPACE at the start of each gesture,
    and save the resulting label file.

    Controls
    --------
    SPACE  — mark current frame as start of next segment
    LEFT   — rewind 30 frames
    RIGHT  — advance 30 frames
    R      — redo last mark (delete it)
    Q      — quit and discard
    S      — save and close
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return False

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx   = 0
    marks       = []   # list of (frame_idx, gesture_name)

    # Build the sequence of expected marks.
    # e.g. reps=3, gestures=[idle, tap_left, ...] =>
    #   [idle, idle, idle, tap_left, tap_left, ...]
    sequence = []
    for g in gestures:
        for _ in range(reps):
            sequence.append(g)

    print(f"\n  Video  : {Path(video_path).name}")
    print(f"  FPS    : {fps:.1f}   |   Frames: {total}")
    print(f"  Marks needed: {len(sequence)}")
    print(f"  Expected order: {', '.join(gestures)} (x{reps} each)")
    print()
    print("  Controls: SPACE=mark  LEFT/RIGHT=seek  R=undo  S=save  Q=quit")
    print()

    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ret, frame = cap.read()
        if not ret:
            return None
        h, w = frame.shape[:2]
        max_h = 800
        if h > max_h:
            scale = max_h / h
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    while True:
        frame = get_frame(frame_idx)
        if frame is None:
            break

        # Build overlay text.
        mark_count = len(marks)
        if mark_count < len(sequence):
            next_label = sequence[mark_count]
            status_txt = f"Mark {mark_count+1}/{len(sequence)}: {next_label}"
        else:
            status_txt = "All marks done — press S to save"

        time_sec = frame_idx / fps
        time_txt = f"Frame {frame_idx}  ({time_sec:.2f}s)"

        cv2.putText(frame, status_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, time_txt,   (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if marks:
            last = marks[-1]
            last_txt = f"Last mark: frame {last[0]} = '{last[1]}'"
            cv2.putText(frame, last_txt, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("FERN v2 — Labeling", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return False

        elif key == ord("s"):
            break

        elif key == ord(" "):  # SPACE = mark
            if len(marks) < len(sequence):
                label = sequence[len(marks)]
                marks.append((frame_idx, label))
                print(f"  Marked frame {frame_idx} ({time_sec:.2f}s) = '{label}'")
            else:
                print("  All marks already recorded.  Press S to save.")

        elif key == ord("r"):  # undo
            if marks:
                removed = marks.pop()
                print(f"  Removed mark: frame {removed[0]} = '{removed[1]}'")

        elif key == 81 or key == 2 or key == ord('a') or key == ord('A'):
            frame_idx = max(0, frame_idx - 5)

        elif key == 83 or key == 3 or key == ord('d') or key == ord('D'):
            frame_idx = min(total - 1, frame_idx + 5)
            
        elif key == ord('f') or key == ord('F'):
            frame_idx = min(total - 1, frame_idx + 200)

        elif key == ord('b') or key == ord('B'):
            frame_idx = max(0, frame_idx - 200)

    cap.release()
    cv2.destroyAllWindows()

    # Build segments from marks.
    # Each segment spans from its start mark to the next mark (exclusive).
    segments = []
    for i, (start_frame, label) in enumerate(marks):
        end_frame = marks[i + 1][0] - 1 if i + 1 < len(marks) else total - 1
        segments.append({
            "gesture":     label,
            "start_frame": start_frame,
            "end_frame":   end_frame,
        })

    label_data = {
        "video_path":    str(video_path),
        "fps":           fps,
        "total_frames":  total,
        "gesture_order": gestures,
        "reps_per_gesture": reps,
        "segments":      segments,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"  Saved {len(segments)} segments to {output_json}")
    return True


# ---------------------------------------------------------------------------
# Auto labeling from a template
# ---------------------------------------------------------------------------

def label_auto(video_path: str, output_json: str, template: dict):
    """
    Copy a fixed timing template to a video.
    Useful when all recordings have the exact same structure.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    label_data = dict(template)
    label_data["video_path"]   = str(video_path)
    label_data["fps"]          = fps
    label_data["total_frames"] = total

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"  Auto-labeled: {Path(video_path).name} -> {output_json}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}
DEFAULT_GESTURES = [
    "foot_lift",
    "sideway_kick",
    "cross_front",
    "heel_tap",
    "flamingo_bend",
    "forward_step",
    "forward_kick",
]


def main():
    parser = argparse.ArgumentParser(
        description="Create timestamp label files for continuous gesture videos."
    )
    parser.add_argument("--mode",      choices=["interactive", "auto"],
                        default="interactive")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--gestures",  nargs="+", default=DEFAULT_GESTURES)
    parser.add_argument("--reps",      type=int, default=3)
    parser.add_argument("--template",  default=None,
                        help="Path to template JSON (auto mode only).")
    args = parser.parse_args()

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                videos.append(os.path.join(root, fname))

    if not videos:
        print(f"No videos found in {args.video_dir}")
        return

    print(f"Found {len(videos)} video(s).  Mode: {args.mode}")

    template = None
    if args.mode == "auto":
        if not args.template:
            print("ERROR: --template is required for auto mode.")
            return
        with open(args.template) as f:
            template = json.load(f)

    for vpath in videos:
        rel        = os.path.relpath(vpath, args.video_dir)
        json_rel   = str(Path(rel).with_suffix(".json"))
        json_path  = os.path.join(args.label_dir, json_rel)

        if args.mode == "interactive":
            label_interactive(vpath, json_path, args.gestures, args.reps)
        else:
            label_auto(vpath, json_path, template)

    print("\nDone.")


if __name__ == "__main__":
    main()
