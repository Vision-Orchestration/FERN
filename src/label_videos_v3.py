"""
FERN v2 — Step 2: Label continuous gesture videos.

Single-pass mode: press S at gesture START, E at gesture END.
The tool records (start_frame, end_frame, gesture_label) pairs in real time
as you scrub through the video. No two-pass workflow.

Controls
--------
S      — mark START of current gesture  (green flash)
E      — mark END of current gesture    (red flash)
R      — undo last mark (removes most recent S or E)
A      — back 5 frames
D      — forward 5 frames
B      — back 50 frames
F      — forward 50 frames
W      — save and move to next video
Q      — quit without saving

Label file format (JSON):
{
    "video_path": "data/raw/session01.mp4",
    "fps": 30.0,
    "total_frames": 1150,
    "gesture_order": ["foot_lift", ...],
    "reps_per_gesture": 3,
    "segments": [
        {"gesture": "foot_lift", "start_frame": 12, "end_frame": 78},
        ...
    ]
}

Usage
-----
python src/label_videos.py \\
    --video_dir   data/raw \\
    --label_dir   data/labels \\
    --gestures    foot_lift sideway_kick cross_front heel_tap flamingo_bend forward_step forward_kick \\
    --reps        3

Auto mode (fixed-timing template):
python src/label_videos.py \\
    --mode      auto \\
    --video_dir data/raw \\
    --label_dir data/labels \\
    --template  data/labels/template.json
"""

import argparse
import json
import os
from pathlib import Path

import cv2

# ── overlay constants ──────────────────────────────────────────────────────────
COL_GREEN  = (0,   220,  50)    # start mark / idle state
COL_RED    = (0,    50, 220)    # end mark / inside-gesture state  (BGR)
COL_YELLOW = (0,   220, 220)    # time / frame counter
COL_GRAY   = (160, 160, 160)    # secondary info
COL_WHITE  = (230, 230, 230)
BAR_ALPHA  = 0.45               # background strip opacity
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(frame, text, pos, color, scale=0.72, thickness=2):
    """Draw text with a dark semi-transparent backing strip for legibility."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - 4,      y - th - 4),
                  (x + tw + 4, y + bl + 2),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, BAR_ALPHA, frame, 1 - BAR_ALPHA, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness,
                cv2.LINE_AA)


def _get_frame(cap, idx, total):
    """Seek to idx and return a display-ready frame (max 800px tall)."""
    idx = max(0, min(idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    if h > 800:
        scale  = 800 / h
        frame  = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


# ── state machine ──────────────────────────────────────────────────────────────
# We track a list of "events":  {'type': 'start'|'end', 'frame': int, 'gesture': str|None}
# A complete segment requires a start event followed by an end event for the same rep.

def _build_segments(events, sequence):
    """
    Pair start/end events into segments.
    Returns list of {'gesture', 'start_frame', 'end_frame'}.
    Incomplete pairs (start without end) are excluded.
    """
    starts = [e for e in events if e['type'] == 'start']
    ends   = [e for e in events if e['type'] == 'end']
    segments = []
    for i, s in enumerate(starts):
        if i < len(ends):
            e = ends[i]
            if e['frame'] > s['frame']:
                segments.append({
                    'gesture':     s['gesture'],
                    'start_frame': s['frame'],
                    'end_frame':   e['frame'],
                })
    return segments


def label_interactive(video_path, output_json, gestures, reps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build expected sequence: [g0,g0,g0, g1,g1,g1, ...]
    sequence = [g for g in gestures for _ in range(reps)]
    n_expected = len(sequence)   # total reps = len(gestures) * reps

    events   = []   # list of {type, frame, gesture}
    frame_idx = 0
    flash     = None   # ('start'|'end', countdown_frames)

    print(f"\n  Video   : {Path(video_path).name}")
    print(f"  FPS     : {fps:.1f}  |  Frames: {total}")
    print(f"  Expected: {n_expected} segments  ({len(gestures)} gestures × {reps} reps)")
    print(f"  Order   : {', '.join(gestures)}")
    print()
    print("  S=mark start  E=mark end  R=undo  A/D=±5f  B/F=±50f  W=save  Q=quit")
    print()

    while True:
        frame = _get_frame(cap, frame_idx, total)
        if frame is None:
            break

        # ── counts ──
        n_starts = sum(1 for e in events if e['type'] == 'start')
        n_ends   = sum(1 for e in events if e['type'] == 'end')
        # Next expected gesture label
        next_gesture = sequence[n_starts] if n_starts < n_expected else None
        inside_gesture = n_starts > n_ends   # started but not yet ended

        # ── flash overlay (brief color burst on mark) ──
        if flash:
            ftype, fcount = flash
            color = COL_GREEN if ftype == 'start' else COL_RED
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            flash = (ftype, fcount - 1) if fcount > 1 else None

        # ── HUD ──
        time_sec = frame_idx / fps

        # Line 1: what to do next
        if inside_gesture:
            cur_g = events[-1]['gesture'] if events else '?'
            status = f"INSIDE GESTURE: {cur_g}  |  press E to mark END"
            s_col  = COL_RED
        elif next_gesture:
            status = f"Seg {n_starts+1}/{n_expected}: press S at START of  [{next_gesture}]"
            s_col  = COL_GREEN
        else:
            status = f"All {n_expected} segments marked.  Press W to save."
            s_col  = COL_WHITE

        _put_text(frame, status,           (10,  32), s_col,   scale=0.70, thickness=2)
        _put_text(frame, f"Frame {frame_idx}  ({time_sec:.2f}s)",
                                           (10,  64), COL_YELLOW, scale=0.62, thickness=2)
        _put_text(frame, f"Starts: {n_starts}  Ends: {n_ends}  Complete segs: {min(n_starts, n_ends)}",
                                           (10,  92), COL_GRAY,   scale=0.55, thickness=1)

        if events:
            last = events[-1]
            last_txt = f"Last: [{last['type'].upper()}] frame {last['frame']}  {last['gesture'] or ''}"
            _put_text(frame, last_txt,     (10, 116), COL_GRAY,   scale=0.52, thickness=1)

        cv2.imshow("FERN v2 — Labeling", frame)
        key = cv2.waitKey(0) & 0xFF

        # ── key handling ──
        if key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

        elif key == ord('w') or key == ord('W'):
            break

        elif key == ord('s') or key == ord('S'):
            if inside_gesture:
                print("  Already inside a gesture — press E to end it first.")
            elif n_starts < n_expected:
                g = sequence[n_starts]
                events.append({'type': 'start', 'frame': frame_idx, 'gesture': g})
                flash = ('start', 4)
                print(f"  START  frame {frame_idx} ({time_sec:.2f}s)  [{g}]")
            else:
                print("  All starts recorded.")

        elif key == ord('e') or key == ord('E'):
            if not inside_gesture:
                print("  No open gesture — press S to start one first.")
            elif frame_idx <= events[-1]['frame']:
                print("  End frame must be after start frame.")
            else:
                g = events[-1]['gesture']
                events.append({'type': 'end', 'frame': frame_idx, 'gesture': g})
                flash = ('end', 4)
                print(f"  END    frame {frame_idx} ({time_sec:.2f}s)  [{g}]")

        elif key == ord('r') or key == ord('R'):
            if events:
                removed = events.pop()
                print(f"  Undone: [{removed['type'].upper()}] frame {removed['frame']} [{removed['gesture']}]")

        elif key == ord('a') or key == ord('A'):
            frame_idx = max(0, frame_idx - 5)

        elif key == ord('d') or key == ord('D'):
            frame_idx = min(total - 1, frame_idx + 5)

        elif key == ord('b') or key == ord('B'):
            frame_idx = max(0, frame_idx - 50)

        elif key == ord('f') or key == ord('F'):
            frame_idx = min(total - 1, frame_idx + 50)

    cap.release()
    cv2.destroyAllWindows()

    # ── build segments and save ──
    segments = _build_segments(events, sequence)

    if not segments:
        print("  No complete segments to save.")
        return False

    label_data = {
        'video_path':       str(video_path),
        'fps':              fps,
        'total_frames':     total,
        'gesture_order':    gestures,
        'reps_per_gesture': reps,
        'segments':         segments,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(label_data, f, indent=2)

    print(f"\n  Saved {len(segments)} segments → {output_json}")
    if len(segments) < n_expected:
        print(f"  WARNING: {n_expected - len(segments)} incomplete segment(s) skipped.")
    return True


# ── auto mode ──────────────────────────────────────────────────────────────────

def label_auto(video_path, output_json, template):
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    data = dict(template)
    data['video_path']   = str(video_path)
    data['fps']          = fps
    data['total_frames'] = total

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Auto-labeled: {Path(video_path).name}")
    return True


# ── main ───────────────────────────────────────────────────────────────────────

SUPPORTED = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}

DEFAULT_GESTURES = [
    'foot_lift',
    'sideway_kick',
    'cross_front',
    'heel_tap',
    'flamingo_bend',
    'forward_step',
    'forward_kick',
]


def main():
    p = argparse.ArgumentParser(
        description='Label continuous gesture videos for FERN v2.')
    p.add_argument('--mode',      choices=['interactive', 'auto'],
                   default='interactive')
    p.add_argument('--video_dir', required=True)
    p.add_argument('--label_dir', required=True)
    p.add_argument('--gestures',  nargs='+', default=DEFAULT_GESTURES)
    p.add_argument('--reps',      type=int,  default=3)
    p.add_argument('--template',  default=None)
    args = p.parse_args()

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED:
                videos.append(os.path.join(root, fname))

    if not videos:
        print(f'No videos found in {args.video_dir}')
        return

    print(f'Found {len(videos)} video(s).  Mode: {args.mode}')

    template = None
    if args.mode == 'auto':
        if not args.template:
            print('ERROR: --template required for auto mode.')
            return
        with open(args.template) as f:
            template = json.load(f)

    for vpath in videos:
        rel       = os.path.relpath(vpath, args.video_dir)
        json_path = os.path.join(args.label_dir,
                                 str(Path(rel).with_suffix('.json')))

        if args.mode == 'interactive':
            label_interactive(vpath, json_path, args.gestures, args.reps)
        else:
            label_auto(vpath, json_path, template)

    print('\nDone.')


if __name__ == '__main__':
    main()
