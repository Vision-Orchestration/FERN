# Auto-activate FERN_V2 venv if not already active
import sys as _sys, os as _os
_venv_python = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'venv', 'Scripts', 'python.exe')
if _sys.prefix == _sys.base_prefix and _os.path.isfile(_venv_python):
    _os.execv(_venv_python, [_venv_python] + _sys.argv)

"""
FERN v2 — Label continuous gesture videos (flexible single-pass mode).

Controls
--------
1-7    — select gesture (1=foot_lift .. 7=forward_kick)
H      — select foot_hold as current gesture
S      — mark START of selected gesture   (green flash)
E      — mark END of selected gesture     (red flash)
R      — undo last mark
Z      — delete a completed segment (cycle through with arrow keys)
A/D    — back / forward 5 frames
B/F    — back / forward 50 frames
W      — save and move to next video
Q      — quit without saving

Usage
-----
python src/label_videos_v3.py \\
    --video_dir   "data/raw videos/front" \\
    --label_dir   data/labels/raw_front

python src/label_videos_v3.py \\
    --video_dir   "data/raw videos/45 from right" \\
    --label_dir   data/labels/raw_45
"""

import argparse
import json
import os
from pathlib import Path

import cv2

# ── gesture map ───────────────────────────────────────────────────────────────
GESTURES = [
    'foot_lift',       # 1
    'sideway_kick',    # 2
    'cross_front',     # 3
    'heel_tap',        # 4
    'flamingo_bend',   # 5
    'forward_step',    # 6
    'forward_kick',    # 7
]
GESTURE_KEYS = {str(i + 1): g for i, g in enumerate(GESTURES)}

# ── overlay constants ──────────────────────────────────────────────────────────
COL_GREEN  = (0,   220,  50)
COL_RED    = (0,    50, 220)
COL_YELLOW = (0,   220, 220)
COL_GRAY   = (160, 160, 160)
COL_WHITE  = (230, 230, 230)
COL_CYAN   = (220, 220,   0)
COL_MAGENTA = (180,  50, 180)
BAR_ALPHA  = 0.45
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(frame, text, pos, color, scale=0.72, thickness=2):
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
    idx = max(0, min(idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    if h > 800:
        scale = 800 / h
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def _build_segments(events):
    """Pair consecutive start/end events into segments."""
    segments = []
    i = 0
    while i < len(events):
        if events[i]['type'] == 'start':
            # find next end
            j = i + 1
            while j < len(events) and events[j]['type'] == 'start':
                j += 1
            if j < len(events) and events[j]['type'] == 'end':
                if events[j]['frame'] > events[i]['frame']:
                    segments.append({
                        'gesture':     events[i]['gesture'],
                        'start_frame': events[i]['frame'],
                        'end_frame':   events[j]['frame'],
                    })
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return segments


def _segment_summary(segments):
    """Return a short summary string of all segments."""
    if not segments:
        return "  (none)"
    lines = []
    for i, seg in enumerate(segments):
        g = seg['gesture']
        s, e = seg['start_frame'], seg['end_frame']
        lines.append(f"    {i+1:>2}. {g:<16}  f{s}-{e}  ({e-s} frames)")
    return "\n".join(lines)


def _gesture_color(gesture):
    """Return a color for the current gesture label."""
    if gesture == 'foot_hold':
        return COL_CYAN
    idx = GESTURES.index(gesture) if gesture in GESTURES else -1
    colors = [
        (0, 200, 255),   # foot_lift - orange
        (255, 100, 0),   # sideway_kick - blue
        (0, 255, 100),   # cross_front - green
        (100, 255, 255), # heel_tap - light blue
        (255, 0, 200),   # flamingo_bend - pink
        (0, 180, 255),   # forward_step - orange-yellow
        (255, 255, 0),   # forward_kick - cyan
    ]
    return colors[idx] if 0 <= idx < len(colors) else COL_WHITE


def label_interactive(video_path, output_json):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_gesture = GESTURES[0]   # default: foot_lift (key 1)
    events   = []                    # list of {type, frame, gesture}
    frame_idx = 0
    flash     = None                 # ('start'|'end', countdown)

    # segment deletion state
    delete_mode   = False
    delete_idx    = 0
    preview_segs  = []

    print(f"\n  Video   : {Path(video_path).name}")
    print(f"  FPS     : {fps:.1f}  |  Frames: {total}")
    print()
    print("  1-7=select gesture  H=foot_hold  S=start  E=end  R=undo")
    print("  Z=delete segment  A/D=±5f  B/F=±50f  W=save  Q=quit")
    print()

    while True:
        frame = _get_frame(cap, frame_idx, total)
        if frame is None:
            break

        n_starts = sum(1 for e in events if e['type'] == 'start')
        n_ends   = sum(1 for e in events if e['type'] == 'end')
        segments = _build_segments(events)
        inside_gesture = n_starts > n_ends

        # ── flash overlay ──
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
        g_col = _gesture_color(current_gesture)

        if delete_mode:
            status = f"DELETE mode: segment {delete_idx+1}/{len(preview_segs)}  " \
                     f"[ENTER]=confirm  [ESC]=cancel  [←→]=cycle"
            s_col = COL_MAGENTA
        elif inside_gesture:
            cur_g = events[-1]['gesture'] if events else '?'
            status = f"INSIDE: {cur_g}  |  press E to mark END"
            s_col = COL_RED
        else:
            status = f"Selected: [{current_gesture.upper()}]  |  press S to mark START"
            s_col = g_col

        _put_text(frame, status, (10, 32), s_col, scale=0.68, thickness=2)
        _put_text(frame, f"Frame {frame_idx}  ({time_sec:.2f}s)",
                                       (10, 62), COL_YELLOW, scale=0.60, thickness=2)
        _put_text(frame, f"Segments: {len(segments)}  |  Starts: {n_starts}  Ends: {n_ends}",
                                       (10, 88), COL_GRAY, scale=0.52, thickness=1)

        # gesture palette at bottom
        y_pal = frame.shape[0] - 10
        x_pal = 10
        for i, g in enumerate(GESTURES):
            label = f"{i+1}:{g}"
            col = g_col if g == current_gesture else COL_GRAY
            thk = 2 if g == current_gesture else 1
            _put_text(frame, label, (x_pal, y_pal - (6 - i) * 22), col, scale=0.45, thickness=thk)
        _put_text(frame, f"H:foot_hold", (x_pal, y_pal), COL_CYAN if current_gesture == 'foot_hold' else COL_GRAY,
                  scale=0.45, thickness=2 if current_gesture == 'foot_hold' else 1)

        # segment list on right side
        if segments:
            x_seg = frame.shape[1] - 280
            _put_text(frame, "Segments:", (x_seg, 32), COL_WHITE, scale=0.50, thickness=1)
            for i, seg in enumerate(segments):
                y = 56 + i * 18
                if y > frame.shape[0] - 20:
                    _put_text(frame, f"  ... +{len(segments) - i} more", (x_seg, y), COL_GRAY, scale=0.42, thickness=1)
                    break
                g = seg['gesture']
                col = _gesture_color(g)
                txt = f"  {i+1:>2}. {g}"
                _put_text(frame, txt, (x_seg, y), col, scale=0.42, thickness=1)

        # delete preview overlay
        if delete_mode and preview_segs:
            seg = preview_segs[delete_idx]
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 180), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            _put_text(frame,
                      f"DELETE: {seg['gesture']}  f{seg['start_frame']}-{seg['end_frame']}",
                      (10, h - 25), COL_WHITE, scale=0.65, thickness=2)

        cv2.imshow("FERN v2 — Labeling", frame)
        key = cv2.waitKey(0) & 0xFF

        # ── key handling ──
        if delete_mode:
            if key == 13:  # Enter
                # confirm delete
                seg = preview_segs[delete_idx]
                # remove matching events
                new_events = []
                skip_start = False
                skip_end = False
                for e in events:
                    if (e['type'] == 'start' and e['frame'] == seg['start_frame']
                            and e['gesture'] == seg['gesture'] and not skip_start):
                        skip_start = True
                        continue
                    if (e['type'] == 'end' and e['frame'] == seg['end_frame']
                            and e['gesture'] == seg['gesture'] and not skip_end):
                        skip_end = True
                        continue
                    new_events.append(e)
                events = new_events
                print(f"  Deleted segment: {seg['gesture']} f{seg['start_frame']}-{seg['end_frame']}")
                delete_mode = False
                preview_segs = []
            elif key == 27:  # Escape
                delete_mode = False
                preview_segs = []
            elif key == ord('d') or key == ord('D') or key == 83:  # right arrow
                delete_idx = (delete_idx + 1) % len(preview_segs)
            elif key == ord('a') or key == ord('A') or key == 81:  # left arrow
                delete_idx = (delete_idx - 1) % len(preview_segs)
            continue

        if key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

        elif key == ord('w') or key == ord('W'):
            break

        # gesture selection
        elif chr(key) in GESTURE_KEYS:
            if not inside_gesture:
                current_gesture = GESTURE_KEYS[chr(key)]
                print(f"  Selected: {current_gesture}")

        elif key == ord('h') or key == ord('H'):
            if not inside_gesture:
                current_gesture = 'foot_hold'
                print(f"  Selected: foot_hold")

        # start / end marks
        elif key == ord('s') or key == ord('S'):
            if inside_gesture:
                print("  Already inside a gesture — press E to end it first.")
            else:
                events.append({'type': 'start', 'frame': frame_idx, 'gesture': current_gesture})
                flash = ('start', 4)
                print(f"  START  frame {frame_idx} ({time_sec:.2f}s)  [{current_gesture}]")

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

        # delete mode
        elif key == ord('z') or key == ord('Z'):
            segments = _build_segments(events)
            if segments:
                delete_mode = True
                preview_segs = segments
                delete_idx = len(segments) - 1  # start at last
            else:
                print("  No segments to delete.")

        # navigation
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

    # ── build and save ──
    segments = _build_segments(events)

    if not segments:
        print("  No segments to save.")
        return False

    label_data = {
        'video_path':   str(video_path),
        'fps':          fps,
        'total_frames': total,
        'segments':     segments,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(label_data, f, indent=2)

    print(f"\n  Saved {len(segments)} segments → {output_json}")
    print(_segment_summary(segments))
    return True


# ── main ───────────────────────────────────────────────────────────────────────

SUPPORTED = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}


def main():
    p = argparse.ArgumentParser(
        description='Label continuous gesture videos for FERN v2.')
    p.add_argument('--video_dir', required=True)
    p.add_argument('--label_dir', required=True)
    args = p.parse_args()

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED:
                videos.append(os.path.join(root, fname))

    if not videos:
        print(f'No videos found in {args.video_dir}')
        return

    print(f'Found {len(videos)} video(s).  Mode: interactive')

    for vpath in videos:
        rel       = os.path.relpath(vpath, args.video_dir)
        json_path = os.path.join(args.label_dir,
                                 str(Path(rel).with_suffix('.json')))
        label_interactive(vpath, json_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
