"""
FERN v2 — Merge v1 clips into long videos per (subject, camera) group.

Directory structure expected:
    v1_dir/
        <person>/
            G{NN}_p{NN}_c{N}_r{NN}.mp4   ← labeled clips
        unlabeled_testing/
            <timestamp>.mp4               ← unlabeled (grouped by prefix)

This script:
  1. Groups clips by (subject, angle) extracted from filename G-code/p-code/c-code
  2. Concatenates each group into one long video via ffmpeg
  3. Writes a label JSON matching FERN v2 format

Output:
    output_dir/
        <subject>_<angle>.mp4
        labels/
            <subject>_<angle>.json

Usage
-----
python src/merge_v1_db.py
python src/merge_v1_db.py --v1_dir data/v1_clips --output_dir data/merged_v1 --gap_frames 5
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2

SUPPORTED = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}

# Map filename G-codes → FERN v2 gesture names.
V1_TO_V2 = {
    'G00': 'neutral',
    'G01': 'heeltap',
    'G02': 'forward_kick',
    'G03': 'foot_lift',
    'G04': 'lateral_slide',
    'G05': 'forward_step',
    'G06': 'cross_front',
    'G07': 'foot_hold',
    'G08': 'flamingo',
}

CLIP_RE = re.compile(r'^G(\d+)_p(\d+)_c(\d+)_r(\d+)')
UNLABELED_CAM_RE = re.compile(r'^(VID_|P\d{3})')


def get_video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames <= 0:
        return None
    return fps, frames, w, h


def concat_with_ffmpeg(clip_paths: list, output_path: str, target_fps: float):
    n = len(clip_paths)

    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json', str(clip_paths[0])
    ], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    w = info['streams'][0]['width']
    h = info['streams'][0]['height']

    inputs = []
    for p in clip_paths:
        inputs.extend(['-i', str(p)])

    scaled = ';'.join(
        f'[{i}:v]scale={w}:{h}:flags=bilinear,setsar=1[s{i}]'
        for i in range(n)
    )

    in_labels = ''.join(f'[s{i}]' for i in range(n))
    filter_str = f'{scaled};{in_labels}concat=n={n}:v=1:a=0[out]'

    cmd = [
        'ffmpeg', '-y',
        *inputs,
        '-filter_complex', filter_str,
        '-map', '[out]',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-r', f'{target_fps:.4f}',
        '-an',
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    ffmpeg error:\n{result.stderr[-800:]}')
        return False
    return True


def build_label_json(clip_info: list, output_video: str,
                     total_frames: int, fps: float) -> dict:
    """
    clip_info: list of {'gesture': str, 'frames': int, 'path': str}
    Returns a label dict compatible with FERN v2 dataset_v2.py.
    """
    segments   = []
    cursor     = 0

    for info in clip_info:
        gesture    = V1_TO_V2.get(info['gesture'], info['gesture'])
        n_frames   = info['frames']
        buf        = 3   # trim 3 frames from each end of the segment
        start      = cursor + buf
        end        = cursor + n_frames - buf - 1

        if end > start:
            segments.append({
                'gesture':     gesture,
                'start_frame': start,
                'end_frame':   end,
            })

        cursor += n_frames

    gesture_order = list(dict.fromkeys(
        V1_TO_V2.get(c['gesture'], c['gesture']) for c in clip_info))

    return {
        'video_path':       str(output_video),
        'fps':              fps,
        'total_frames':     total_frames,
        'gesture_order':    gesture_order,
        'reps_per_gesture': None,   # variable in v1
        'source':           'fern_v1_merged',
        'segments':         segments,
    }


def discover_clips(v1_dir: str):
    """
    Walk v1_dir/<person>/<clip> structure.

    Labeled clips follow: G{NN}_p{NN}_c{N}_r{NN}.mp4
      → gesture from G-code, subject from p-code, angle from c-code.

    unlabeled_testing/ clips by prefix:
      VID_* → angle='vid',  P\d+* → angle='p101',  other → angle='timestamp'
    """
    groups = {}
    v1_path = Path(v1_dir)

    for person_dir in sorted(v1_path.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name

        for clip in sorted(person_dir.glob('*')):
            if clip.suffix.lower() not in SUPPORTED:
                continue

            stem = clip.stem
            m = CLIP_RE.match(stem)
            if m:
                gesture = f'G{int(m.group(1)):02d}'
                subject = f'p{int(m.group(2)):02d}'
                angle = f'c{m.group(3)}'
            elif person_name == 'unlabeled_testing':
                gesture = 'unknown'
                subject = 'testing'
                cam_m = UNLABELED_CAM_RE.match(stem)
                if cam_m and cam_m.group(1):
                    angle = cam_m.group(1).rstrip('_').lower()
                else:
                    angle = 'timestamp'
            else:
                continue

            info = get_video_info(str(clip))
            if info is None:
                print(f'    SKIP corrupt: {clip.name}')
                continue
            fps, frames, w, h = info

            key = (subject, angle)
            groups.setdefault(key, []).append({
                'gesture': gesture,
                'path':    str(clip.resolve()),
                'frames':  frames,
                'fps':     fps,
                'w':       w,
                'h':       h,
            })

    return groups


def main():
    p = argparse.ArgumentParser(
        description='Merge FERN v1 clips into long videos with auto-labels.')
    p.add_argument('--v1_dir', default='data/v1_clips',
                   help='Root of v1 clip directory (default data/v1_clips).')
    p.add_argument('--output_dir', default='data/merged_v1',
                   help='Where to write merged videos + labels (default data/merged_v1).')
    p.add_argument('--gap_frames', type=int, default=5,
                   help='Black-frame gap inserted between clips (default 5).')
    args = p.parse_args()

    v1_dir = Path(args.v1_dir).resolve()
    out_video = Path(args.output_dir).resolve()
    out_label = out_video / 'labels'
    out_video.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    if not v1_dir.is_dir():
        print(f'v1_dir not found: {v1_dir}')
        return

    groups = discover_clips(str(v1_dir))

    if not groups:
        print('No clips found. Check --v1_dir and folder structure.')
        return

    print(f'Found {len(groups)} (subject, angle) groups.')

    gesture_order = list(V1_TO_V2.keys())

    for (subj, angle), clips in sorted(groups.items()):
        group_name = f'{subj}_{angle}'
        out_mp4    = out_video / f'{group_name}.mp4'
        out_json   = out_label / f'{group_name}.json'

        fps_vals = [c['fps'] for c in clips]
        target_fps = max(set(fps_vals), key=fps_vals.count)

        clips.sort(key=lambda c: (gesture_order.index(c['gesture'])
                                  if c['gesture'] in gesture_order else 999,
                                  c['path']))

        # Filter out clips that ffprobe considers invalid
        good_clips = []
        for c in clips:
            r = subprocess.run(
                ['ffprobe', '-v', 'error', c['path']],
                capture_output=True, text=True)
            if r.returncode == 0:
                good_clips.append(c)
            else:
                print(f'    SKIP (corrupt): {Path(c["path"]).name}')

        if not good_clips:
            print(f'  No valid clips in {group_name}, skipping.')
            continue

        print(f'\n  [{group_name}]  {len(good_clips)} clips  '
              f'fps={target_fps:.1f}')
        for c in good_clips:
            print(f'    {Path(c["path"]).name}  '
                  f'gesture={c["gesture"]}  frames={c["frames"]}')

        ok = concat_with_ffmpeg(
            [c['path'] for c in good_clips],
            str(out_mp4),
            target_fps,
        )

        if not ok:
            print(f'  FAILED to merge {group_name}')
            continue

        actual_info = get_video_info(str(out_mp4))
        if actual_info is None:
            print(f'  FAILED to read merged video {group_name}')
            continue
        _, actual_frames, _, _ = actual_info

        label = build_label_json(
            clips, str(out_mp4), actual_frames,
            target_fps,
        )

        with open(out_json, 'w') as f:
            json.dump(label, f, indent=2)

        n_segs = len(label['segments'])
        print(f'  -> {out_mp4.name}  ({actual_frames} frames, '
              f'{n_segs} labeled segments)')
        print(f'  -> {out_json.name}')

    print('\nAll groups done.')
    print('Next: run extract_skeleton.py on the merged videos,')
    print('then add them to your training set.')


if __name__ == '__main__':
    main()
