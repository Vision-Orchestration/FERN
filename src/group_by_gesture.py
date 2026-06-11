"""
Group v1 clips by gesture — all persons/cameras merged per gesture.

Output:
    output_dir/
        heeltap.mp4              ← all G01 clips from every person/camera
        forward_kick.mp4         ← all G02 clips
        ...
        labels/
            heeltap.json
            forward_kick.json
            ...

Usage:
    python src/group_by_gesture.py
    python src/group_by_gesture.py --v1_dir data/v1_clips --output_dir data/grouped_by_gesture
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2

SUPPORTED = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}

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


def concat_chunk(clip_paths: list, output_path: Path,
                 target_fps: float, w: int, h: int) -> bool:
    """Concat a small chunk of clips using concat filter with full normalization."""
    n = len(clip_paths)
    inputs = []
    for p in clip_paths:
        inputs.extend(['-i', str(p)])

    scaled = ';'.join(
        f'[{i}:v]settb=1/{target_fps:.4f},setpts=PTS-STARTPTS,'
        f'scale={w}:{h}:flags=bilinear,setsar=1,fps={target_fps:.4f},format=yuv420p[s{i}]'
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
        '-an',
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    ffmpeg error:\n{result.stderr[-800:]}')
        return False
    return True


def concat_with_ffmpeg(clip_paths: list, output_path: str, target_fps: float):
    CHUNK_SIZE = 25
    out = Path(output_path)
    tmp_dir = out.parent / '__tmp_concat'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json', str(clip_paths[0])
    ], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    w = info['streams'][0]['width']
    h = info['streams'][0]['height']

    n = len(clip_paths)

    if n <= CHUNK_SIZE:
        ok = concat_chunk(clip_paths, out, target_fps, w, h)
        return ok

    partials: list[Path] = []
    for i in range(0, n, CHUNK_SIZE):
        chunk = clip_paths[i:i + CHUNK_SIZE]
        part = tmp_dir / f'part_{i // CHUNK_SIZE}.mp4'
        print(f'    concat chunk {i // CHUNK_SIZE + 1}/{(n + CHUNK_SIZE - 1) // CHUNK_SIZE} '
              f'({len(chunk)} clips)')
        ok = concat_chunk(chunk, part, target_fps, w, h)

        if not ok:
            for p in partials:
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
            return False
        partials.append(part)

    with tempfile.NamedTemporaryFile('w', suffix='.txt',
                                     delete=False) as tf:
        list_path = tf.name
        for p in partials:
            safe = str(p).replace('\\', '/').replace("'", "\\'")
            tf.write(f"file '{safe}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', list_path,
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-r', f'{target_fps:.4f}',
        '-an',
        str(out),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(list_path)

    for p in partials:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()

    if result.returncode != 0:
        print(f'    ffmpeg error:\n{result.stderr[-800:]}')
        return False
    return True


def build_label_json(clip_info: list, output_video: str,
                     total_frames: int, fps: float) -> dict:
    segments = []
    cursor = 0

    for info in clip_info:
        gesture = V1_TO_V2.get(info['gesture'], info['gesture'])
        n_frames = info['frames']
        buf = 3
        start = cursor + buf
        end = cursor + n_frames - buf - 1

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
        'reps_per_gesture': None,
        'source':           'fern_v1_grouped_by_gesture',
        'segments':         segments,
    }


def discover_clips(v1_dir: str):
    groups = {}
    v1_path = Path(v1_dir)

    for person_dir in sorted(v1_path.iterdir()):
        if not person_dir.is_dir():
            continue

        for clip in sorted(person_dir.glob('*')):
            if clip.suffix.lower() not in SUPPORTED:
                continue

            stem = clip.stem
            m = CLIP_RE.match(stem)
            if m:
                gesture = f'G{int(m.group(1)):02d}'
                subject = f'p{int(m.group(2)):02d}'
                angle = f'c{m.group(3)}'
            elif person_dir.name == 'unlabeled_testing':
                gesture = 'unknown'
                subject = 'testing'
                cam_m = UNLABELED_CAM_RE.match(stem)
                angle = cam_m.group(1).rstrip('_').lower() if cam_m and cam_m.group(1) else 'timestamp'
            else:
                continue

            info = get_video_info(str(clip))
            if info is None:
                print(f'    SKIP corrupt: {clip.name}')
                continue
            fps, frames, w, h = info

            key = gesture
            groups.setdefault(key, []).append({
                'gesture': gesture,
                'subject': subject,
                'angle':   angle,
                'path':    str(clip.resolve()),
                'frames':  frames,
                'fps':     fps,
                'w':       w,
                'h':       h,
            })

    return groups


def main():
    p = argparse.ArgumentParser(
        description='Group v1 clips by gesture into merged videos.')
    p.add_argument('--v1_dir', default='data/v1_clips',
                   help='Root of v1 clip directory (default data/v1_clips).')
    p.add_argument('--output_dir', default='data/grouped_by_gesture',
                   help='Output directory (default data/grouped_by_gesture).')
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
        print('No clips found.')
        return

    print(f'Found {len(groups)} gesture groups.')

    gesture_order = list(V1_TO_V2.keys())

    for gesture, clips in sorted(groups.items()):
        v2_name = V1_TO_V2.get(gesture, gesture)
        out_mp4 = out_video / f'{v2_name}.mp4'
        out_json = out_label / f'{v2_name}.json'

        fps_vals = [c['fps'] for c in clips]
        target_fps = max(set(fps_vals), key=fps_vals.count)

        clips.sort(key=lambda c: (c['subject'], c['angle'], c['path']))

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
            print(f'  No valid clips for {v2_name}, skipping.')
            continue

        print(f'\n  [{v2_name}]  {len(good_clips)} clips  fps={target_fps:.1f}')
        for c in good_clips:
            print(f'    {Path(c["path"]).name}  '
                  f'subject={c["subject"]} angle={c["angle"]} frames={c["frames"]}')

        ok = concat_with_ffmpeg(
            [c['path'] for c in good_clips],
            str(out_mp4),
            target_fps,
        )

        if not ok:
            print(f'  FAILED to merge {v2_name}')
            continue

        actual_info = get_video_info(str(out_mp4))
        if actual_info is None:
            print(f'  FAILED to read merged video {v2_name}')
            continue
        _, actual_frames, _, _ = actual_info

        label = build_label_json(
            good_clips, str(out_mp4), actual_frames, target_fps,
        )

        with open(out_json, 'w') as f:
            json.dump(label, f, indent=2)

        n_segs = len(label['segments'])
        print(f'  -> {out_mp4.name}  ({actual_frames} frames, '
              f'{n_segs} labeled segments)')
        print(f'  -> {out_json.name}')

    print('\nAll gesture groups done.')


if __name__ == '__main__':
    main()
