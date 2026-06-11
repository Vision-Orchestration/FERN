"""
Robust skeleton extraction wrapper — processes videos one at a time,
skips to next on failure, logs progress in real time.
"""
import subprocess, sys, os, time
from pathlib import Path

SKEL_DIR = "data/skeletons/merged_v1"
VIDEO_DIR = "data/merged_v1"
SRC_DIR = "src"
TIMEOUT = 600  # seconds per video

done = {p.stem for p in Path(SKEL_DIR).glob("*.csv")}
videos = sorted(Path(VIDEO_DIR).glob("*.mp4"))
remaining = [v for v in videos if v.stem not in done]

print(f"CSVs so far: {len(done)} / {len(videos)}")
print(f"Remaining: {len(remaining)}")
print()

for i, vpath in enumerate(remaining, 1):
    csv_path = Path(SKEL_DIR) / f"{vpath.stem}.csv"
    print(f"[{i}/{len(remaining)}] {vpath.name} ... ", end="", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "src/extract_skeleton.py",
             "--video_dir", str(vpath.parent),
             "--output_dir", SKEL_DIR],
            capture_output=True, text=True, timeout=TIMEOUT,
            env={**os.environ, "PYTHONPATH": str(Path(SRC_DIR).resolve())},
            cwd=Path.cwd().parent if Path.cwd().name == "src" else Path.cwd(),
        )
        elapsed = time.time() - t0
        if csv_path.exists() and csv_path.stat().st_size > 100:
            print(f"OK  ({elapsed:.0f}s)")
        else:
            print(f"FAIL (size={csv_path.stat().st_size if csv_path.exists() else 0})")
            print(f"  stderr: {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({TIMEOUT}s)")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone.  Total: {len(list(Path(SKEL_DIR).glob('*.csv')))} / {len(videos)}")
