"""
apply_transform_batch.py
Batch-apply geometric transform to all CSVs in a directory.
Copies label JSONs unchanged (geometry change does not affect labels).

Usage:
    python src/apply_transform_batch.py \
        --input_skel   data/skeletons/raw_45 \
        --input_label  data/labels/raw_45 \
        --output_skel  data/skeletons/transformed_45 \
        --output_label data/labels/transformed_45 \
        --angle        45.0
"""
import argparse
import shutil
from pathlib import Path

from transform_skeleton import transform_csv


def main(args):
    in_skel   = Path(args.input_skel)
    in_label  = Path(args.input_label)
    out_skel  = Path(args.output_skel)
    out_label = Path(args.output_label)
    out_skel.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_skel.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {in_skel}")

    print(f"Transforming {len(csv_files)} files  (angle={args.angle}deg) ...")
    ok, failed = 0, []

    for csv_path in csv_files:
        out_csv = out_skel / csv_path.name
        try:
            s = transform_csv(
                str(csv_path), str(out_csv),
                camera_angle_deg=args.angle,
                zero_z_after=True,
            )
            print(f"  OK  {csv_path.name:40s} "
                  f"{s['frames']:5d} frames  "
                  f"{s['detect_pct']:5.1f}% detect")
            ok += 1
        except Exception as e:
            print(f"  ERR {csv_path.name}: {e}")
            failed.append(csv_path.name)

        json_src = in_label / csv_path.with_suffix(".json").name
        json_dst = out_label / csv_path.with_suffix(".json").name
        if json_src.exists():
            shutil.copy2(json_src, json_dst)
        else:
            print(f"  WARN no label for {csv_path.name}")

    print(f"\nResult: {ok}/{len(csv_files)} succeeded.")
    if failed:
        print(f"Failed files: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_skel",   required=True)
    p.add_argument("--input_label",  required=True)
    p.add_argument("--output_skel",  required=True)
    p.add_argument("--output_label", required=True)
    p.add_argument("--angle", type=float, default=45.0,
                   help="Camera angle in degrees (positive = right of front)")
    main(p.parse_args())
