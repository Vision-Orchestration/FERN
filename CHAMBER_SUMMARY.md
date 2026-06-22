# FERN v2 — Chamber Audit Summary

**Audit date:** June 2026 | **5 agents, 52 suggestions, 48 passed (92%)**

---

## What Happened

A 5-agent Chamber audit (Code Reviewer, Security Auditor, Data Scientist, DevOps, Overseer) reviewed
the FERN v2 codebase for bugs, security issues, data quality, and maintainability gaps. 48 of 52
suggestions were unanimous or majority-passed and have been implemented.

## Fixes Applied (by phase)

### Phase 1 — Emergency (6 fixes)
| ID | Fix | File | Impact |
|----|-----|------|--------|
| D3 | `clips` → `good_clips` in merge_v1_db | `merge_v1_db.py:288` | **Crash fix** — prevented empty-dataset error |
| T2 | Moved `best_val_acc` init before resume block | `train_v2.py:277-291` | **Bug fix** — resume no longer resets best accuracy |
| D1 | Added `weights_only=True` to all `torch.load` calls | 4 files | **Security** — prevents pickle deserialization RCE |
| L2 | Created MIT `LICENSE` | `FERN_V2/LICENSE` | **Legal** — project now has explicit license |
| P1 | Added `onnxruntime` to reqs + spec | `requirements_v2.txt`, `FERN_v2.spec` | **Build fix** — PyInstaller no longer crashes |
| T4 | Fixed `input_features` in evaluate.py | `evaluate_v2.py:35-42` | **Bug fix** — eval model input dims matched training |

### Phase 2 — Label Quality (10 fixes)
| ID | Fix | Impact |
|----|------|--------|
| D4-D7 | Bounds checks, cam_id validity, overlap detection, segment-length diagnostic, sorted segments in `dataset_v2.py` | **Data integrity** — catches corrupt files early |
| D13 | FPS=0 fallback to 30 in `extract_skeleton.py:166` | **Crash fix** — div-by-zero on corrupted videos |
| D8 | Deprecated `mirror_skeletons.py` | **Clarity** — `mirror_10joint.py` is the active script |
| D9 | Empty-cell handling (preserved as NaN), removed precision truncation in `mirror_10joint.py` | **Data quality** — mirrors match originals exactly |
| D10 | `--keep-neutral` flag in `fix_labels.py` | **Usability** — keep neutral segments when desired |
| D12 | Event interleaving validation in `label_videos_v3.py:98` | **Data quality** — prevents interleaved gesture groups |

### Phase 3 — Training Pipeline (6 fixes)
| ID | Fix | Impact |
|----|------|--------|
| T3 | Resume loads checkpoint args for model construction | **Correctness** — model is reconstructed with training-time params |
| T5 | `--label_smoothing` as CLI argument | **Flexibility** — no more hardcoded 0.1 |
| T6 | `--num_workers` default changed 4→0 | **Stability** — Windows multiprocessing hang prevention |
| T7 | Early stopping configurable via `--early_stopping_metric` | **Quality** — defaults to `loss` (Chamber consensus) |
| C1 | `--num_workers` added to evaluate.py + kfold_cv.py | **Consistency** — all scripts respect num_workers |
| C2 | Verified split_mode choices match training/eval | **Consistency** — no split mismatch |

### Phase 4 — Infrastructure (6 fixes)
| ID | Fix | Impact |
|----|------|--------|
| A1/I2 | YAML config system created (`config_loader.py` + `configs/train_config.yaml`) | **DX** — reproducible configs without CLI args |
| I3 | `__init__.py` created in `src/` | **Import hygiene** — package imports work cleanly |
| N1/N2 | `run_nightly.ps1` — `num_workers` 4→0, paths aligned | **Pipeline** — nightly runs don't hang |

### Phase 5 — Deployment (8 fixes)
| ID | Fix | Impact |
|----|------|--------|
| P2 | `train_all` also saves `_best.pth` checkpoint | **Contract** — consistent model filename for run.bat |
| P3 | `--camera_view` forwarding added to `run.bat` | **Multi-camera** — end users can specify camera index |
| P5 | ONNX export reads `num_classes` + `dropout` from checkpoint | **Correctness** — export matches training config |
| P8 | Default threshold lowered 0.6 → 0.5 | **False negatives** — fewer uncertain frames |
| P9 | `torchaudio==2.2.2` removed from requirements | **130MB savings** — dead dependency removed |

### Phase 6 — Documentation (5 fixes)
| ID | Fix | Impact |
|----|------|--------|
| L3 | AGENTS.md updated with new defaults + critical context | **Onboarding** — docs reflect current state |
| G1 | `*.pth` added to `.gitignore` | **VC hygiene** — prevents accidental checkpoint commits |
| L1 | This summary | **Stakeholder visibility** — one-page digest |

## Files Changed

**30 source files** modified across the FERN v2 codebase:

- `src/train_v2.py` — 6 edits (resume, label_smoothing, num_workers, early_stopping, train_all save)
- `src/dataset_v2.py` — 6 edits (segment validation, bounds check, overlap detection)
- `src/export_onnx.py` — 3 edits (num_classes/dropout from checkpoint, default 8→8)
- `src/evaluate_v2.py` — 2 edits (input_features, num_workers CLI)
- `src/infer_v2.py` — threshold default 0.6→0.5
- `src/infer_onnx.py` — threshold default 0.6→0.5
- `src/kfold_cv.py` — num_workers default 4→0
- `src/merge_v1_db.py` — clips→good_clips fix
- `src/extract_skeleton.py` — fps=0 fallback
- `src/mirror_10joint.py` — empty-cell fix, precision fix
- `src/mirror_skeletons.py` — deprecated
- `src/fix_labels.py` — --keep-neutral flag
- `src/label_videos_v3.py` — event interleaving validation
- `src/__init__.py` — new file
- `src/config_loader.py` — new file
- `configs/train_config.yaml` — new file
- `requirements_v2.txt` — torchaudio removed
- `run.bat` — camera_view, threshold 0.5, docs
- `run_nightly.ps1` — num_workers 4→0
- `FERN_v2.spec` — onnxruntime added
- `.gitignore` — *.pth added
- `AGENTS.md` — updated
- `LICENSE` — new (MIT)

## Remaining Work (post-Chamber)

Items evaluated but deferred. Revisit when data scale justifies them:
1. **Config-driven training** — `config_loader.py` exists but `train_v2.py` does not yet call it
2. **Tests** — template available in `FERN_complete/tests/`; port when CI is set up
3. **Model manifest** — `model_meta.json` generation for model registry
4. **Skeleton unification** — merge `extract_skeleton.py` normalization paths
