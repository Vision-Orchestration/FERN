# FERN v2 — Roadmap to Publication-Grade Results

## 0. Stop and verify one thing before trusting the 484-file number

Your augmentation scripts (`augment_rotate.py`, `augment_spatial.py`) generate ~11 synthetic variants per real subject. If their output filenames don't carry the original `pXX_c1` prefix the same way `mirror_10joint.py` does, `kfold_cv.py`'s subject-grouping regex (`^(p\d{2})_c\d`) will either silently misgroup them (same bug that inflated your first 73.82% run) or — even if it matches — group synthetic siblings of the same person together, which doesn't prevent leakage the way grouping across genuinely different people does.

**Action:** run this before anything else, and paste the output before proceeding:
```powershell
Get-ChildItem data/skeletons/front_aug -Filter *.csv | Select-Object -First 15 Name
```
Confirm every filename still starts with `pXX_c` where `XX` matches the *original* subject it was derived from. If not, fix the naming in the augmentation scripts first — this is a correctness blocker, not an optimization.

---

## 1. The honest framing problem (do this before touching code)

484 files from 44 real people is **not** 484 independent samples — it's 44 independent samples wrapped in synthetic perturbations. A paper reviewer will ask "how many actual subjects?" and the honest answer stays 44 (well, 50 now if you count the earlier 6 you added — verify your true unique-person count). This isn't disqualifying, but it must be stated explicitly, or it reads as inflating N.

**What to write in the paper's dataset section** (draft this now, not later):
> "The dataset comprises N sessions from M unique subjects (front-facing camera, 8 gesture classes). To address the small subject pool, we apply geometric and temporal augmentation (rotation, spatial perturbation, interpolation) yielding K training instances. All cross-validation is performed at the subject level to prevent augmented variants of the same individual from appearing in both training and validation folds."

Fill in N, M, K once you've confirmed step 0. This single paragraph pre-empts the most common desk-reject reason for small-N gesture-recognition papers.

---

## 2. Immediate technical checks (before more training runs)

| # | Action | Why | Time |
|---|--------|-----|------|
| 1 | Verify step 0 (subject prefix in augmented filenames) | Correctness blocker | 5 min |
| 2 | Run 5-fold CV on `front_aug` with `--group_by subject` | Get the real number for the augmented set — you haven't reported this yet, only train-all numbers exist in the log | ~15-20 min (484 files, more epochs) |
| 3 | Compare CV(front_aug) vs CV(front, 74.23%) | If augmented CV isn't meaningfully better, the augmentation isn't earning its complexity — report both either way | — |
| 4 | Re-run with `--group_by video` as a secondary check | Confirms subject-level grouping isn't accidentally too strict/lenient given synthetic siblings | 15 min |

Do not report the 484-file train-all number as your headline result. Train-all accuracy on synthetic-heavy data is close to meaningless for a paper — reviewers care about the CV number.

---

## 3. What actually moves the needle for a paper (ranked by effort vs. credibility gain)

### Tier 1 — cheap, do these first

1. **Report both real and augmented CV side by side.** A table showing CV accuracy with vs. without augmentation is itself a contribution ("augmentation improves generalization by X pp") — this is more publishable than a single high number.
2. **Per-class confusion matrix**, not just aggregate accuracy. You already have `evaluate_v2.py` — run it and include the matrix. Reviewers always ask which gestures get confused; foot_lift vs forward_step, or heel_tap vs flamingo_bend, are the likely confusions given your class list — show it rather than have them assume it.
3. **Ablation table**: BiLSTM-abandoned decision, cnn_out sweep (64 vs 128), mirror-only vs full augmentation. You already ran most of these experiments — collect them into one table instead of scattered logs. This is free academic credit sitting in your git history.
4. **Report subject demographics** if available (age range, self-reported gender split, etc.) — even a one-line disclosure strengthens generalizability claims and is standard in HCI/gesture papers.

### Tier 2 — moderate effort, real accuracy gains

5. **Test-time augmentation (TTA):** at inference, run the same window through the model with 2-3 augmented views (e.g. small rotation ± mirror) and average logits. Free accuracy bump (typically 1-3pp) with no retraining, and it's a standard technique reviewers recognize — cite it as such rather than presenting it as novel.
6. **Class-weighted loss or oversampling** for your minority classes. Your class distribution from the last run:
   - foot_hold: 4648 windows (63.7%)
   - cross_front: 556, forward_step: 448, forward_kick: 384, foot_lift: 338, sideway_kick: 318, heel_tap: 308, flamingo_bend: 294

   foot_hold dominates ~15:1 over the rarest class. This imbalance is likely a real contributor to your fold variance — a "hard" fold probably has proportionally more of a rare class in validation. Try `--label_smoothing` (already in place) plus class-weighted `CrossEntropyLoss` and compare CV before/after.
7. **Confidence calibration** (you already have this flagged as a next step — calibrate threshold from validation histogram, not hardcoded). Report expected calibration error (ECE) — cheap to compute, standard metric, adds rigor.
8. **Leave-one-subject-out (LOSO) cross-validation** in addition to 5-fold. With 44-50 subjects this is expensive (44-50 training runs) but is the gold-standard evaluation for gesture recognition papers with small subject pools — it removes fold-size variance entirely and each subject gets its own honest test. Consider running it only on the final model config, not during hyperparameter search.

### Tier 3 — real new data (highest credibility, highest cost)

9. **More real subjects remains the single highest-leverage action.** Your own findings already state this ("data scarcity is the primary bottleneck," architecture has diminishing returns). Every real new subject is worth more than another augmentation variant. Prioritize contributor recruitment over further synthetic multiplication.
10. **Demographic/condition diversity in new recordings**, not just count. If your current 44-50 subjects skew toward one age group, body type, or clothing style, actively recruit outside that skew — a gesture recognition paper claiming general applicability needs to show it wasn't just tested on people who look like the existing pool.
11. **Held-out true test set.** Before running any more experiments, freeze 5-8 subjects (never touched by any hyperparameter tuning, augmentation design, or model selection) as a final test set. Report CV during development, then report this held-out number exactly once at the end. This is what separates a credible paper from one that overfit its own validation protocol through repeated tuning.
12. **Cross-environment recordings**: different lighting, floor color/pattern, clothing. MediaPipe detection can degrade with contrast — even a small set of "hard condition" recordings, evaluated separately, gives you a robustness section.

---

## 4. Specific analysis your agent (OpenCode) can run without new data

These are pure code/analysis tasks — good candidates for the AGENTS.md delegation workflow:

1. **Per-fold, per-class breakdown**: which specific gesture classes drop hardest in the bad folds (fold 3, 44.66%)? This tells you if it's a genuinely hard subject or a systematically confused gesture pair.
2. **t-SNE or PCA visualization** of the CNN's learned window embeddings, colored by class. Standard figure in gesture-recognition papers, cheap to generate from an existing checkpoint.
3. **Learning curve**: train-all accuracy vs. number of subjects included (train on 10, 20, 30, 40, 50 subjects, plot accuracy). This directly visualizes and *proves* your own "data scarcity is the bottleneck" finding — turns an internal note into a publication-ready figure.
4. **Params vs. accuracy comparison table**: you already tested BiLSTM (abandoned) and cnn_out=64 vs 128 — formalize this into a clean ablation table with all configs tried, not just the winner. Reviewers want to see the search space, not just the destination.
5. **Inference latency/throughput benchmark** on your RTX 3070 (ONNX vs PyTorch, batch=1 real-time scenario) — if FERN targets real-time use, this is a required, and currently missing, results section.

---

## 5. What NOT to do

- Don't keep multiplying synthetic augmentation past ~3-5x without new real subjects backing it — diminishing/negative returns, and it weakens your "M unique subjects" honesty story further.
- Don't report train-all accuracy as your primary metric anywhere in the paper. It's fine as a footnote ("final model refit on full dataset for deployment achieves X%") but never as the headline number.
- Don't skip the held-out test set to save time — repeatedly tuning against your only CV split, even with good intentions, is textbook overfitting-to-validation and reviewers who know gesture-recognition literature will ask about it directly.

---

## 6. Suggested immediate sequence

1. Verify augmented filenames carry correct subject prefix (§0) — **blocking**
2. Freeze a held-out test set now, before more tuning (§3.11)
3. Run 5-fold CV on `front_aug` with confirmed grouping (§2)
4. Generate confusion matrix + per-class breakdown on current best model (§3.2, §4.1)
5. Try class-weighted loss, compare CV (§3.6)
6. Decide: is LOSO worth the compute time given your deadline? If yes, kick it off as a background/overnight run (§3.8)
7. Draft the dataset-honesty paragraph (§1) using final confirmed N/M/K numbers
