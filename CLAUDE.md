# Project context for Claude Code (handoff)

This file primes a fresh Claude Code session (e.g. on the local 4090) with the state of this work.
It was written on a cluster session that built the STMC→MuscleMimic bridge; the live pipeline (GMR
retarget + MJX eval) moved local because the cluster node had no GPU, no GMR library, and no git creds.

## What this repo is

The **STMC → MuscleMimic bridge**: feed real STMC text-to-motion output through MuscleMimic's GMR
retargeting + muscle-actuated tracker, as a *faithful* OOD test. See `README.md` for the 4-step pipeline.
This repo holds only the bridge layer (script + timelines); it sits on top of two external repos you
clone alongside it:
- `amathislab/musclemimic` (branch `exp-qflex-inspired-stage1`) — the tracker, GMR-Fit retargeting, eval.
- `nv-tlabs/stmc` — the text-to-motion generator (MDM diffusion, checkpoint
  `outputs/mdm-smpl_clip_smplrifke_humanml3d`).

## The bigger project (MuscleMimic)

Muscle-actuated motion-imitation RL (JAX/MJX PPO, MyoFullBody, 354 muscles). Goal: one policy tracking
many motions, then build scoped text-to-motion (STMC) on top. **Standing constraint:** only test motions
*within* the tracker's repertoire — walk / turn / circle — and their **combinations**; NOT new skills
(kicks/jumps). Scoped STMC will only generate within that locomotion range, so that's the deployment
distribution.

## Key findings so far (don't re-derive)

- **In-distribution tracking is saturated (~0.99 frame coverage); OOD is the real bottleneck.**
- **Regularization buys precision, not robustness.** mild_1p5 (action_rate=1.5e-4, act_energy=1.5e-5) vs
  noreg(0,0): mild is more accurate in-dist (err_joint_vel −16%) but ties noreg on OOD/composites.
  mild_1p5 kept as the working baseline checkpoint.
- **Hard-stitched MoCap composites were a biased proxy and are abandoned.** They diverge from STMC three
  ways: hard cuts vs STMC smooth DiffCollage blends; inherited MoCap standing intros/outros; KIT "turn"
  clips are ~180° in-place turnarounds, not gentle turns. Their only transferable result:
- **Failure mode = long-horizon yaw drift during sustained turn/circle, NOT seams.** Across 30 composites,
  only 2/51 failures were near a seam (even a 2.35-rad hard cut was crossed); the rest were in-segment
  drift. → **Do NOT build seam-smoothing (v2 slerp).** It was *predicted* the drift mechanism would carry
  to STMC's continuous turning/circling — but see the faithful-test result below, which did NOT reproduce
  it.
- **The faithful test is real STMC output** (this repo), which is why we built the bridge.
- **FAITHFUL TEST RESULT (2026-06-04): STMC locomotion tracks essentially perfectly — the predicted yaw
  drift did NOT materialize.** mild_1p5 (`checkpoint_12500`) on all 4 bridged+retargeted STMC motions:
  frame_coverage **0.997–0.998, zero early terminations**, root_yaw err 0.03–0.05 rad (~2–3°) even on the
  sustained-turn `loco_walk_circle_0` (net +153°). That's in-distribution quality (~0.99) and far above
  the abandoned composite proxy (0.66) — **confirming the composites were pessimistically biased, not the
  real OOD.** ⇒ the scoped-STMC ↔ one-policy link is **validated**; locomotion text-to-motion is within
  the tracker's repertoire. Caveats: single eval_seed, short clips (4–7 s), and these STMC samples wander
  but don't push very aggressive sustained turns — longer/sharper sequences could still expose drift.

- **STRESS TEST RESULT (2026-06-05): yaw drift stays REFUTED under heavy sustained turning. Two
  multi-prompt clips fail, but the cause is NOT yaw drift, NOT a frame-0 transient, and NOT a single
  identifiable artifact — it stays OPEN (best read: persistently jerky low-quality references).** Built 3
  long/sharp multi-prompt timelines (15–23 s) and eval'd mild_1p5 `checkpoint_12500` (eval_seed 0):
  - `loco_stress_long_circle_0` (3 chained "walking in a circle", **net +469° ≈ 1.3 loops, 14.6 s**):
    **PASS — coverage 0.999, 0 early-term, root_yaw 0.045 rad.** This is the *most* sustained turning in
    the whole project and it tracks at in-distribution quality. ⇒ sustained-turn yaw drift is refuted even
    at the stress limit. **This is the solid, load-bearing result.**
  - `loco_stress_sustained_left_0` (walk + 3×"turning left" + walk): **FAIL, early-term, coverage 0.13,
    dies ~frame 152.**
  - `loco_stress_long_walk_turn_walk_0` (walk7s+turnL+walk7s+turnR+walk7s): **FAIL, coverage 0.03, dies
    ~frame 46.**
  - **Not yaw drift / not turn-amount:** the PASSING motion has *by far the most* net turning (+469°); the
    two FAILS turn *less* (+51°, +197°). Failure does not track turn amount.
  - **The frame-0 "artifact" hypothesis was tested and FALSIFIED.** Both fails open with a high-velocity
    transient (joint-vel norm ~11 vs ~3.7, dominated by a ~6 rad/s `pro_sup_l/r` forearm-pronation spin),
    and it looked like the cause. But: (1) the *passing* `loco_walk_forward_0` has an even larger frame-0
    transient (jnorm[0]/settled ratio 23.6×) and tracks fine — so the transient is a near-universal GMR
    qvel **boundary effect, not causal**; (2) directly trimming the opening off `sustained_left` (cache
    `..._clean.npz`, `reference_sanity.py --trim_lead 44`; frame-0 jnorm now 0.90, VERDICT=CLEAN) and
    re-eval'ing **still FAILS** (dies 164/1083, coverage 0.15). ⇒ the opening transient is **not** why
    these clips fail.
  - **What the failures actually look like:** `sustained_left`'s death region (frames ~120–220) has **no
    impossible frame** — feet on ground (lowest geom z ≈ 0, no penetration/float), root z 0.84–0.92
    (slightly crouched), moderate velocities — but it is **persistently jerky and never settles**: root
    angular vel oscillates 0.2↔2.4 rad/s frame-to-frame and jnorm never drops to a quiet stance phase
    (its jnorm p10 ≈ 0.78 vs ≈ 0.3 for passing clips). This is the "STMC samples wander/wobble" quality
    issue, i.e. low-quality *throughout*, not one bad frame. `long_walk_turn_walk` DOES contain one genuine
    physically-impossible interior blow-up (frames 249–256, root_wz → 35.8 rad/s, 0.38-rad qpos jump), but
    it **dies at frame 46, before reaching it**, so even that hard artifact is not the proximate cause.
  - **Smoothing test (2026-06-05) — RESOLVED via `stmc_smpl_smooth.py`: it's a generation *content* quality
    issue, and smoothing of any kind cannot fix it.** (1) *Localize:* the FAIL source `_smpl.npz` is jerkier
    than the PASS source **at the SMPL level, before GMR** — pose ang-speed p10 **1.59 (fail) vs 0.67
    (pass)**, ~2.4× (matching the robot-cache jnorm p10 ratio 0.78/0.3), so GMR faithfully passes the jerk
    through; it is *generation-side*, not GMR-introduced. (2) *But it's not high-freq jitter:* a low-pass
    cutoff sweep (4/3/2/1.5 Hz) cuts the peaks (p90/max/accel) but **the p10 floor is irreducible — stays
    ~1.4–1.8, never approaches the passing 0.67** — because "never settles" is low-frequency *content*
    (continuous whole-body agitation with planted feet, no stance phase), not noise. (3) *Decisive eval:*
    smoothed at 2 Hz (peaks ~halved) → re-bridge → re-retarget → re-eval **STILL FAILS, 155/1127, coverage
    0.137** — essentially identical to the original (152/1127, 0.135) and the trim (164/1083, 0.151).
    Halving the peaks bought zero extra frames ⇒ the driver is the never-settling floor, which is
    irreducible. ⇒ **the robot-qpos-smooth + re-FK fallback would give the same negative result; do NOT
    build it.**
  - **Honest conclusion:** the faithful-test verdict holds and is strengthened (clean scoped locomotion
    incl. the heaviest sustained turn tracks ~0.99; yaw drift refuted). The 2 multi-prompt failures are a
    **generation content-quality problem**: these particular STMC samples are *over-agitated, never-resting*
    motions (high ang-speed/jnorm **p10** = no stance phase) that the tracker can't follow to completion —
    while it tracks every clean rhythmic-gait clip incl. the heaviest turn. It is **NOT** yaw drift, **NOT**
    a frame-0 transient, **NOT** GMR-introduced, and **NOT** removable by smoothing. **Deployment guard is a
    generation-quality GATE, not a filter**: reject-and-regenerate STMC samples whose reference never
    settles (`reference_sanity.py` `jnorm_p10` high) — do NOT trim or smooth. Caveat: single eval_seed.
    Logs: `musclemimic/stress_eval_logs/*_seed0.log`, `motion_generation/stress_eval_{clean,smooth2}.log`.

## Bridge facts (so the format layer is "free")

STMC `extract_joints(jointstype="both")` saves `<name>_smpl.npz` = `poses (T,66)` + `trans (T,3)` +
`mocap_framerate=20`. MuscleMimic needs poses[:, :66]/trans/betas/gender/mocap_framerate, so the bridge
only adds `betas=zeros(16)` + `gender="neutral"`. **`mocap_framerate=20` is mandatory** (GMR resamples
20→30 fps; wrong tag desyncs speed/phase 1.5×). GMR uses `use_fitted_shape=True` so betas content is moot.

**Update (2026-06-04, first real GMR run): the bridge also zero-pads `poses 66→156`.** The GMR path
reads via `general_motion_retargeting...load_smplh_file`, NOT `read_single_amass_motion`; it only accepts
AMASS SMPL-H width 156 (root3+body63+lhand45+rhand45) and `KeyError: 'root_orient'`s on bare 66. STMC's
`[:, :66]` (root+body) maps to AMASS `[:, :66]`; hands padded to zero (irrelevant to locomotion). Done in
`bridge_stmc_to_amass.py`. Also: GMR needs an SMPL-H **body** model at `$SMPL_MODEL_PATH`
(default `~/.musclemimic/smpl`) as `SMPLH_NEUTRAL.pkl`; STMC's `stmc/deps/smplh/SMPLH_NEUTRAL.pkl`
(16-beta SMPL+H w/ MANO) works — symlink it there.

## Env gotchas

- Install: `cd musclemimic && uv sync --extra cuda --extra smpl --extra gmr` (GMR =
  `general-motion-retargeting @ git+https://github.com/amathislab/gmr_plus.git`).
- STMC `generate.py`: pass `run_dir=outputs/mdm-smpl_clip_smplrifke_humanml3d` explicitly (default is
  wrong). CPU-feasible via `device=cpu sampler=ddim sampling_steps=50`; GPU does full ddpm.
- STMC's tail matplotlib `.mp4` render crashes without ffmpeg — harmless, the `_smpl.npz` is saved first.
- Only HuggingFace Hub was blocked on the cluster (`HF_HUB_OFFLINE=1`); github/pypi were fine. Your local
  box has internet, so this likely won't bite you.
- **MJX eval GPU gotcha (local 4090):** this box has a system CUDA 12.8 on `LD_LIBRARY_PATH`
  (`/usr/local/cuda-12.8/lib64`) whose `libcusparse.so.12` shadows JAX's pip-wheel CUDA libs →
  `RuntimeError: Unable to load cuSPARSE`, JAX silently falls back to CPU. Fix: run eval with
  `env -u LD_LIBRARY_PATH ...` so JAX uses its bundled `nvidia-*-cu12` wheels → `jax.devices()==[CudaDevice]`.
- **Eval a single custom motion:** `fullbody/eval.py --metrics` alone runs the config's *validation set*
  (108 KIT motions) and ignores `--motion_path`; it also crashed in the MJX cvel metric. Use
  `--evaluate_all --metrics --metrics_only --motion_path STMC/<name> --metrics_envs 8 --no_render`, which
  evaluates the env built from `--motion_path` (just our motion) and works. Full command that produced the
  results: `env -u LD_LIBRARY_PATH SMPL_MODEL_PATH=~/.musclemimic/smpl .venv/bin/python fullbody/eval.py
  --path checkpoints/stage1_baseline_seed1_mild_1p5/checkpoint_12500 --motion_path STMC/<name>
  --evaluate_all --metrics --metrics_only --no_render --metrics_envs 8 --eval_seed 0`.

## Immediate next steps (where it was left)

**Progress 2026-06-04 (RTX 4090):** Steps 1–3 run end-to-end locally and validated.
- Step 1 ✅ STMC gen `loco_walk_turn_walk` → `poses (160,66)`, fps=20.
- Step 2 ✅ bridge → `AMASS/STMC/loco_walk_turn_walk_0.npz` (`poses (160,156)`).
- Step 3 ✅ GMR retarget — **all 4 timelines** generated+bridged+retargeted+ground-checked (cached at
  `~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/`, videos in `musclemimic/stmc_videos/`). IK err
  ~1.9–2.0 cm. **Ground alignment correct on all 4** (the known wiring risk is cleared): frame-0 lowest
  foot geom ≈ +2 cm, per-frame lowest geom mean ≈ 0, worst penetration ≤5 cm — no float/sink. See the
  per-motion table in README. Caveat: STMC samples *wander* (high total curvature, small net turn);
  `loco_walk_circle_0` is the only clean sustained-turn ref (net +153°). Eval keys ready:
  `STMC/loco_walk_{forward,turn_walk,circle,turn_circle}_0`.

- Step 4 ✅ DONE (2026-06-04). mild_1p5 `checkpoints/stage1_baseline_seed1_mild_1p5/checkpoint_12500`
  eval on all 4 STMC motions → ~0.998 coverage, 0 early-term, no yaw drift (see Key findings). The
  primary question ("does faithful scoped-STMC locomotion track?") is answered: **yes.**

- Stress test ✅ DONE (2026-06-05). 3 long/sharp multi-prompt timelines built + eval'd. Yaw drift stays
  refuted (`long_circle` +469° passes at 0.999). 2 multi-prompt clips fail; both the frame-0-transient
  hypothesis (trim) **and** the jerk/smoothing hypothesis (`stmc_smpl_smooth.py`, 2 Hz, re-retarget+re-eval)
  were **tested and falsified**. **Resolved: a generation content-quality issue (never-settling motion), not
  yaw drift / GMR / tracker gap, and not smoothing-fixable.** See "STRESS TEST RESULT" in Key findings.
  New: `timelines/loco_stress_*.txt`, `reference_sanity.py`, `stmc_smpl_smooth.py`.

**What's left / where to push next:**
1. **Deployment guard = generation-quality GATE (not smoothing/trimming).** The mechanism is settled: gate
   scoped-STMC output on the reference *never settling* (`reference_sanity.py` `jnorm_p10` high = no stance
   phase) → **reject-and-regenerate** that sample. Hard-reject `INTERIOR_BLOWUP` (root-ang-vel ≫ a few
   rad/s / single-frame qpos jump) too. Do NOT trim, smooth, or build the robot-qpos-smooth + re-FK tool —
   smoothing can't reduce the p10 floor (proven). Optional robustness: multi-`--eval_seed` (single so far),
   and confirm the gate's `jnorm_p10` threshold on more samples.
2. **Optionally finish `reference_sanity.py`'s gate.** `jnorm_p10` is now the primary reported signal; turn
   it into the explicit reject criterion (it cleanly separated 5 pass ~0.3 vs 2 fail ~0.78). `FRAME0_TRANSIENT`
   is a non-actionable boundary effect (documented). The continuous-agitation gate is the deployment lever.
3. The composite proxy can now be formally retired in the writeup — both the faithful test and the stress
   test refute its 0.66 / yaw-drift pessimism (the heaviest sustained turn, +469°, tracks at 0.999).
