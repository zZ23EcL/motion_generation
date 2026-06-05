# STMC → MuscleMimic Bridge

Feed **real STMC text-to-motion output** through MuscleMimic's GMR retargeting + muscle-actuated
tracker, as a faithful out-of-distribution (OOD) test of the tracker on the motions a scoped
text-to-motion system would actually generate.

This replaces the earlier proxy of hard-stitching MoCap clips (which diverges from STMC in three ways:
hard cuts vs STMC's smooth DiffCollage blends, inherited MoCap standing intros/outros, and ~180°
in-place turnarounds instead of prompt-driven turns). With STMC and MuscleMimic both available locally,
the real pipeline is cheap, so we test against genuine STMC output instead.

## Why the bridge is tiny

STMC's `extract_joints(..., jointstype="both")` already saves a `<name>_smpl.npz` containing:

| key | shape | note |
| --- | --- | --- |
| `poses` | `(T, 66)` | axis-angle, 22 SMPL joints, no hands |
| `trans` | `(T, 3)` | root translation |
| `joints` | `(T, 24, 3)` | unused here |
| `mocap_framerate` | scalar | `20.0` (STMC fps) |

MuscleMimic's `read_single_amass_motion` reads an AMASS-style npz needing
`poses[:, :66]`, `trans`, `betas`, `gender`, `mocap_framerate`. So the only missing fields are
**`betas` and `gender`** — the bridge adds `betas=zeros(16)` + `gender="neutral"` and lets GMR's
`use_fitted_shape=True` fit the robot body. The `mocap_framerate=20` tag is **mandatory** (GMR resamples
to 30 fps from it; a wrong/missing tag desyncs speed and motion-phase by 1.5×).

**One non-trivial format detail (learned on the first real GMR run):** the GMR retargeting path does
*not* go through `read_single_amass_motion`; it uses `general_motion_retargeting.utils.smpl.load_smplh_file`,
which only accepts the full **AMASS SMPL-H pose width of 156** (`root 3 + body 63 + left_hand 45 +
right_hand 45`). Given bare `poses (T,66)` it falls to a branch that needs `root_orient`/`pose_body`
keys and raises `KeyError: 'root_orient'`. So the bridge **zero-pads poses 66→156** (flat hands; hands
are irrelevant to the locomotion repertoire). `poses[:, :66]` is untouched, so the non-GMR loader still
works. This padding is the only real wiring the bridge does beyond betas/gender.

## Pipeline

```
text prompt
   │  STMC (MDM diffusion, generate.py)
   ▼
<name>_smpl.npz   (poses 66 + trans + mocap_framerate=20)
   │  bridge_stmc_to_amass.py   (+ betas=0, gender=neutral)
   ▼
AMASS/STMC/<name>.npz
   │  GMR-Fit retargeting (use_fitted_shape, resample 20→30fps)
   ▼
MyoFullBody robot trajectory
   │  MuscleMimic tracker π(a|s,g)   (mild_1p5 checkpoint)
   ▼
muscle actions → MJX physics → tracking metrics
```

## Setup (local, GPU recommended)

```bash
# MuscleMimic with the optional smpl + gmr extras (jax-cuda + torch + smplx + joblib + GMR)
cd musclemimic
uv sync --extra cuda --extra smpl --extra gmr

# STMC: its own env (torch 2.0.1, smplx, joblib, hydra, pytorch-lightning).
# Checkpoint dir: stmc/outputs/mdm-smpl_clip_smplrifke_humanml3d

# SMPL-H body model for GMR. GMR forward-kinematics the source human, so it needs an SMPL-H model at
#   $SMPL_MODEL_PATH  (default ~/.musclemimic/smpl)  named  SMPLH_NEUTRAL.pkl  (16-beta SMPL+H w/ MANO
#   hand components, use_pca=False). STMC already ships a compatible one — just point GMR at it:
mkdir -p ~/.musclemimic/smpl
ln -sf <repo>/stmc/deps/smplh/SMPLH_NEUTRAL.pkl ~/.musclemimic/smpl/SMPLH_NEUTRAL.pkl
export SMPL_MODEL_PATH=~/.musclemimic/smpl   # or run `musclemimic-set-smpl-model-path --path ...`
```

> The `--extra cuda` is required for step 4 (MJX eval on GPU). On a box where only the smpl+gmr extras
> were synced, JAX falls back to CPU (`jax.devices() == [CpuDevice]`); GMR retargeting (steps 1–3) still
> works on CPU, but MJX physics eval will be unusably slow until jax-cuda is installed.

> GMR comes from `general-motion-retargeting @ git+https://github.com/amathislab/gmr_plus.git`
> (the `gmr` extra). The shared CPU cluster lacked it entirely — that's why this work moved local.

## Run (4 steps)

```bash
# 1. Generate with STMC (GPU). NOTE: pass run_dir explicitly — the default points elsewhere.
cd stmc
python generate.py \
    run_dir=outputs/mdm-smpl_clip_smplrifke_humanml3d \
    timeline=../stmc-musclemimic-bridge/timelines/loco_walk_turn_walk.txt \
    device=cuda
#   (CPU fallback works too: device=cpu sampler=ddim sampling_steps=50)
#   The tail matplotlib .mp4 render may crash if ffmpeg is missing — harmless,
#   the *_smpl.npz is saved BEFORE rendering.

# 2. Bridge the generated npz → AMASS/STMC/<name>.npz
cd ../musclemimic
python ../stmc-musclemimic-bridge/bridge_stmc_to_amass.py \
    --stmc_dir ../stmc/outputs/mdm-smpl_clip_smplrifke_humanml3d/generations/loco_walk_turn_walk_last_timeline_to_motion/
#   single file:  --stmc_npz <path>_smpl.npz --out_name loco_walk_turn_walk_0

# 3. Retarget + render reference playback to eyeball it (CHECK FIRST FRAME: no ground
#    penetration / no float — STMC smplrifke is z=0 canonical; align GMR offset_to_ground).
python examples/retargeting/retarget_visualize.py \
    --motion STMC/loco_walk_turn_walk_0 --retargeting-method gmr \
    --record --output-dir stmc_videos --n-steps 300

# 4. Evaluate the tracker on the bridged motion (mild_1p5 checkpoint, MJX → needs GPU)
python fullbody/eval.py --motion_path STMC/loco_walk_turn_walk_0 ...
```

## Files

- `bridge_stmc_to_amass.py` — STMC `_smpl.npz` → AMASS-style npz. `--stmc_npz` (single) or
  `--stmc_dir` (batch); stages a copy to `data/stmc_refs/` for cross-node eval.
- `reference_sanity.py` — reference-quality check + cleaning on the GMR-retargeted cache npz. Reports a
  per-clip `VERDICT` (CLEAN / FRAME0_TRANSIENT / INTERIOR_BLOWUP / BOTH), the `jnorm_p10` never-settles
  correlate, and can `--trim_lead`/`--trim_range`. See **Reference sanity & cleaning** below.
- `stmc_smpl_smooth.py` — localize + fix jerk at the **SMPL source** (before retargeting), the decisive
  next experiment after the frame-0 hypothesis was falsified. `--report` measures source jerk (is the STMC
  output itself jerky, or does GMR introduce it?); `--smooth` low-pass-filters poses (rotation-aware 6D) +
  trans → new `_smooth_smpl.npz` to re-bridge/retarget/eval (no re-FK). See **Smoothing the source** below.
- `timelines/loco_*.txt` — 4 scoped, in-repertoire locomotion timelines. STMC timeline syntax is
  one interval per line: `text # start_s # end_s # bodypart` (e.g. `legs`).
  - `loco_walk_forward` — single walk
  - `loco_walk_circle` — walk in a circle
  - `loco_walk_turn_walk` — walk → turn left → walk
  - `loco_walk_turn_circle` — walk → turn right → walk in a circle

## Validated / not yet validated

- ✅ STMC generation produces valid SMPL (8 s timeline → `poses (160, 66)`, mocap_framerate=20).
- ✅ Bridge produces a correctly-typed AMASS npz (now `poses (T,156)`, hands zero-padded — see format
  note above).
- ✅ **GMR retargeting validated locally** (2026-06-04, RTX 4090). `STMC/loco_walk_turn_walk_0` → robot
  trajectory `(160, 89)` qpos, retarget IK error mean 1.9 cm / max 6.4 cm. Recorded playback:
  `musclemimic/stmc_videos/myofullbody_retargeted/loco_walk_turn_walk_0.mp4`.
- ✅ **Ground alignment (the one known wiring risk) is correct, across all 4 timelines.** With GMR
  `offset_to_ground=False` on STMC's z=0-canonical data, frame-0 lowest foot **geom** sits ~+2 cm (on
  floor) and the per-frame lowest geom averages ≈0 over each clip. No gross float/sink; only mild ≤5 cm
  stance-contact noise typical of contact-free IK retargeting, which MJX contact resolves. Per-motion
  (geom-level FK; pen% = frames >2 cm into floor; IK = mean per-site retarget error):

  | motion | frames | f0 foot z | mean low-z | worst low-z | pen% | IK err | net turn | total turn |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | loco_walk_forward_0 | 120 | +2.4 cm | +1.3 cm | −1.9 cm | 0% | 2.0 cm | +37° | 134° |
  | loco_walk_turn_walk_0 | 160 | +2.2 cm | +0.3 cm | −5.0 cm | 17% | 1.9 cm | −14° | 247° |
  | loco_walk_circle_0 | 160 | +2.5 cm | +1.1 cm | −3.2 cm | 9% | 2.0 cm | **+153°** | 258° |
  | loco_walk_turn_circle_0 | 200 | +2.2 cm | +1.0 cm | −3.8 cm | 9% | 1.9 cm | +4° | 174° |

  Videos: `musclemimic/stmc_videos/myofullbody_retargeted/<motion>.mp4`.
- ⚠️ **STMC locomotion samples wander rather than execute clean turn primitives.** `loco_walk_circle_0`
  is the one with genuine sustained directed turning (net +153°). The others have high *total* curvature
  (lots of |Δheading|) but small *net* turn — they meander/wobble, and the explicit "turning to the
  left" segments don't yield clean net heading changes (diffusion RNG also differs from the cluster run
  despite seed 1234). For the OOD yaw-drift test, `loco_walk_circle_0` is the cleanest sustained-turn
  reference; the wandering clips are still valid continuous-locomotion OOD inputs.
- ✅ **Step 4 (tracker eval) DONE — STMC locomotion tracks essentially perfectly.** mild_1p5
  (`checkpoints/stage1_baseline_seed1_mild_1p5/checkpoint_12500`, env MjxMyoFullBody) evaluated on all 4
  bridged+retargeted STMC motions (MJX/GPU, stochastic, eval_seed 0):

  | motion | frames | coverage | early-term | root_yaw err | root_xyz err | reward_total |
  | --- | --- | --- | --- | --- | --- | --- |
  | loco_walk_forward_0 | 392/393 | 0.997 | 0% | 0.050 | 0.511 | 1.030 |
  | loco_walk_turn_walk_0 | 526/527 | 0.998 | 0% | 0.035 | 0.754 | 0.960 |
  | loco_walk_circle_0 (+153°) | 526/527 | 0.998 | 0% | 0.027 | 0.455 | 1.003 |
  | loco_walk_turn_circle_0 | 659/660 | 0.998 | 0% | 0.035 | 0.678 | 0.991 |

  **All ~0.998 coverage, zero early terminations**, root-yaw error ~2–3° even on sustained turning. This
  matches in-distribution (~0.99) and far exceeds the abandoned composite proxy (0.66): **the predicted
  long-horizon yaw drift did not reproduce on faithful STMC output.** The scoped-STMC ↔ one-policy
  text-to-motion link is validated. Eval command:
  ```bash
  env -u LD_LIBRARY_PATH SMPL_MODEL_PATH=~/.musclemimic/smpl .venv/bin/python fullbody/eval.py \
      --path checkpoints/stage1_baseline_seed1_mild_1p5/checkpoint_12500 \
      --motion_path STMC/loco_walk_circle_0 \
      --evaluate_all --metrics --metrics_only --no_render --metrics_envs 8 --eval_seed 0
  ```
  (`env -u LD_LIBRARY_PATH` is required on this box so JAX uses its bundled CUDA wheels instead of the
  system CUDA 12.8 that shadows them; `--evaluate_all` is required so eval uses the `--motion_path` env
  rather than the config's 108-motion validation set.)

- ✅ **Stress test DONE (2026-06-05) — yaw drift stays refuted; the 2 failures are NOT yaw drift and NOT a
  frame-0 artifact (that hypothesis was tested and falsified) — their cause is still open.** 3 long/sharp
  multi-prompt timelines (`timelines/loco_stress_*.txt`, 15–23 s), same checkpoint, eval_seed 0:

  | motion | result | coverage | early-term | dies @frame | net turn | reference signature |
  | --- | --- | --- | --- | --- | --- | --- |
  | loco_stress_long_circle_0 | **PASS** | 0.999 | 0% | — | **+469°** | clean (jnorm settles to quiet stance) |
  | loco_stress_sustained_left_0 | FAIL | 0.13 | 100% | ~152 | +51° | persistently jerky, never settles; no impossible frame |
  | loco_stress_long_walk_turn_walk_0 | FAIL | 0.03 | 100% | ~46 | +197° | jerky + a true interior blow-up @f249–256 (but dies before it) |

  **Not yaw drift, not turn amount:** the PASSING motion has *by far the most* net turning (+469° ≈ 1.3
  loops, the heaviest sustained turn in the project) yet tracks at in-distribution quality; the two FAILS
  turn *less*. **The frame-0 transient is not the cause** (it looked like it — both fails open with a
  ~6 rad/s `pro_sup_l/r` forearm-pronation spin, joint-vel norm ~11 vs ~3.7 — but two checks falsify it):
  (1) the *passing* `loco_walk_forward_0` has an even larger frame-0 transient (ratio 23.6×) and tracks
  fine, so it's a near-universal GMR qvel **boundary effect**; (2) trimming the opening off `sustained_left`
  (`reference_sanity.py --trim_lead 44`, frame-0 jnorm → 0.90, VERDICT CLEAN) and re-eval'ing **still
  fails** (164/1083, coverage 0.15). What the fails actually look like: `sustained_left`'s death region has
  no impossible frame (feet on ground, root z 0.84–0.92) but is **persistently jerky and never settles**
  (root-ang-vel oscillating 0.2↔2.4 rad/s; jnorm p10 ≈ 0.78 vs ≈ 0.3 for passing clips) — the "STMC
  samples wander/wobble" quality issue, low-quality *throughout*. `long_walk_turn_walk` does contain one
  genuine physically-impossible interior blow-up (f249–256, root_wz → 35.8 rad/s) but dies at f46 before
  reaching it. ⇒ **the predicted sustained-turn yaw drift is refuted even at the stress limit (the solid
  result).**

  **Resolved (2026-06-05, `stmc_smpl_smooth.py`) — it's a generation content-quality issue, not smoothing-fixable.**
  (1) The jerk is *generation-side*: the FAIL source `_smpl.npz` has pose ang-speed **p10 1.59 vs 0.67** for
  the PASS source (~2.4×, matching the robot-cache jnorm-p10 ratio), so GMR faithfully passes it through.
  (2) But it is **not high-freq jitter**: a low-pass cutoff sweep (4/3/2/1.5 Hz) cuts the peaks but the
  **p10 floor is irreducible (~1.4–1.8, never reaches 0.67)** — "never settles" is low-frequency content
  (continuous whole-body agitation, no stance phase), not noise. (3) Smoothing at 2 Hz (peaks ~halved) →
  re-bridge → re-retarget → re-eval **still fails (155/1127, coverage 0.137)**, ≈ identical to the original.
  ⇒ the failure is a **generation content-quality problem** (over-agitated, never-resting STMC samples), NOT
  yaw drift, NOT a frame-0 transient, NOT GMR-introduced, and NOT removable by smoothing — and the
  robot-qpos-smooth + re-FK fallback would give the same negative result, so it is not worth building. The
  deployment guard is a **generation-quality GATE: reject-and-regenerate** samples whose reference never
  settles (`reference_sanity.py` `jnorm_p10` high), *not* trim/smooth. Logs: `stress_eval_logs/*_seed0.log`,
  `stress_eval_clean.log`, `stress_eval_smooth2.log`.

## Reference sanity & cleaning (`reference_sanity.py`)

Diagnose and clean the GMR-retargeted reference *before* blaming the tracker. Built to resolve the two
stress-test failure signatures (frame-0 velocity transient; mid-clip blow-up) and to serve as a
deployment guard (is a generated reference physically followable?).

It reads the robot-level cache npz directly (no GMR / no mujoco needed):
`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/<name>.npz`.

**Two independent discriminators** (absolute joint-vel norm is *not* one — normal MyoFullBody gait
legitimately spikes jnorm to ~13 during knee swing, so it's reported for info only):
- **frame-0 transient** — `jnorm[0] / settled-median` above `--frame0_ratio` (1.8) *and* an absolute
  floor `--frame0_floor` (6). This is a near-universal **GMR qvel boundary artifact** (also present on the
  hard-cut composites, ratio 5–34×, dominated by upper-body `shoulder*/elv_angle` DOFs) — and trimming
  *does* remove it from the cache. **But note (see Correction below): removing it does NOT fix tracking**,
  so this flag is a boundary-effect detector, not a failure predictor.
- **interior blow-up** — root angular-vel `> --rootang_thresh` (10 rad/s; gait/turning is <2, the observed
  glitch was 35.8) **or** single-frame joint jump `> --jump_thresh` (0.3 rad). Physically-impossible
  reference; cannot be patched without re-FK, so re-generate or trim past it.

```bash
# report one motion
python reference_sanity.py ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/loco_stress_sustained_left_0.npz

# batch artifact-rate scan (how often does multi-prompt STMC emit an unfollowable ref?)
for f in ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/*.npz; do
    python reference_sanity.py "$f" --quiet
done

# test the frame-0-transient hypothesis on a failing clip: trim the opening, re-cache, re-eval
python reference_sanity.py <cache>.npz --trim_lead 5 \
    --out ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/<name>_trim.npz
# then eval STMC/<name>_trim with the step-4 command; if it now tracks ~0.99 the fail was a boundary artifact
```

`--trim` slices all 8 time-indexed arrays (`qpos/qvel/xpos/xquat/cvel/subtree_com/site_xpos/site_xmat`)
consistently and fixes `split_points`, so the trimmed cache stays FK-consistent and is directly evaluable.

**Deployment GATE (now implemented):** the one-line output leads with `GATE=ACCEPT|REJECT`. A reference is
`REJECT`ed if it has an `INTERIOR_BLOWUP` (physically-impossible frame) **or** never-settles
(`jnorm_p10 > --settle_thresh`, default 0.5; calibrated on n=7: pass ~0.3 vs fail ~0.78). Both are
**reject-and-regenerate** signals — per the resolved stress-test finding, never-settling is over-agitated
generation *content*, not smoothing-fixable, so the guard is a generation-quality gate, **not** a
trim/smooth filter. `FRAME0_TRANSIENT` does not affect the gate (falsified boundary effect). Batch-scan
example: `for f in .../STMC/*.npz; do python reference_sanity.py "$f" --quiet; done | grep GATE=REJECT`.

> ⚠️ **Correction (2026-06-05, after testing):** the `FRAME0_TRANSIENT` flag turned out **not** to be a
> useful failure predictor and "auto-trim-then-track" does **not** work. The passing `loco_walk_forward_0`
> is flagged `FRAME0_TRANSIENT` (ratio 23.6×) yet tracks fine, and trimming the opening off the failing
> `loco_stress_sustained_left_0` (`--trim_lead 44`, frame-0 now CLEAN) **still fails** on re-eval. The
> frame-0 transient is a near-universal GMR qvel boundary effect, not the cause of the stress failures.
> The signal that actually *correlates* with failure is "the clip never settles" (jnorm p10 ≈ 0.78 for
> fails vs ≈ 0.3 for passes) — but that's correlational, not a validated followability guard. `--trim`
> and the `INTERIOR_BLOWUP` detector remain useful; the `FRAME0_TRANSIENT`→trim story does not.

## Smoothing the source (`stmc_smpl_smooth.py`)

After the frame-0 transient was falsified as the failure cause, the open question is **persistently jerky
reference** (generation quality) vs **a real tracker edge**. The decisive test is to low-pass-smooth the
reference and re-eval — but smoothing the robot `qpos` needs a mujoco re-FK to keep `xpos/site_xpos/cvel`
consistent. This tool sidesteps that by smoothing at the **SMPL source**, *before* retargeting, so the
normal GMR pass regenerates every derived array consistently.

It also answers *where* the jerk is. Run `--report` on a failing clip's `<name>_smpl.npz` and a passing
one: if the failing **source** is itself much jerkier (high pose-angular-speed p10, no quiet phases), the
jerk is **generation**; if the source is smooth, **GMR's per-frame IK** introduced it (and smoothing at
the source won't fully help — escalate to robot-level smoothing + re-FK).

```bash
# localize: compare source jerk of a fail vs a pass
python stmc_smpl_smooth.py <stmc_gen>/loco_stress_sustained_left_0_smpl.npz
python stmc_smpl_smooth.py <stmc_gen>/loco_stress_long_circle_0_smpl.npz

# fix (if source is the culprit): low-pass, then re-run the normal pipeline on the smoothed source
python stmc_smpl_smooth.py <...>_smpl.npz --smooth --cutoff_hz 4 --out <...>_smooth_smpl.npz
python bridge_stmc_to_amass.py --stmc_npz <...>_smooth_smpl.npz --out_name loco_stress_sustained_left_smooth
#   retarget (step 3) + eval (step 4) on STMC/loco_stress_sustained_left_smooth
```

Poses are filtered in the continuous 6D rotation representation then re-orthonormalized, so the output is
always a valid rotation; `--cutoff_hz` defaults to 4 Hz (locomotion is <~3 Hz) — lower it to smooth more
aggressively.

> **Result (2026-06-05) — this test was run and it RESOLVES the open question: it's generation content,
> not smoothing-fixable.** `--report` localized the jerk to the **generation side** (FAIL source pose
> ang-speed p10 **1.59** vs PASS **0.67**), so GMR is not the culprit. But a cutoff sweep (4/3/2/1.5 Hz)
> showed the **p10 floor is irreducible** (~1.4–1.8, never reaching 0.67): "never settles" is low-frequency
> *content* (continuous agitation, no stance phase), not high-freq jitter. Smoothing at 2 Hz (peaks halved)
> → re-bridge → re-retarget → re-eval **still fails (155/1127, coverage 0.137)**, ≈ identical to the
> original. ⇒ the failure is a generation **content-quality** problem, *not* removable by any smoothing
> (source-level here, and by extension the robot-qpos + re-FK escalation — don't build it). **Deployment
> fix = a generation-quality gate (reject-and-regenerate over-agitated samples via `jnorm_p10`), not
> smoothing.** This tool's `--report` localizer stays useful; its `--smooth` fix does not apply to this
> failure mode.
