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
  per-clip `VERDICT` (CLEAN / FRAME0_TRANSIENT / INTERIOR_BLOWUP / BOTH) and can `--trim_lead`/`--trim_range`
  to produce an FK-consistent cleaned reference. See **Reference sanity & cleaning** below.
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

- ✅ **Stress test DONE (2026-06-05) — yaw drift stays refuted; the failures are reference artifacts, not
  drift.** 3 long/sharp multi-prompt timelines (`timelines/loco_stress_*.txt`, 15–23 s), same checkpoint,
  eval_seed 0:

  | motion | result | coverage | early-term | dies @frame | net turn | frame0 jnorm | reference artifact |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | loco_stress_long_circle_0 | **PASS** | 0.999 | 0% | — | **+469°** | 3.7 (normal) | none |
  | loco_stress_sustained_left_0 | FAIL | 0.13 | 100% | ~152 | +51° | 10.8 | ~6 rad/s forearm-pronation spin at f0 |
  | loco_stress_long_walk_turn_walk_0 | FAIL | 0.03 | 100% | ~46 | +197° | 11.1 | f0 forearm spin **+ 35.8 rad/s root blow-up @f252** |

  **The discriminator is reference quality, not turn amount.** The PASSING motion has *by far the most*
  net turning (+469° ≈ 1.3 loops, the heaviest sustained turn in the project) yet the *cleanest*
  reference — it tracks at in-distribution quality. The two FAILS have *less* turning but start with a
  physically-odd frame-0 transient (joint-vel norm ~11 vs 3.7, dominated by a ~6 rad/s `pro_sup_l/r`
  forearm-pronation spin — a wrist-twist DOF, irrelevant to locomotion), and `long_walk_turn_walk` also
  contains a gross GMR/STMC blow-up at frame 252 (root angular velocity 35.8 rad/s ≈ 2050°/s, a 0.70-rad
  single-frame qpos jump) — physically impossible, an unfollowable retargeting artifact. Trajectory-export
  rollout of the fast fail shows the policy diverging in **root translation** (loses balance off the poor
  non-gait opening), not in a turning-specific way. ⇒ **the predicted sustained-turn yaw drift is refuted
  even at the stress limit; the new failure mode is multi-prompt STMC DiffCollage timelines emitting
  physically-implausible references — a generation/bridge-quality issue to clean up (velocity-sanity
  filter / smooth the opening transient), NOT a tracker robustness gap and NOT yaw drift.** Logs:
  `stress_eval_logs/*_seed0.log`.

## Reference sanity & cleaning (`reference_sanity.py`)

Diagnose and clean the GMR-retargeted reference *before* blaming the tracker. Built to resolve the two
stress-test failure signatures (frame-0 velocity transient; mid-clip blow-up) and to serve as a
deployment guard (is a generated reference physically followable?).

It reads the robot-level cache npz directly (no GMR / no mujoco needed):
`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/<name>.npz`.

**Two independent discriminators** (absolute joint-vel norm is *not* one — normal MyoFullBody gait
legitimately spikes jnorm to ~13 during knee swing, so it's reported for info only):
- **frame-0 transient** — `jnorm[0] / settled-median` above `--frame0_ratio` (1.8) *and* an absolute
  floor `--frame0_floor` (6). This is the **GMR qvel boundary artifact**: verified universal across all 4
  hard-cut composites too (ratio 5–34×, dominated by upper-body `shoulder*/elv_angle` DOFs), i.e. it is a
  retargeting boundary effect, not STMC content — so it is fixable by trimming a few opening frames.
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

**Intended use of the verdict for scoped-STMC deployment:** treat `INTERIOR_BLOWUP`/`BOTH` clips as
reject-and-regenerate (the reference is non-physical); treat `FRAME0_TRANSIENT` as auto-trim-then-track.
This is the input sanity guard the stress test pointed at — *not* tracker retraining.
