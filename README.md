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
```

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
- `timelines/loco_*.txt` — 4 scoped, in-repertoire locomotion timelines. STMC timeline syntax is
  one interval per line: `text # start_s # end_s # bodypart` (e.g. `legs`).
  - `loco_walk_forward` — single walk
  - `loco_walk_circle` — walk in a circle
  - `loco_walk_turn_walk` — walk → turn left → walk
  - `loco_walk_turn_circle` — walk → turn right → walk in a circle

## Validated / not yet validated

- ✅ STMC generation produces valid SMPL (8 s timeline → `poses (160, 66)`, root xy disp 5.33 m, no
  ground penetration).
- ✅ Bridge produces a correctly-typed AMASS npz (Δyaw ≈ +167° for walk→left-turn→walk, as expected).
- ⚠️ **GMR retargeting step not yet run** (the cluster node had no GMR library). On the first local run,
  scrutinize the retarget_visualize first frame for ground alignment — this is the one known wiring risk
  (STMC's z=0 canonical frame vs GMR's `offset_to_ground` convention).
