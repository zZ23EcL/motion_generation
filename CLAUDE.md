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
  drift. → **Do NOT build seam-smoothing (v2 slerp).** The drift mechanism WILL carry to STMC's continuous
  turning/circling — that's the real lever.
- **The faithful test is real STMC output** (this repo), which is why we built the bridge.

## Bridge facts (so the format layer is "free")

STMC `extract_joints(jointstype="both")` saves `<name>_smpl.npz` = `poses (T,66)` + `trans (T,3)` +
`mocap_framerate=20`. MuscleMimic needs poses[:, :66]/trans/betas/gender/mocap_framerate, so the bridge
only adds `betas=zeros(16)` + `gender="neutral"`. **`mocap_framerate=20` is mandatory** (GMR resamples
20→30 fps; wrong tag desyncs speed/phase 1.5×). GMR uses `use_fitted_shape=True` so betas content is moot.

## Env gotchas

- Install: `cd musclemimic && uv sync --extra cuda --extra smpl --extra gmr` (GMR =
  `general-motion-retargeting @ git+https://github.com/amathislab/gmr_plus.git`).
- STMC `generate.py`: pass `run_dir=outputs/mdm-smpl_clip_smplrifke_humanml3d` explicitly (default is
  wrong). CPU-feasible via `device=cpu sampler=ddim sampling_steps=50`; GPU does full ddpm.
- STMC's tail matplotlib `.mp4` render crashes without ffmpeg — harmless, the `_smpl.npz` is saved first.
- Only HuggingFace Hub was blocked on the cluster (`HF_HUB_OFFLINE=1`); github/pypi were fine. Your local
  box has internet, so this likely won't bite you.

## Immediate next steps (where it was left)

1. **Run the bridge pipeline end-to-end locally** (README steps 1–4). It is validated through step 2
   (STMC gen + bridge produce correct data); **steps 3–4 (GMR retarget + eval) have never been run.**
2. **On the first retarget, scrutinize `retarget_visualize` frame 0 for ground alignment** — STMC
   smplrifke is z=0 canonical, GMR has an `offset_to_ground` convention; mismatch = body floats or clips
   the floor. This is the one known wiring risk. Everything else in the bridge is mechanical.
3. Then evaluate the tracker (mild_1p5 ckpt) on the bridged STMC motions and compare coverage/early-term
   to in-dist (~0.99) and to the composite proxy (0.66). If STMC locomotion tracks well → one-policy ↔
   text2motion link validated. If it drifts → that's the faithful failure to train against.
4. Training-side lever (later): train the tracker against long continuous in-repertoire sequences +
   STMC-shaped artifacts (20fps resample residual, smooth blends) to fight long-horizon yaw drift —
   NOT seam smoothing.
