#!/usr/bin/env python3
"""Bridge STMC-generated SMPL motion -> AMASS-style npz that MuscleMimic's GMR retargeting reads.

Context (task #16): we decided the hard-stitched MoCap composites are a biased proxy for STMC
(hard cuts vs smooth DiffCollage blends, inherited standing intros/outros, 180-deg turnarounds),
so the faithful OOD test is to feed REAL STMC output through the same GMR -> MyoFullBody tracker
pipeline. STMC's `extract_joints(..., jointstype="both")` already saves a `<name>_smpl.npz` with:
    poses (T,66)  axis-angle, 22 SMPL joints, no hands   -> exactly poses[:, :66]
    trans (T,3)   root translation
    joints(T,24,3)                                        (unused here)
    mocap_framerate () = 20.0                             (STMC fps)

MuscleMimic's `read_single_amass_motion` (loco_mujoco/smpl/retargeting.py:~293-311) needs an npz with
keys: poses, trans, betas, gender, mocap_framerate. So the bridge is nearly trivial -- the only
missing fields are `betas` and `gender`. Per docs/stmc_text2motion_interface.md we hand GMR a neutral
shape (betas=0, gender=neutral) and let `use_fitted_shape=True` fit the robot body; the 20fps tag is
mandatory (GMR resamples to 30fps from `mocap_framerate`, so a wrong/missing tag desyncs speed+phase
by 1.5x).

Output goes under the raw AMASS path as `STMC/<out_name>.npz` so the motion key is `STMC/<out_name>`,
which `fullbody/eval.py --motion_path STMC/<out_name>` (or retarget_visualize --motion STMC/<out_name>)
will retarget on first use and cache to caches/AMASS/<env>/gmr/STMC/<out_name>.npz. Optionally also
stages a copy to a shared-disk dir (default data/stmc_refs/) for cross-node eval, mirroring composites.

Usage:
  .venv/bin/python scripts/bridge_stmc_to_amass.py \
      --stmc_npz /path/to/<name>_smpl.npz \
      --out_name loco_walk_turn_walk_0 \
      [--amass_root /root/.musclemimic/AMASS] [--stage_dir data/stmc_refs] [--no_stage]

  # or bridge every *_smpl.npz under an STMC generations dir, names taken from filenames:
  .venv/bin/python scripts/bridge_stmc_to_amass.py --stmc_dir /path/to/<...>_to_motion/
"""
import argparse
import glob
import os
import shutil

import numpy as np


def _amass_root_default():
    # Resolve the same raw-AMASS path the retargeting pipeline reads (env/config/default).
    try:
        import loco_mujoco.core  # noqa: F401  (import order)
        from loco_mujoco.smpl.retargeting import get_amass_dataset_path
        return get_amass_dataset_path()
    except Exception:
        return os.path.expanduser("~/.musclemimic/AMASS")


def _yaw(poses_root_aa):
    """Net heading change (deg) from the root joint axis-angle column, for a quick sanity readout."""
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(poses_root_aa)  # (T,3) -> rotations
    # heading = rotation about gravity axis; use yaw of the euler ZYX as a rough proxy
    eul = rot.as_euler("zyx")  # (T,3): z=yaw
    yaw = np.unwrap(eul[:, 0])
    return np.degrees(yaw[-1] - yaw[0])


def bridge_one(stmc_npz, out_name, amass_root, stage_dir=None, verbose=True):
    d = np.load(stmc_npz, allow_pickle=True)
    poses = np.asarray(d["poses"], dtype=np.float32)
    trans = np.asarray(d["trans"], dtype=np.float32)
    assert poses.ndim == 2 and poses.shape[1] >= 66, f"poses shape {poses.shape} not (T,>=66)"
    assert trans.shape == (poses.shape[0], 3), f"trans shape {trans.shape} mismatch"
    fps = float(d["mocap_framerate"]) if "mocap_framerate" in d else 20.0

    # GMR's loader (general_motion_retargeting.utils.smpl.load_smplh_file) only accepts the AMASS
    # SMPL-H layout poses (T,156) = root(3)+body(63)+left_hand(45)+right_hand(45); on anything else it
    # falls to a branch that needs explicit root_orient/pose_body keys and KeyErrors. STMC gives only
    # root+body (66 = poses[:, :66]), which maps exactly to AMASS poses[:, :66]. Pad the 90 hand DOFs
    # with zeros (flat/neutral hands -- irrelevant to the locomotion repertoire) to trigger the AMASS
    # branch. poses[:, :66] is unchanged, so the non-GMR read_single_amass_motion path still works too.
    T = poses.shape[0]
    poses156 = np.zeros((T, 156), dtype=np.float32)
    poses156[:, :66] = poses[:, :66]

    out_npz = {
        "poses": poses156,                      # (T,156) AMASS SMPL-H: root+body from STMC, hands=0
        "trans": trans,                         # (T,3)
        "betas": np.zeros(16, dtype=np.float32),  # neutral shape; GMR fits the robot body
        "gender": "neutral",
        "mocap_framerate": np.float64(fps),     # MUST be 20 for STMC -> GMR resamples to 30
    }

    out_dir = os.path.join(amass_root, "STMC")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_name}.npz")
    np.savez(out_path, **out_npz)

    staged = None
    if stage_dir is not None:
        os.makedirs(stage_dir, exist_ok=True)
        staged = os.path.join(stage_dir, f"{out_name}.npz")
        shutil.copy2(out_path, staged)

    if verbose:
        disp = float(np.linalg.norm(trans[-1, :2] - trans[0, :2]))
        try:
            dyaw = _yaw(poses[:, 0:3])
        except Exception:
            dyaw = float("nan")
        print(f"[bridge] {out_name}: T={poses.shape[0]} ({poses.shape[0]/fps:.1f}s @ {fps:g}fps) "
              f"xy_disp={disp:.2f}m Δyaw={dyaw:+.0f}° -> {out_path}"
              + (f"  (staged {staged})" if staged else ""))
    return {"motion_key": f"STMC/{out_name}", "out_path": out_path, "staged": staged,
            "frames": int(poses.shape[0]), "fps": fps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stmc_npz", help="single STMC <name>_smpl.npz")
    g.add_argument("--stmc_dir", help="dir containing *_smpl.npz (batch); out_name from each filename")
    ap.add_argument("--out_name", help="output motion name (single mode). Default: derived from filename.")
    ap.add_argument("--amass_root", default=None, help="raw AMASS root (default: pipeline-resolved).")
    ap.add_argument("--stage_dir", default="data/stmc_refs", help="shared-disk staging dir.")
    ap.add_argument("--no_stage", action="store_true", help="skip staging copy.")
    args = ap.parse_args()

    amass_root = args.amass_root or _amass_root_default()
    stage_dir = None if args.no_stage else args.stage_dir
    print(f"[bridge] AMASS root = {amass_root}")

    def derive_name(p):
        b = os.path.basename(p)
        return b[:-len("_smpl.npz")] if b.endswith("_smpl.npz") else os.path.splitext(b)[0]

    results = []
    if args.stmc_npz:
        name = args.out_name or derive_name(args.stmc_npz)
        results.append(bridge_one(args.stmc_npz, name, amass_root, stage_dir))
    else:
        files = sorted(glob.glob(os.path.join(args.stmc_dir, "*_smpl.npz")))
        if not files:
            ap.error(f"no *_smpl.npz under {args.stmc_dir}")
        for p in files:
            results.append(bridge_one(p, derive_name(p), amass_root, stage_dir))

    print(f"\n[bridge] done: {len(results)} motion(s). Eval keys:")
    for r in results:
        print(f"  {r['motion_key']}  ({r['frames']} frames @ {r['fps']:g}fps)")


if __name__ == "__main__":
    main()
