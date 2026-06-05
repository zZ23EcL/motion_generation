#!/usr/bin/env python3
"""Localize and (optionally) fix jerk in STMC output at the SMPL source -- the decisive next experiment
after the frame-0 hypothesis was falsified (2026-06-05).

Where we are: scoped-STMC locomotion tracks at in-distribution quality (~0.998) and yaw drift is refuted
(a +469deg sustained-turn clip passes at 0.999). But 2 long multi-prompt clips fail and the cause is OPEN:
their death region has no single physically-impossible frame, just a reference that is "persistently jerky
/ never settles" (robot-level joint-vel-norm p10 ~0.78 vs ~0.3 for passing clips). Open question: is that
jerk (a) already in the STMC SMPL output (generation quality), or (b) introduced by GMR's per-frame IK?
And does removing it make the clip trackable (generation problem) or not (a real tracker edge)?

This tool works on the STMC `<name>_smpl.npz` (poses (T,66) axis-angle 22 joints, trans (T,3),
mocap_framerate=20) -- BEFORE retargeting -- so it sidesteps the re-FK consistency problem entirely:

  REPORT (default) -- jerk metrics on the SMPL source: per-frame pose angular speed (sum of inter-frame
    joint rotation angles) and root translation speed/accel, plus a "never-settles" indicator (the p10 of
    pose angular speed: low => the clip has quiet phases; high => persistently busy). Run it on a FAILING
    clip's _smpl.npz and a PASSING one: if the failing source is itself much jerkier (high p10, no quiet
    phases), the jerk is GENERATION; if the source is smooth, GMR introduced it.

  SMOOTH (--smooth) -- low-pass filter the source (rotation-aware 6D filtering of poses + Butterworth on
    trans), write a new `<name>_smooth_smpl.npz`. Then re-run the normal pipeline on it
    (bridge_stmc_to_amass.py -> GMR retarget -> eval). GMR regenerates xpos/site_xpos/cvel consistently,
    so no manual FK. If the smoothed clip then tracks ~0.99 => the failure was generation jerk; if it
    still fails => a genuine tracker limit on this content (escalate to tracker side).

Usage:
  python stmc_smpl_smooth.py <name>_smpl.npz
  python stmc_smpl_smooth.py <name>_smpl.npz --smooth --cutoff_hz 4 --out <name>_smooth_smpl.npz
  # then: bridge_stmc_to_amass.py --stmc_npz <name>_smooth_smpl.npz --out_name <name>_smooth ; retarget ; eval
"""
import argparse
import os

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


def pose_angular_speed(aa, fps):
    """aa: (T,J,3) axis-angle. Returns (T,) = sum over joints of inter-frame rotation angle * fps (rad/s)."""
    T, J, _ = aa.shape
    rots = R.from_rotvec(aa.reshape(T * J, 3)).as_matrix().reshape(T, J, 3, 3)
    # relative rotation between consecutive frames, per joint: angle = arccos((tr(R_t^T R_{t-1})-1)/2)
    rel = np.einsum("tjab,tjcb->tjac", rots[1:], rots[:-1])  # R_t @ R_{t-1}^T
    tr = np.clip((np.einsum("tjii->tj", rel) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)                                       # (T-1, J)
    speed = np.zeros(T)
    speed[1:] = ang.sum(axis=1) * fps
    return speed


def report(d, fps, label):
    poses = np.asarray(d["poses"], dtype=np.float64)
    trans = np.asarray(d["trans"], dtype=np.float64)
    T = poses.shape[0]
    aa = poses[:, :66].reshape(T, 22, 3)

    pas = pose_angular_speed(aa, fps)
    tspeed = np.zeros(T)
    tspeed[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=1) * fps      # m/s
    tacc = np.zeros(T)
    tacc[2:] = np.linalg.norm(np.diff(trans, n=2, axis=0), axis=1) * fps * fps  # m/s^2

    p = lambda a, q: float(np.percentile(a[1:], q)) if T > 1 else 0.0
    print(f"[smpl-jerk] {label}: T={T} ({T/fps:.1f}s @ {fps:g}fps)")
    print(f"  pose ang-speed (rad/s): med={p(pas,50):.2f} p10={p(pas,10):.2f} p90={p(pas,90):.2f} "
          f"max={pas.max():.2f}   <- p10 is the 'never-settles' signal (low=has quiet phases)")
    print(f"  trans speed (m/s):      med={p(tspeed,50):.2f} p90={p(tspeed,90):.2f} max={tspeed.max():.2f}")
    print(f"  trans accel (m/s^2):    med={p(tacc,50):.2f} p90={p(tacc,90):.2f} max={tacc.max():.2f}")
    return {"pose_p10": p(pas, 10), "pose_med": p(pas, 50), "pose_max": float(pas.max())}


def _lowpass(x, cutoff_hz, fps, order=4):
    """Zero-phase Butterworth low-pass along axis 0. Falls back to no-op if too short."""
    ny = fps / 2.0
    wn = min(cutoff_hz / ny, 0.99)
    if x.shape[0] <= 3 * (order + 1):
        return x
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x, axis=0)


def smooth_poses(aa, cutoff_hz, fps):
    """Rotation-aware low-pass: aa(T,J,3) -> rotmat -> 6D (continuous, no wrap) -> filter -> re-orthonormalize."""
    T, J, _ = aa.shape
    Rm = R.from_rotvec(aa.reshape(T * J, 3)).as_matrix().reshape(T, J, 3, 3)
    six = Rm[..., :, :2].reshape(T, J * 6)               # first two columns = 6D rep
    six_f = _lowpass(six, cutoff_hz, fps).reshape(T, J, 3, 2)
    # Gram-Schmidt back to a valid rotation
    a1, a2 = six_f[..., 0], six_f[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    Rf = np.stack([b1, b2, b3], axis=-1)                 # (T,J,3,3) columns
    aa_f = R.from_matrix(Rf.reshape(T * J, 3, 3)).as_rotvec().reshape(T, J, 3)
    return aa_f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="STMC <name>_smpl.npz")
    ap.add_argument("--smooth", action="store_true", help="write a low-pass-smoothed copy")
    ap.add_argument("--cutoff_hz", type=float, default=4.0, help="low-pass cutoff (locomotion is <~3Hz)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = dict(np.load(args.npz, allow_pickle=True))
    fps = float(d["mocap_framerate"]) if "mocap_framerate" in d else 20.0
    name = os.path.basename(args.npz)

    r0 = report(d, fps, name)

    if not args.smooth:
        return

    poses = np.asarray(d["poses"], dtype=np.float64)
    T = poses.shape[0]
    aa = poses[:, :66].reshape(T, 22, 3)
    aa_s = smooth_poses(aa, args.cutoff_hz, fps)
    poses_s = poses.copy()
    poses_s[:, :66] = aa_s.reshape(T, 66)
    trans_s = _lowpass(np.asarray(d["trans"], dtype=np.float64), args.cutoff_hz, fps)

    out = dict(d)
    out["poses"] = poses_s.astype(np.float32)
    out["trans"] = trans_s.astype(np.float32)
    outp = args.out or args.npz.replace("_smpl.npz", f"_smooth{int(args.cutoff_hz)}_smpl.npz")
    np.savez(outp, **out)
    print(f"\n[smooth] cutoff={args.cutoff_hz}Hz -> {outp}")
    print("[smooth] re-report on smoothed source:")
    report({"poses": poses_s, "trans": trans_s, "mocap_framerate": fps}, fps, os.path.basename(outp))
    print(f"\n  next: bridge_stmc_to_amass.py --stmc_npz {outp} --out_name <name>_smooth ; retarget ; eval")


if __name__ == "__main__":
    main()
