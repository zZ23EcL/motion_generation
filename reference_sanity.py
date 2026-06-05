#!/usr/bin/env python3
"""Reference-quality sanity check + cleaning for GMR-retargeted reference trajectories.

Why (2026-06-05): the faithful STMC test showed scoped-locomotion tracks at in-distribution quality
(~0.998 coverage) and REFUTED the long-horizon yaw-drift hypothesis (a +469deg sustained-turn clip
passes at 0.999). The only stress-test failures were multi-prompt STMC DiffCollage timelines whose
RETARGETED REFERENCE is physically implausible -- a generation/bridge quality problem, not a tracker
robustness gap. The two signatures observed:
  1. a frame-0 velocity transient (joint-vel norm ~11 vs ~3.7 normal), dominated by a ~6 rad/s
     pro_sup forearm-pronation spin -- plausibly a qvel boundary artifact at the first frame;
  2. a mid-clip blow-up (root angular velocity 35.8 rad/s, a 0.70-rad single-frame qpos jump) --
     an unfollowable retargeting glitch.

This tool operates on the GMR-retargeted cache npz
(`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/<name>.npz`, the robot-level Trajectory:
qpos (T,89), qvel (T,88), plus FK-derived xpos/xquat/site_xpos/...). It does two things, neither of
which needs GMR or mujoco:

  REPORT (default) -- per-frame joint-vel norm, root angular-vel magnitude, single-frame qpos jump;
    a frame-0 boundary check (is frame 0 an outlier vs the settled clip?); the dominant DOF at frame 0
    and at each flagged frame; and a one-line VERDICT (CLEAN / FRAME0_TRANSIENT / INTERIOR_BLOWUP /
    BOTH). This is the deployment guard signal (is this reference followable?) and the artifact-rate
    instrument (run over N clips, count non-CLEAN).

  TRIM (--trim_lead K or --trim_range a:b) -- slice ALL time-indexed arrays consistently and re-save,
    producing an FK-consistent cleaned reference. Trimming the opening K frames is the cheap test of the
    frame-0-transient hypothesis: if the trimmed clip then tracks ~0.99, the failure was a boundary
    artifact, not the tracker. (Interior blow-ups can't be patched without re-running FK -- the tool
    flags them and you should re-generate or trim to before them.)

Usage:
  python reference_sanity.py <cache.npz>
  python reference_sanity.py <cache.npz> --trim_lead 5  --out <cache_trim.npz>
  python reference_sanity.py <cache.npz> --trim_range 10:520 --out <cache_clip.npz>
  # batch artifact-rate scan:
  for f in ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/STMC/*.npz; do python reference_sanity.py "$f"; done
"""
import argparse
import os

import numpy as np

# The 8 arrays whose first axis is time T in the robot-level Trajectory cache. Everything else
# (joint_names, body_*, site_*, split_points, metadata, njnt, frequency, *_parent ...) is static.
TIME_KEYS = ("qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")

# qpos: [0:3] root pos, [3:7] root quat (wxyz), [7:] 82 joint angles (joint_names[1:])
# qvel: [0:3] root lin vel, [3:6] root ang vel, [6:] 82 joint velocities (joint_names[1:])
ROOT_QVEL = ("root_vx", "root_vy", "root_vz", "root_wx", "root_wy", "root_wz")


def dof_name(qvel_idx, joint_names):
    """Name the qvel column. 0..5 = root free-joint; >=6 maps to joint_names[1 + (idx-6)]."""
    if qvel_idx < 6:
        return ROOT_QVEL[qvel_idx]
    ji = 1 + (qvel_idx - 6)
    return str(joint_names[ji]) if ji < len(joint_names) else f"q{qvel_idx}"


def analyze(d, args):
    """Two independent discriminators (absolute joint-vel norm is NOT one of them: normal MyoFullBody
    gait legitimately spikes jnorm to ~13 during knee swing, so it's reported for info only):
      * frame-0 transient -> the GMR qvel boundary artifact; detected by ratio(jnorm[0] / settled median)
        above --frame0_ratio AND an absolute floor --frame0_floor (so low-median clips don't false-fire).
      * interior blow-up   -> physically-impossible reference; detected by root angular-vel > --rootang_thresh
        (gait/turning is <2 rad/s, the observed glitch was 35.8) OR single-frame joint jump > --jump_thresh.
    """
    qpos, qvel = np.asarray(d["qpos"]), np.asarray(d["qvel"])
    jn = d["joint_names"]
    T = qpos.shape[0]

    jnorm = np.linalg.norm(qvel[:, 6:], axis=1)          # joint-vel norm (info only)
    rootang = np.linalg.norm(qvel[:, 3:6], axis=1)       # root angular-vel magnitude (discriminator)

    qjump = np.zeros(T)                                   # single-frame max joint-angle jump (discriminator)
    if T > 1:
        qjump[1:] = np.abs(np.diff(qpos[:, 7:], axis=0)).max(axis=1)

    # frame-0 boundary check vs the settled window [1:win]
    w = slice(1, min(args.win + 1, T))
    med_jnorm = float(np.median(jnorm[w])) if T > 1 else float(jnorm[0])
    ratio = float(jnorm[0] / med_jnorm) if med_jnorm > 1e-6 else float("inf")
    frame0_transient = (ratio > args.frame0_ratio) and (float(jnorm[0]) > args.frame0_floor)
    f0 = {"jnorm": float(jnorm[0]), "jnorm_med": med_jnorm, "ratio": ratio,
          "rootang": float(rootang[0]), "transient": frame0_transient, "top_dof": []}
    for v in np.argsort(-np.abs(qvel[0]))[:5]:
        f0["top_dof"].append((dof_name(int(v), jn), float(qvel[0, int(v)])))

    # interior flags (skip the frame-0 window; those are covered by the boundary check)
    flagged = []
    for t in range(args.frame0_window, T):
        reasons = []
        if rootang[t] > args.rootang_thresh:
            reasons.append(f"rootang={rootang[t]:.1f}")
        if qjump[t] > args.jump_thresh:
            reasons.append(f"qjump={qjump[t]:.2f}")
        if reasons:
            dom = int(np.argmax(np.abs(qvel[t])))
            flagged.append((t, reasons, dof_name(dom, jn), float(qvel[t, dom])))

    # "never-settles" correlate (the signal that actually tracked failure in the stress test:
    # fails have jnorm p10 ~0.78 vs ~0.3 for passing clips -- i.e. the clip never reaches a quiet phase)
    jnorm_p10 = float(np.percentile(jnorm, 10)) if T else 0.0

    return {"T": T, "jnorm": jnorm, "rootang": rootang, "qjump": qjump,
            "f0": f0, "flagged": flagged, "med_jnorm": med_jnorm, "jnorm_p10": jnorm_p10}


def verdict(res):
    """CLEAN / FRAME0_TRANSIENT / INTERIOR_BLOWUP / BOTH."""
    has_f0 = res["f0"]["transient"]
    has_interior = len(res["flagged"]) > 0
    if has_f0 and has_interior:
        return "BOTH"
    if has_f0:
        return "FRAME0_TRANSIENT"
    if has_interior:
        return "INTERIOR_BLOWUP"
    return "CLEAN"


def do_report(path, args):
    d = dict(np.load(path, allow_pickle=True))
    res = analyze(d, args)
    T = res["T"]
    v = verdict(res)
    name = os.path.basename(path)
    freq = float(d["frequency"]) if "frequency" in d else float("nan")

    # DEPLOYMENT GATE (the settled lever): a scoped-STMC reference is followable iff it has no
    # physically-impossible interior frame AND it actually settles (reaches a quiet stance phase).
    # never-settles is the resolved failure mode (over-agitated generation content; NOT smoothing-fixable),
    # so it is a reject-and-REGENERATE signal, not something to filter. frame-0 transient is ignored here
    # (falsified boundary effect). Threshold --settle_thresh calibrated on n=7 (5 pass ~0.3 vs 2 fail ~0.78).
    never_settles = res["jnorm_p10"] > args.settle_thresh
    reject_reasons = []
    if res["flagged"]:
        reject_reasons.append("interior-blowup")
    if never_settles:
        reject_reasons.append(f"never-settles(p10={res['jnorm_p10']:.2f}>{args.settle_thresh})")
    gate = "REJECT" if reject_reasons else "ACCEPT"

    # one-line machine-greppable summary first (good for batch scans)
    print(f"[sanity] {name}: GATE={gate}  VERDICT={v}  T={T} ({T/freq:.1f}s)  "
          f"jnorm_p10={res['jnorm_p10']:.2f}  rootang_max={res['rootang'].max():.2f}  "
          f"qjump_max={res['qjump'].max():.3f}  interior_flags={len(res['flagged'])}"
          + (f"  reject:{','.join(reject_reasons)}" if reject_reasons else ""))
    if args.quiet:
        return v

    f0 = res["f0"]
    print(f"  frame-0 boundary: jnorm[0]={f0['jnorm']:.2f} vs settled-median {f0['jnorm_med']:.2f} "
          f"(ratio {f0['ratio']:.1f}x){'  <-- TRANSIENT' if f0['transient'] else ''}")
    print(f"  frame-0 top DOFs by |vel|: " +
          ", ".join(f"{n}={val:+.1f}" for n, val in f0["top_dof"]))
    if res["flagged"]:
        print(f"  interior blow-up frames ({len(res['flagged'])}):")
        for (t, reasons, dom, dval) in res["flagged"][:args.max_list]:
            print(f"    f{t:5d} ({t/freq:5.2f}s): {', '.join(reasons)}  dominant={dom}({dval:+.1f})")
        if len(res["flagged"]) > args.max_list:
            print(f"    ... +{len(res['flagged']) - args.max_list} more")
    # never-settles correlate (the signal that actually tracked stress-test failure)
    print(f"  never-settles: jnorm p10={res['jnorm_p10']:.2f}  "
          f"(stress fails ~0.78, passes ~0.3 -> high p10 = persistently jerky, the real failure correlate)")
    # actionable hints
    if f0["transient"]:
        print(f"  note: frame-0 transient present (ratio {f0['ratio']:.1f}x) but this is a near-universal "
              f"GMR qvel BOUNDARY EFFECT -- FALSIFIED as a failure cause (a passing clip has a bigger one, "
              f"and trimming it does NOT fix tracking). Not actionable.")
    if v in ("INTERIOR_BLOWUP", "BOTH"):
        bad = [t for (t, *_2) in res["flagged"]]
        print(f"  -> interior glitch at frame(s) {bad[:5]} -- physically-implausible reference, valid HARD "
              f"REJECT; re-generate (different seed) or --trim_range to before {min(bad)}.")
    if never_settles:
        print(f"  -> NEVER-SETTLES (jnorm p10={res['jnorm_p10']:.2f}): over-agitated generation content, no "
              f"stance phase. RESOLVED as not smoothing-fixable (the p10 floor is low-freq content, not "
              f"jitter) -> GATE = reject-and-REGENERATE this STMC sample (do NOT trim/smooth).")
    if gate == "ACCEPT":
        print("  -> GATE=ACCEPT: settles to a stance phase, no impossible frame -> followable reference.")
    return v


def do_trim(path, args):
    d = dict(np.load(path, allow_pickle=True))
    T = int(np.asarray(d["qpos"]).shape[0])
    if args.trim_range:
        a, b = (int(x) for x in args.trim_range.split(":"))
    else:
        a, b = args.trim_lead, T
    a = max(0, a)
    b = T if b in (-1, 0) else min(b, T)
    assert 0 <= a < b <= T, f"bad trim [{a}:{b}] for T={T}"
    new_T = b - a

    # sanity: warn if any unexpected array is time-indexed (would desync if not sliced)
    for k, val in d.items():
        if k in TIME_KEYS:
            continue
        if hasattr(val, "shape") and val.ndim >= 1 and len(val.shape) and val.shape[0] == T:
            print(f"[trim][warn] key '{k}' has first-dim==T but is not in TIME_KEYS; copying unsliced.")

    out = {}
    for k, val in d.items():
        if k in TIME_KEYS:
            out[k] = np.asarray(val)[a:b]
        elif k == "split_points":
            out[k] = np.array([0, new_T], dtype=np.asarray(val).dtype)
        else:
            out[k] = val

    outp = args.out or path.replace(".npz", f"_trim{a}_{b}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    np.savez(outp, **out)
    print(f"[trim] {os.path.basename(path)} [{a}:{b}] -> {new_T} frames  ->  {outp}")
    # re-report the trimmed result so you can see frame-0 is now clean
    print("[trim] re-report on trimmed clip:")
    args2 = argparse.Namespace(**{**vars(args), "quiet": False})
    do_report(outp, args2)
    return outp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="GMR-retargeted cache npz (robot-level Trajectory)")
    ap.add_argument("--rootang_thresh", type=float, default=10.0,
                    help="interior root angular-vel flag rad/s (even aggressive turning <2; 10 = impossible)")
    ap.add_argument("--jump_thresh", type=float, default=0.3, help="interior single-frame max joint jump (rad)")
    ap.add_argument("--settle_thresh", type=float, default=0.5,
                    help="GATE rejects if jnorm p10 exceeds this (never-settles). Calibrated n=7: pass ~0.3, fail ~0.78.")
    ap.add_argument("--frame0_ratio", type=float, default=1.8,
                    help="frame-0 transient if jnorm[0]/settled-median exceeds this")
    ap.add_argument("--frame0_floor", type=float, default=6.0,
                    help="...and jnorm[0] absolute exceeds this (avoids firing on low-median clips)")
    ap.add_argument("--frame0_window", type=int, default=3, help="frames treated as the boundary region")
    ap.add_argument("--win", type=int, default=20, help="settled-window length for frame-0 comparison")
    ap.add_argument("--max_list", type=int, default=20, help="max flagged frames to print")
    ap.add_argument("--quiet", action="store_true", help="one-line summary only (for batch scans)")
    ap.add_argument("--trim_lead", type=int, default=None, help="drop the first K frames, then re-save")
    ap.add_argument("--trim_range", default=None, help="keep frames a:b (e.g. 10:520), then re-save")
    ap.add_argument("--out", default=None, help="output path for trimmed npz")
    args = ap.parse_args()

    if args.trim_lead is not None or args.trim_range is not None:
        if args.trim_lead is None:
            args.trim_lead = 0
        do_trim(args.npz, args)
    else:
        do_report(args.npz, args)


if __name__ == "__main__":
    main()
