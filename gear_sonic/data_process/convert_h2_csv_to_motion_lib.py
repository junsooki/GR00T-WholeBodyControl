"""Convert retargeted H2 CSV motions into SONIC motion-library PKLs.

The H2 counterpart of convert_soma_csv_to_motion_lib.py. Same CSV convention as
Bones-SEED (Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, <joint>_dof), but
31 DOF / 32 bodies instead of G1's 29 / 30.

Input CSV, one file per motion:
    Frame, root_translateX/Y/Z (cm), root_rotateX/Y/Z (deg, intrinsic xyz),
    then one <joint>_dof column per H2 joint (deg).

Output PKL, joblib, one dict keyed by motion name:
    root_trans_offset (T, 3)      float32, meters
    pose_aa           (T, 32, 3)  float32, axis-angle per body, MuJoCo body order
    dof               (T, 31)     float32, radians, MuJoCo actuator order
    root_rot          (T, 4)      float32, *xyzw* (scipy convention)
    smpl_joints       (T, 24, 3)  float32, zeros unless SMPL data is supplied
    fps               int

Note on root_rot: the SONIC docs describe this field as wxyz, but the G1
converter writes xyzw and motion_lib performs the xyzw->wxyz conversion at load
time. We match the code, not the docs -- writing wxyz here trains on a silently
wrong root orientation.

Usage:
    python gear_sonic/data_process/convert_h2_csv_to_motion_lib.py \
        --input /path/to/h2_retargeted_csvs/ \
        --output data/h2_motions/robot \
        --fps 30 --fps_source 120 --individual --num_workers 16
"""

import argparse
import multiprocessing
import os
import sys

import joblib
import numpy as np
from scipy.spatial import transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2_constants import H2_CSV_JOINT_NAMES, H2_DOF_AXIS, NUM_BODIES, NUM_DOF  # noqa: E402


def load_h2_csv(csv_path: str) -> dict:
    """Load one retargeted H2 CSV. Angles in degrees, positions in centimeters."""
    import pandas as pd

    data = pd.read_csv(csv_path)

    root_pos = (
        np.stack(
            [data["root_translateX"].values, data["root_translateY"].values, data["root_translateZ"].values],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )

    euler_deg = np.stack(
        [data["root_rotateX"].values, data["root_rotateY"].values, data["root_rotateZ"].values],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )

    # Select DOF columns *by name* in MuJoCo actuator order. The G1 converter
    # relies on CSV column order matching actuator order; being explicit here
    # turns a silent scramble into a clear error.
    missing = [c for c in H2_CSV_JOINT_NAMES if c not in data.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing {len(missing)} DOF columns, first few: {missing[:5]}")
    dof = np.deg2rad(data[H2_CSV_JOINT_NAMES].values).astype(np.float32)

    return {"root_pos": root_pos, "root_quat_xyzw": root_quat_xyzw, "dof": dof}


def convert_sequence(seq: dict, fps: int) -> dict:
    """Build a motion_lib entry from a loaded CSV sequence."""
    dof = seq["dof"]
    T = dof.shape[0]
    if dof.shape[1] != NUM_DOF:
        raise ValueError(f"expected {NUM_DOF} DOF, got {dof.shape[1]}")

    # pose_aa[0] is the root rotvec; pose_aa[k+1] is joint k's axis scaled by its angle.
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(seq["root_quat_xyzw"]).as_rotvec()
    pose_aa[:, 1:NUM_BODIES, :] = H2_DOF_AXIS[None, :, :] * dof[:, :, None]

    return {
        "root_trans_offset": seq["root_pos"].astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": seq["root_quat_xyzw"].astype(np.float32),  # xyzw
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": fps,
    }


def downsample_sequence(entry: dict, fps_source: int, fps_target: int) -> dict:
    """Stride-downsample, matching the G1 converter's behaviour."""
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    return {
        "root_trans_offset": entry["root_trans_offset"][::jump],
        "pose_aa": entry["pose_aa"][::jump],
        "dof": entry["dof"][::jump],
        "root_rot": entry["root_rot"][::jump],
        "smpl_joints": entry["smpl_joints"][::jump],
        "fps": fps_target,
    }


def process_session(job) -> tuple:
    session_dir, session_name, out_root, fps, fps_source = job
    out_dir = os.path.join(out_root, session_name)
    os.makedirs(out_dir, exist_ok=True)

    csvs = sorted(f for f in os.listdir(session_dir) if f.endswith(".csv"))
    converted = failed = 0
    for fn in csvs:
        name = os.path.splitext(fn)[0]
        try:
            entry = convert_sequence(load_h2_csv(os.path.join(session_dir, fn)), fps)
            if fps_source and fps_source != fps:
                entry = downsample_sequence(entry, fps_source, fps)
            joblib.dump({name: entry}, os.path.join(out_dir, f"{name}.pkl"))
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {session_name}/{fn}: {exc}")
            failed += 1
    return session_name, converted, failed, len(csvs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Directory of session dirs (or one session dir)")
    p.add_argument("--output", required=True, help="Output root for PKLs")
    p.add_argument("--fps", type=int, default=30, help="Target output FPS")
    p.add_argument("--fps_source", type=int, default=None, help="Source FPS (Bones-SEED is 120)")
    p.add_argument("--individual", action="store_true", help="One PKL per motion")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    print(f"H2 {NUM_DOF} DOFs, {NUM_BODIES} bodies (axes derived from h2.xml)")
    if not os.path.isdir(args.input):
        sys.exit("ERROR: --input must be a directory")

    subdirs = sorted(d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d)))
    has_session_subdirs = any(
        any(f.endswith(".csv") for f in os.listdir(os.path.join(args.input, d))) for d in subdirs[:3]
    )

    jobs = []
    if has_session_subdirs:
        for d in subdirs:
            sub = os.path.join(args.input, d)
            if any(f.endswith(".csv") for f in os.listdir(sub)):
                jobs.append((sub, d, args.output, args.fps, args.fps_source))
    elif any(f.endswith(".csv") for f in os.listdir(args.input)):
        jobs.append(
            (args.input, os.path.basename(args.input.rstrip("/")), args.output, args.fps, args.fps_source)
        )

    os.makedirs(args.output, exist_ok=True)
    print(f"Converting {len(jobs)} sessions with {args.num_workers} workers -> {args.output}")

    total_c = total_f = total_n = 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for name, c, f, n in pool.imap_unordered(process_session, jobs):
            total_c += c
            total_f += f
            total_n += n
            print(f"  {name}: {c}/{n} converted" + (f" ({f} failed)" if f else ""))

    print(f"\nDone: {total_c}/{total_n} motions converted, {total_f} failed")


if __name__ == "__main__":
    main()
