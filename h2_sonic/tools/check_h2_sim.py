#!/usr/bin/env python3
"""Smoke test for the H2 MuJoCo sim setup (no DDS / unitree_sdk2py needed).

Checks, against a patched GR00T-WholeBodyControl checkout:
  1. the H2 scene MJCF loads and has 31 actuated joints + floating base
  2. actuator order == joint order (so the identity MOTOR2JOINT map is valid)
  3. every per-joint list in the WBC yaml has length 31 and
     WeakMotorJointIndex matches the MJCF joint order
  4. yaml position limits equal the MJCF joint ranges
  5. base_sim.py's joint-name filter would pick up all 31 joints (head included)
  6. a 3 s headless PD-hold rollout at DEFAULT_DOF_ANGLES stays finite
     (same control law base_sim.py applies, torques clipped to effort limits)

Usage:
    python tools/check_h2_sim.py --upstream <path-to-GR00T-WholeBodyControl>
"""

import argparse
import os
import re
import sys

import mujoco
import numpy as np
import yaml

BODY_JOINT_PARTS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "head"]


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    args = ap.parse_args()
    up = args.upstream

    cfg_path = os.path.join(up, "gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    scene_path = os.path.join(up, cfg["ROBOT_SCENE"])

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    print(f"loaded {cfg['ROBOT_SCENE']}: nq={model.nq} nv={model.nv} nu={model.nu}")

    joint_names = []
    has_free = False
    for i in range(model.njnt):
        j = model.joint(i)
        if j.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            has_free = True
            continue
        if j.name in ("", "floor"):
            continue
        joint_names.append(j.name)
    if not has_free:
        fail("no floating base joint in scene")
    if len(joint_names) != 31:
        fail(f"expected 31 actuated joints, got {len(joint_names)}")

    act_joints = [model.joint(model.actuator(i).trnid[0]).name for i in range(model.nu)]
    if act_joints != joint_names:
        fail("actuator order != joint order (MOTOR2JOINT identity is wrong)")
    print("OK: 31 joints, actuator order matches joint order")

    n = cfg["NUM_MOTORS"]
    if n != 31 or cfg["NUM_JOINTS"] != 31:
        fail("NUM_MOTORS/NUM_JOINTS != 31")
    for key in [
        "MOTOR2JOINT", "JOINT2MOTOR", "MOTOR_KP", "MOTOR_KD", "JOINT_KP", "JOINT_KD",
        "DEFAULT_DOF_ANGLES", "DEFAULT_MOTOR_ANGLES", "motor_pos_lower_limit_list",
        "motor_pos_upper_limit_list", "motor_vel_limit_list", "motor_effort_limit_list",
    ]:
        if len(cfg[key]) != 31:
            fail(f"{key} has length {len(cfg[key])}, expected 31")
    weak = cfg["WeakMotorJointIndex"]
    for i, name in enumerate(joint_names):
        if weak.get(name) != i:
            fail(f"WeakMotorJointIndex[{name}] = {weak.get(name)}, expected {i}")
    print("OK: all yaml per-joint lists have length 31, WeakMotorJointIndex matches MJCF order")

    for i, name in enumerate(joint_names):
        rng = model.joint(name).range
        lo, hi = cfg["motor_pos_lower_limit_list"][i], cfg["motor_pos_upper_limit_list"][i]
        if abs(lo - rng[0]) > 1e-4 or abs(hi - rng[1]) > 1e-4:
            fail(f"pos limits for {name}: yaml [{lo}, {hi}] vs mjcf [{rng[0]}, {rng[1]}]")
    print("OK: yaml position limits match MJCF joint ranges")

    matched = [nm for nm in joint_names if any(p in nm for p in BODY_JOINT_PARTS)]
    if len(matched) != 31:
        missing = set(joint_names) - set(matched)
        fail(f"base_sim joint-name filter misses: {missing}")
    upper = joint_names[12:]
    if len(upper) != cfg["NUM_UPPER_BODY_JOINTS"]:
        fail(f"NUM_UPPER_BODY_JOINTS={cfg['NUM_UPPER_BODY_JOINTS']} but MJCF has {len(upper)}")
    print("OK: base_sim name filter covers all 31 joints; 19 upper-body joints")

    # headless PD hold at the default pose, mirroring base_sim.compute_body_torques
    kp = np.array(cfg["MOTOR_KP"], dtype=float)
    kd = np.array(cfg["MOTOR_KD"], dtype=float)
    q_des = np.array(cfg["DEFAULT_DOF_ANGLES"], dtype=float)
    effort = np.array(cfg["motor_effort_limit_list"], dtype=float)
    model.opt.timestep = cfg["SIMULATE_DT"]
    data.qpos[7 : 7 + 31] = q_des
    mujoco.mj_forward(model, data)
    steps = int(3.0 / cfg["SIMULATE_DT"])
    for _ in range(steps):
        q = data.qpos[7 : 7 + 31]
        dq = data.qvel[6 : 6 + 31]
        tau = kp * (q_des - q) - kd * dq
        data.ctrl[:31] = np.clip(tau, -effort, effort)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            fail("simulation went non-finite during PD hold")
    height = data.qpos[2]
    err = np.abs(data.qpos[7 : 7 + 31] - q_des)
    print(f"OK: 3s PD-hold rollout finite; pelvis height {height:.3f} m, "
          f"max joint err {err.max():.3f} rad ({joint_names[int(err.argmax())]})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
