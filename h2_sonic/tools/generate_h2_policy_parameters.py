#!/usr/bin/env python3
"""Generate deploy/h2_policy_parameters.hpp — the H2 analog of
gear_sonic_deploy's policy_parameters.hpp (which is hardcoded for the G1).

Sources:
  - IsaacLab<->MuJoCo DOF/body mappings: copied verbatim from
    gear_sonic/envs/manager_env/robots/h2.py (H2_ISAACLAB_TO_MUJOCO_MAPPING)
  - PD gains / armatures / action-scale effort limits: same file (H2_CFG)
  - default angles: H2_CFG.init_state
  - joint order: H2 MJCF tree order (verified against the mappings)

Derived arrays (upper/lower-body index lists, vr point indices) are computed
from the mappings, not hand-typed.

Usage:
    python tools/generate_h2_policy_parameters.py [--out deploy/h2_policy_parameters.hpp]
"""

import argparse
import os

# ---- verbatim from gear_sonic/envs/manager_env/robots/h2.py ----------------
H2_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11,
    16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30,
]
H2_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10,
    16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30,
]
# body list (isaaclab order), pelvis first
H2_ISAACLAB_BODIES = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "head_pitch_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "head_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]

# H2 MJCF tree order (== actuator order; verified by tools/check_h2_sim.py)
MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
N = 31

# (stiffness constant expr, damping constant expr, effort_limit_sim, multiplier)
# straight from H2_CFG.actuators in robots/h2.py; effort limits here are the
# TRAINING values (used for action_scale), not the MJCF clip values.
GAINS = {
    "hip_pitch": ("7520_22", 417.0, 1.0),
    "hip_roll": ("7520_22", 417.0, 1.0),
    "hip_yaw": ("7520_14", 264.0, 1.0),
    "knee": ("7520_22", 417.0, 1.0),
    "ankle_roll": ("5020", 150.0, 2.0),
    "ankle_pitch": ("5020", 150.0, 2.0),
    "waist_yaw": ("7520_14", 264.0, 1.0),
    "waist_roll": ("5020", 150.0, 2.0),
    "waist_pitch": ("5020", 150.0, 2.0),
    "head_pitch": ("5020", 150.0, 2.0),
    "head_yaw": ("5020", 150.0, 2.0),
    "shoulder_pitch": ("5020", 75.0, 1.0),
    "shoulder_roll": ("5020", 75.0, 1.0),
    "shoulder_yaw": ("5020", 75.0, 1.0),
    "elbow": ("5020", 75.0, 1.0),
    "wrist_roll": ("5020", 75.0, 1.0),
    "wrist_pitch": ("4010", 15.0, 1.0),
    "wrist_yaw": ("4010", 15.0, 1.0),
}

DEFAULT_ANGLES = {
    "hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
}
SIDE_ANGLES = {
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
}


def joint_kind(name):
    for kind in GAINS:
        if kind in name:
            return kind
    raise KeyError(name)


def default_angle(name):
    if name in SIDE_ANGLES:
        return SIDE_ANGLES[name]
    for kind, v in DEFAULT_ANGLES.items():
        if kind in name:
            return v
    return 0.0


def carr(type_, name, vals, size=None, per_line=8):
    size = size if size is not None else len(vals)
    body = ""
    for i in range(0, len(vals), per_line):
        body += "    " + ", ".join(str(v) for v in vals[i : i + per_line]) + ",\n"
    return f"const std::array<{type_}, {size}> {name} = {{\n{body[:-2]}\n}};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "deploy/h2_policy_parameters.hpp"),
    )
    args = ap.parse_args()

    # sanity: the two mappings must be inverses
    for i in range(N):
        assert H2_MUJOCO_TO_ISAACLAB_DOF[H2_ISAACLAB_TO_MUJOCO_DOF[i]] == i

    kinds = [joint_kind(j) for j in MUJOCO_JOINTS]
    kp_expr = [
        (f"{m:.1f} * " if (m := GAINS[k][2]) != 1.0 else "") + f"STIFFNESS_{GAINS[k][0]}"
        for k in kinds
    ]
    kd_expr = [
        (f"{m:.1f} * " if (m := GAINS[k][2]) != 1.0 else "") + f"DAMPING_{GAINS[k][0]}"
        for k in kinds
    ]
    # action_scale = 0.25 * effort_limit / stiffness (training values)
    scale_expr = [
        f"0.25 * {GAINS[k][1]} / ({GAINS[k][2]} * STIFFNESS_{GAINS[k][0]})" for k in kinds
    ]
    defaults = [default_angle(j) for j in MUJOCO_JOINTS]

    # annotate arrays with joint names
    def annotated(type_, name, exprs, fmt=str):
        rows = "".join(
            f"    {fmt(e)}, // {j}\n" for e, j in zip(exprs, MUJOCO_JOINTS)
        )
        return f"const std::array<{type_}, {N}> {name} = {{\n{rows}}};\n"

    # body-part index lists (both orderings), derived from the mappings
    lower_mj = list(range(12))
    upper_mj = list(range(12, 31))
    wrist_mj = [i for i, j in enumerate(MUJOCO_JOINTS) if "wrist" in j]

    def in_isaaclab(mj_indices):
        return [H2_MUJOCO_TO_ISAACLAB_DOF[i] for i in mj_indices]

    # isaaclab-order variants: sort by isaaclab index
    lower_il_pairs = sorted((H2_MUJOCO_TO_ISAACLAB_DOF[i], i) for i in lower_mj)
    upper_il_pairs = sorted((H2_MUJOCO_TO_ISAACLAB_DOF[i], i) for i in upper_mj)
    wrist_il_pairs = sorted((H2_MUJOCO_TO_ISAACLAB_DOF[i], i) for i in wrist_mj)

    b = H2_ISAACLAB_BODIES.index
    vr3 = [b("left_wrist_yaw_link"), b("right_wrist_yaw_link"), b("torso_link")]
    vr5 = [b("left_wrist_yaw_link"), b("right_wrist_yaw_link"), b("pelvis"),
           b("left_ankle_pitch_link"), b("right_ankle_pitch_link")]

    def cvec(name, vals):
        return f"const std::vector<int> {name} = {{{', '.join(map(str, vals))}}};\n"

    out = f"""/**
 * @file h2_policy_parameters.hpp
 * @brief H2 (31 DOF) analog of policy_parameters.hpp.
 *
 * GENERATED by tools/generate_h2_policy_parameters.py -- edit the generator.
 *
 * Joint order: H2 MJCF tree order (== actuator order). Note vs G1:
 * ankle is ROLL then PITCH, and head pitch/yaw sit at MuJoCo indices 15-16
 * between waist and arms. Upper body = waist 3 + head 2 + arms 14 = 19.
 *
 * Mappings copied from gear_sonic/envs/manager_env/robots/h2.py
 * (H2_ISAACLAB_TO_MUJOCO_MAPPING); gains/action scales computed exactly as
 * that file does (stiffness = armature * (2*pi*10Hz)^2, damping = 2 * zeta *
 * armature * 2*pi*10Hz, zeta = 2; action_scale = 0.25 * effort / stiffness
 * with TRAINING effort limits).
 */

#ifndef H2_POLICY_PARAMETERS_HPP
#define H2_POLICY_PARAMETERS_HPP

#include <array>
#include <vector>

const int H2_NUM_MOTOR = {N};
const int H2_NUM_UPPER_BODY_JOINTS = 19;
const int H2_NUM_LOWER_BODY_JOINTS = 12;

// Motor armature constants (same motor families as G1)
const double H2_ARMATURE_5020 = 0.003609725;
const double H2_ARMATURE_7520_14 = 0.010177520;
const double H2_ARMATURE_7520_22 = 0.025101925;
const double H2_ARMATURE_4010 = 0.00425;

const double H2_NATURAL_FREQ = 10 * 2.0 * 3.1415926535; // 10 Hz
const double H2_DAMPING_RATIO = 2.0;

#define STIFFNESS_OF(a) ((a) * H2_NATURAL_FREQ * H2_NATURAL_FREQ)
#define DAMPING_OF(a) (2.0 * H2_DAMPING_RATIO * (a) * H2_NATURAL_FREQ)
const double STIFFNESS_5020 = STIFFNESS_OF(H2_ARMATURE_5020);
const double STIFFNESS_7520_14 = STIFFNESS_OF(H2_ARMATURE_7520_14);
const double STIFFNESS_7520_22 = STIFFNESS_OF(H2_ARMATURE_7520_22);
const double STIFFNESS_4010 = STIFFNESS_OF(H2_ARMATURE_4010);
const double DAMPING_5020 = DAMPING_OF(H2_ARMATURE_5020);
const double DAMPING_7520_14 = DAMPING_OF(H2_ARMATURE_7520_14);
const double DAMPING_7520_22 = DAMPING_OF(H2_ARMATURE_7520_22);
const double DAMPING_4010 = DAMPING_OF(H2_ARMATURE_4010);
#undef STIFFNESS_OF
#undef DAMPING_OF

// IsaacLab <-> MuJoCo DOF mappings (from robots/h2.py, verified inverses)
{carr("int", "h2_isaaclab_to_mujoco", H2_ISAACLAB_TO_MUJOCO_DOF)}
{carr("int", "h2_mujoco_to_isaaclab", H2_MUJOCO_TO_ISAACLAB_DOF)}
// Body-part joint index lists.
// mujoco order = iterate joints in mujoco order; isaaclab order = sorted by
// isaaclab index. "*_in_isaaclab_index" gives the isaaclab dof index of each,
// "*_in_mujoco_index" the mujoco dof index.
{cvec("h2_lower_body_joint_mujoco_order_in_isaaclab_index", in_isaaclab(lower_mj))}
{cvec("h2_lower_body_joint_mujoco_order_in_mujoco_index", lower_mj)}
{cvec("h2_lower_body_joint_isaaclab_order_in_isaaclab_index", [p[0] for p in lower_il_pairs])}
{cvec("h2_lower_body_joint_isaaclab_order_in_mujoco_index", [p[1] for p in lower_il_pairs])}
{cvec("h2_upper_body_joint_mujoco_order_in_isaaclab_index", in_isaaclab(upper_mj))}
{cvec("h2_upper_body_joint_mujoco_order_in_mujoco_index", upper_mj)}
{cvec("h2_upper_body_joint_isaaclab_order_in_isaaclab_index", [p[0] for p in upper_il_pairs])}
{cvec("h2_upper_body_joint_isaaclab_order_in_mujoco_index", [p[1] for p in upper_il_pairs])}
{cvec("h2_wrist_joint_mujoco_order_in_isaaclab_index", in_isaaclab(wrist_mj))}
{cvec("h2_wrist_joint_mujoco_order_in_mujoco_index", wrist_mj)}
{cvec("h2_wrist_joint_isaaclab_order_in_isaaclab_index", [p[0] for p in wrist_il_pairs])}
{cvec("h2_wrist_joint_isaaclab_order_in_mujoco_index", [p[1] for p in wrist_il_pairs])}
// VR tracking body indices (IsaacLab body order, pelvis = 0)
const std::array<int, 3> h2_vr_3point_index = {{{", ".join(map(str, vr3))}}}; // l_wrist, r_wrist, torso
const std::array<int, 5> h2_vr_5point_index = {{{", ".join(map(str, vr5))}}}; // l_wrist, r_wrist, pelvis, l_ankle, r_ankle

// Action scale: 0.25 * training_effort_limit / stiffness  (mujoco order)
{annotated("double", "h2_action_scale", scale_expr)}
// PD gains (mujoco order) -- must match the sim yaml and training config
{annotated("float", "h2_kps", kp_expr)}
{annotated("float", "h2_kds", kd_expr)}
// Default standing pose (mujoco order), from H2_CFG.init_state
{annotated("double", "h2_default_angles", defaults)}
#endif // H2_POLICY_PARAMETERS_HPP
"""
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
