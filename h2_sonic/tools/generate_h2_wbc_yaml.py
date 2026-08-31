#!/usr/bin/env python3
"""Generate the H2 WBC yaml for the GR00T-WholeBodyControl MuJoCo sim.

Derives everything from primary sources instead of hand-typed numbers:
  - joint order, position limits, and effort limits are read from the H2 MJCF
    (gear_sonic/data/assets/robot_description/mjcf/h2.xml) via mujoco
  - PD gains, armature-derived stiffness/damping, and velocity limits mirror
    the Isaac Lab training config (gear_sonic/envs/manager_env/robots/h2.py)
  - default joint angles mirror H2_CFG.init_state in the same file

Usage:
    python tools/generate_h2_wbc_yaml.py --upstream <path-to-GR00T-WholeBodyControl> \
        [--out overlay/gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml]

Requires: mujoco (and the H2 meshes present, see setup.sh).
"""

import argparse
import math
import os
import re

import mujoco

# --- constants copied from gear_sonic/envs/manager_env/robots/h2.py ---------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0


def kp(armature):
    return armature * NATURAL_FREQ**2


def kd(armature):
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


# joint-name pattern -> (armature multiplier, armature) per the actuator groups
# in robots/h2.py ("legs", "feet", "waist", "waist_yaw", "head", "arms")
GAIN_TABLE = [
    (r".*_hip_pitch_joint", 1.0, ARMATURE_7520_22),
    (r".*_hip_roll_joint", 1.0, ARMATURE_7520_22),
    (r".*_hip_yaw_joint", 1.0, ARMATURE_7520_14),
    (r".*_knee_joint", 1.0, ARMATURE_7520_22),
    (r".*_ankle_(pitch|roll)_joint", 2.0, ARMATURE_5020),
    (r"waist_(roll|pitch)_joint", 2.0, ARMATURE_5020),
    (r"waist_yaw_joint", 1.0, ARMATURE_7520_14),
    (r"head_(pitch|yaw)_joint", 2.0, ARMATURE_5020),
    (r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", 1.0, ARMATURE_5020),
    (r".*_wrist_(pitch|yaw)_joint", 1.0, ARMATURE_4010),
]

# velocity_limit_sim per robots/h2.py
VEL_TABLE = [
    (r".*_hip_yaw_joint", 32.0),
    (r".*_hip_(roll|pitch)_joint", 20.0),
    (r".*_knee_joint", 20.0),
    (r".*_ankle_(pitch|roll)_joint", 37.0),
    (r"waist_yaw_joint", 32.0),
    (r"waist_(roll|pitch)_joint", 37.0),
    (r"head_(pitch|yaw)_joint", 37.0),
    (r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", 37.0),
    (r".*_wrist_(pitch|yaw)_joint", 22.0),
]

# H2_CFG.init_state.joint_pos per robots/h2.py
DEFAULT_ANGLES = [
    (r".*_hip_pitch_joint", -0.312),
    (r".*_knee_joint", 0.669),
    (r".*_ankle_pitch_joint", -0.363),
    (r".*_elbow_joint", 0.6),
    (r"left_shoulder_roll_joint", 0.2),
    (r"left_shoulder_pitch_joint", 0.2),
    (r"right_shoulder_roll_joint", -0.2),
    (r"right_shoulder_pitch_joint", 0.2),
]


def lookup(table, name):
    for pattern, *vals in table:
        if re.fullmatch(pattern, name):
            return vals
    raise KeyError(f"no table entry matches joint {name!r}")


def fmt_list(vals, per_line=6, indent=4, fmt="{:g}"):
    lines = []
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(fmt.format(v) for v in vals[i : i + per_line])
        lines.append(" " * indent + chunk)
    return "[\n" + ",\n".join(lines) + "\n]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="path to GR00T-WholeBodyControl checkout")
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "overlay/gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml",
        ),
    )
    args = ap.parse_args()

    mjcf = os.path.join(args.upstream, "gear_sonic/data/assets/robot_description/mjcf/h2.xml")
    model = mujoco.MjModel.from_xml_path(mjcf)

    joints = []
    for i in range(model.njnt):
        j = model.joint(i)
        if j.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joints.append(
            {
                "name": j.name,
                "lo": float(j.range[0]),
                "hi": float(j.range[1]),
            }
        )
    # actuator order must equal joint order for the identity MOTOR2JOINT map
    act_joint_names = [
        model.joint(model.actuator(i).trnid[0]).name for i in range(model.nu)
    ]
    assert act_joint_names == [j["name"] for j in joints], (
        "actuator order differs from joint order; MOTOR2JOINT would not be identity"
    )
    # effort limits from the actuators' actuatorfrcrange/forcerange
    for i in range(model.nu):
        fr = model.actuator(i).forcerange
        jid = model.actuator(i).trnid[0]
        afr = model.jnt_actfrcrange[jid]
        limit = fr[1] if fr[1] > 0 else afr[1]
        joints[i]["effort"] = float(limit)

    n = len(joints)
    assert n == 31, f"expected 31 actuated joints, got {n}"

    names = [j["name"] for j in joints]
    kp_list, kd_list, vel_list, default_list = [], [], [], []
    for name in names:
        mult, armature = lookup(GAIN_TABLE, name)
        kp_list.append(round(mult * kp(armature), 4))
        kd_list.append(round(mult * kd(armature), 4))
        vel_list.append(lookup(VEL_TABLE, name)[0])
        angle = 0.0
        for pattern, val in DEFAULT_ANGLES:
            if re.fullmatch(pattern, name):
                angle = val
                break
        default_list.append(angle)

    lo_list = [round(j["lo"], 6) for j in joints]
    hi_list = [round(j["hi"], 6) for j in joints]
    effort_list = [j["effort"] for j in joints]
    identity = list(range(n))

    weak_motor = "\n".join(f"  {name}: {i}" for i, name in enumerate(names))
    default_annotated = "[\n" + "\n".join(
        f"    {v:g},  # {name}" if i < n - 1 else f"    {v:g}  # {name}"
        for i, (v, name) in enumerate(zip(default_list, names))
    ) + "\n]"

    upper_names = names[12:]  # waist(3) + head(2) + arms(14)
    assert len(upper_names) == 19
    loco_upper = [default_list[12 + i] for i in range(19)]

    out = f"""# H2 (31 DOF) WBC config for the GR00T-WholeBodyControl MuJoCo sim loop.
# GENERATED by tools/generate_h2_wbc_yaml.py -- edit the generator, not this file.
#
# Joint order is the H2 MJCF kinematic-tree order (identical to actuator order):
#   legs: hip pitch/roll/yaw, knee, ankle ROLL, ankle PITCH (note: roll before
#   pitch, opposite of G1), then waist yaw/roll/pitch, head pitch/yaw, and
#   7-DOF arms (shoulder p/r/y, elbow, wrist r/p/y), left before right.
# PD gains and velocity limits mirror the Isaac Lab training config
# (gear_sonic/envs/manager_env/robots/h2.py); position and effort limits come
# from the MJCF (gear_sonic/data/assets/robot_description/mjcf/h2.xml).

ROBOT_TYPE: 'h2_31dof'
ROBOT_SCENE: "gear_sonic/data/assets/robot_description/mjcf/scene_h2.xml"

DOMAIN_ID: 0
INTERFACE: "lo"
SIMULATOR: "mujoco"

USE_JOYSTICK: 0
JOYSTICK_TYPE: "xbox"
JOYSTICK_DEVICE: 0

FREE_BASE: False

PRINT_SCENE_INFORMATION: True
# Virtual gantry (press 9/8/7 in the viewer). Keep on: without a trained H2
# policy the robot is just a PD hold at DEFAULT_DOF_ANGLES and will not balance.
ENABLE_ELASTIC_BAND: True

SIMULATE_DT: 0.005
VIEWER_DT: 0.02
REWARD_DT: 0.02
USE_SENSOR: False
USE_HISTORY: True
USE_HISTORY_LOCO: True
USE_HISTORY_MIMIC: True

GAIT_PERIOD: 0.9

MOTOR2JOINT: {identity}

JOINT2MOTOR: {identity}

UNITREE_LEGGED_CONST:
  HIGHLEVEL: 0xEE
  LOWLEVEL: 0xFF
  TRIGERLEVEL: 0xF0
  PosStopF: 2146000000.0
  VelStopF: 16000.0
  MODE_MACHINE: 5
  MODE_PR:  0

JOINT_KP: {fmt_list(kp_list)}

JOINT_KD: {fmt_list(kd_list)}

MOTOR_KP: {fmt_list(kp_list)}

MOTOR_KD: {fmt_list(kd_list)}

WeakMotorJointIndex:
{weak_motor}

NUM_MOTORS: {n}
NUM_JOINTS: {n}
NUM_HAND_MOTORS: 0
NUM_HAND_JOINTS: 0
NUM_UPPER_BODY_JOINTS: 19

DEFAULT_DOF_ANGLES: {default_annotated}

DEFAULT_MOTOR_ANGLES: {default_annotated}

motor_pos_lower_limit_list: {fmt_list(lo_list)}
motor_pos_upper_limit_list: {fmt_list(hi_list)}
motor_vel_limit_list: {fmt_list(vel_list)}
motor_effort_limit_list: {fmt_list(effort_list)}

history_config: {{
                  base_ang_vel: 4,
                  projected_gravity: 4,
                  command_lin_vel: 4,
                  command_ang_vel: 4,
                  command_base_height: 4,
                  command_stand: 4,
                  ref_upper_dof_pos: 4,
                  dof_pos: 4,
                  dof_vel: 4,
                  actions: 4,
                  ref_motion_phase: 4,
                  sin_phase: 4,
                  cos_phase: 4
                }}
history_loco_config: {{
                  base_ang_vel: 4,
                  projected_gravity: 4,
                  command_lin_vel: 4,
                  command_ang_vel: 4,
                  command_stand: 4,
                  ref_upper_dof_pos: 4,
                  dof_pos: 4,
                  dof_vel: 4,
                  actions: 4,
                  sin_phase: 4,
                  cos_phase: 4
                }}
history_loco_height_config: {{
                  base_ang_vel: 4,
                  projected_gravity: 4,
                  command_lin_vel: 4,
                  command_ang_vel: 4,
                  command_base_height: 4,
                  command_stand: 4,
                  ref_upper_dof_pos: 4,
                  dof_pos: 4,
                  dof_vel: 4,
                  actions: 4,
                  sin_phase: 4,
                  cos_phase: 4
                }}
history_mimic_config: {{
                  base_ang_vel: 4,
                  projected_gravity: 4,
                  dof_pos: 4,
                  dof_vel: 4,
                  actions: 4,
                  ref_motion_phase: 4,
                }}
obs_dims: {{
            base_lin_vel: 3,
            base_ang_vel: 3,
            projected_gravity: 3,
            command_lin_vel: 2,
            command_ang_vel: 1,
            command_stand: 1,
            command_base_height: 1,
            ref_upper_dof_pos: 19, # upper body actions (waist 3 + head 2 + arms 14)
            dof_pos: {n},
            dof_vel: {n},
            actions: {n}, # full body actions
            phase_time: 1,
            ref_motion_phase: 1,
            sin_phase: 1,
            cos_phase: 1,
          }}
obs_loco_dims: {{
            base_lin_vel: 3,
            base_ang_vel: 3,
            projected_gravity: 3,
            command_lin_vel: 2,
            command_ang_vel: 1,
            command_stand: 1,
            command_base_height: 1,
            ref_upper_dof_pos: 19, # upper body actions
            dof_pos: {n},
            dof_vel: {n},
            actions: 12, # lower body actions
            phase_time: 1,
            sin_phase: 1,
            cos_phase: 1,
          }}
obs_mimic_dims: {{
            base_lin_vel: 3,
            base_ang_vel: 3,
            projected_gravity: 3,
            dof_pos: {n},
            dof_vel: {n},
            actions: {n},
            ref_motion_phase: 1,
          }}
obs_scales: {{
    base_lin_vel: 2.0,
    base_ang_vel: 0.25,
    projected_gravity: 1.0,
    command_lin_vel: 1,
    command_ang_vel: 1,
    command_stand: 1,
    command_base_height: 2,
    ref_upper_dof_pos: 1.0,
    dof_pos: 1.0,
    dof_vel: 0.05,
    history: 1.0,
    history_loco: 1.0,
    history_mimic: 1.0,
    actions: 1.0,
    phase_time: 1.0,
    ref_motion_phase: 1.0,
    sin_phase: 1.0,
    cos_phase: 1.0
  }}

# Upper-body hold pose during locomotion (waist 3, head 2, left arm 7,
# right arm 7) -- the H2 default pose from robots/h2.py.
loco_upper_body_dof_pos: {fmt_list(loco_upper, per_line=7, indent=2)}

robot_dofs: {{
  "h2_31dof": {[1] * n},
}}

# No released H2 mimic checkpoints yet.
mimic_robot_types: {{}}
mimic_models: {{}}
start_upper_body_dof_pos: {{}}
motion_length_s: {{}}
"""

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"wrote {out_path} ({n} joints)")


if __name__ == "__main__":
    main()
