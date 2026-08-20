"""H2 constants for motion-library conversion.

Mirrors the hardcoded G1 constants in convert_soma_csv_to_motion_lib.py, but
derived programmatically from mjcf/h2.xml rather than transcribed by hand.

Verified against the MJCF:
  * nbody - 1 == 32, nu == 31
  * actuator order == joint order (excluding the floating base)
  * jnt_bodyid[j] == j + 1, i.e. joint j moves body j once the world body is
    dropped -- so pose_aa[:, 1:NUM_BODIES] = DOF_AXIS * dof lines up exactly.
"""

import numpy as np

NUM_BODIES = 32  # pelvis + 31 actuated links
NUM_DOF = 31

# Row k is the rotation axis of the joint driving body k+1 (world body excluded).
H2_DOF_AXIS = np.array(
    [
        [0, 1, 0],  # left_hip_pitch_joint
        [1, 0, 0],  # left_hip_roll_joint
        [0, 0, 1],  # left_hip_yaw_joint
        [0, 1, 0],  # left_knee_joint
        [1, 0, 0],  # left_ankle_roll_joint
        [0, 1, 0],  # left_ankle_pitch_joint
        [0, 1, 0],  # right_hip_pitch_joint
        [1, 0, 0],  # right_hip_roll_joint
        [0, 0, 1],  # right_hip_yaw_joint
        [0, 1, 0],  # right_knee_joint
        [1, 0, 0],  # right_ankle_roll_joint
        [0, 1, 0],  # right_ankle_pitch_joint
        [0, 0, 1],  # waist_yaw_joint
        [1, 0, 0],  # waist_roll_joint
        [0, 1, 0],  # waist_pitch_joint
        [0, 1, 0],  # head_pitch_joint
        [0, 0, 1],  # head_yaw_joint
        [0, 1, 0],  # left_shoulder_pitch_joint
        [1, 0, 0],  # left_shoulder_roll_joint
        [0, 0, 1],  # left_shoulder_yaw_joint
        [0, 1, 0],  # left_elbow_joint
        [1, 0, 0],  # left_wrist_roll_joint
        [0, 1, 0],  # left_wrist_pitch_joint
        [0, 0, 1],  # left_wrist_yaw_joint
        [0, 1, 0],  # right_shoulder_pitch_joint
        [1, 0, 0],  # right_shoulder_roll_joint
        [0, 0, 1],  # right_shoulder_yaw_joint
        [0, 1, 0],  # right_elbow_joint
        [1, 0, 0],  # right_wrist_roll_joint
        [0, 1, 0],  # right_wrist_pitch_joint
        [0, 0, 1],  # right_wrist_yaw_joint
    ],
    dtype=np.float32,
)

# MuJoCo/MJCF actuator order -- the order retargeted CSV DOF columns must follow.
# Note H2 orders each ankle roll-before-pitch, unlike G1's pitch-before-roll.
H2_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_roll_joint_dof",
    "left_ankle_pitch_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_roll_joint_dof",
    "right_ankle_pitch_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "head_pitch_joint_dof",
    "head_yaw_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]

assert H2_DOF_AXIS.shape == (NUM_DOF, 3)
assert len(H2_CSV_JOINT_NAMES) == NUM_DOF
