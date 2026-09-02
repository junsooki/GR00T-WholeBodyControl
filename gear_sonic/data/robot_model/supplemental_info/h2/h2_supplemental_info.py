"""H2-specific supplemental info: actuated joints, limits, and default poses by waist/elbow config."""

from dataclasses import dataclass

import numpy as np

# WaistLocation and ElbowPose describe joint-group layout and arm posture, not G1
# geometry, so H2 reuses them rather than defining a parallel pair of enums.
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    WaistLocation,
)
from gear_sonic.data.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


@dataclass
class H2SupplementalInfo(RobotSupplementalInfo):
    """
    Supplemental information for the H2 robot.

    Differences from G1 that this class exists to capture:

    - 31 actuated joints, not 29: H2 adds a 2-DOF head (head_pitch, head_yaw).
    - The ankle chain is knee -> ankle_roll -> ankle_pitch, the reverse of G1.
    - The hands are passive rubber shells, so there are no hand joints at all.

    Args:
        waist_location: Where to place waist joints in the joint groups
        elbow_pose: Which elbow pose configuration to use for default joint positions
    """

    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
    ):
        name = "H2"

        # Ordered as in h2.xml's actuator block, which is also the hardware
        # motor order the deployment side indexes with.
        body_actuated_joints = [
            # Left leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_roll_joint",
            "left_ankle_pitch_joint",
            # Right leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_roll_joint",
            "right_ankle_pitch_joint",
            # Waist
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Head
            "head_pitch_joint",
            "head_yaw_joint",
            # Left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # H2 ships with passive rubber hands: the palm joints in h2.urdf are fixed.
        left_hand_actuated_joints = []
        right_hand_actuated_joints = []

        # Joint limits from h2.urdf (h2.xml agrees on every joint).
        joint_limits = {
            # Left leg
            "left_hip_pitch_joint": [-2.4526, 2.77542],
            "left_hip_roll_joint": [-0.467441, 2.16886],
            "left_hip_yaw_joint": [-2.827, 2.827],
            "left_knee_joint": [-0.08725, 2.53025],
            "left_ankle_roll_joint": [-0.349066, 0.296706],
            "left_ankle_pitch_joint": [-1.13446, 0.610865],
            # Right leg
            "right_hip_pitch_joint": [-2.4526, 2.77542],
            "right_hip_roll_joint": [-2.16886, 0.467441],
            "right_hip_yaw_joint": [-2.827, 2.827],
            "right_knee_joint": [-0.08725, 2.53025],
            "right_ankle_roll_joint": [-0.296706, 0.349066],
            "right_ankle_pitch_joint": [-1.13446, 0.610865],
            # Waist
            "waist_yaw_joint": [-1.7453, 1.7453],
            "waist_roll_joint": [-0.5236, 0.5236],
            "waist_pitch_joint": [-0.43633, 0.5236],
            # Head
            "head_pitch_joint": [-0.5236, 0.83775],
            "head_yaw_joint": [-1.7453, 1.7453],
            # Left arm
            "left_shoulder_pitch_joint": [-2.61799, 1.8326],
            "left_shoulder_roll_joint": [-0.516617, 2.63545],
            "left_shoulder_yaw_joint": [-2.61799, 2.61799],
            "left_elbow_joint": [-0.986111, 3.07178],
            "left_wrist_roll_joint": [-2.61799, 2.61799],
            "left_wrist_pitch_joint": [-0.436332, 0.436332],
            "left_wrist_yaw_joint": [-1.22173, 1.22173],
            # Right arm
            "right_shoulder_pitch_joint": [-2.61799, 1.8326],
            "right_shoulder_roll_joint": [-2.63545, 0.516617],
            "right_shoulder_yaw_joint": [-2.61799, 2.61799],
            "right_elbow_joint": [-0.986111, 3.07178],
            "right_wrist_roll_joint": [-2.61799, 2.61799],
            "right_wrist_pitch_joint": [-0.436332, 0.436332],
            "right_wrist_yaw_joint": [-1.22173, 1.22173],
        }

        # Define joint groups
        joint_groups = {
            # Body groups
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            "head": {
                "joints": ["head_pitch_joint", "head_yaw_joint"],
                "groups": [],
            },
            # Leg groups
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_roll_joint",
                    "left_ankle_pitch_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_roll_joint",
                    "right_ankle_pitch_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            # Arm groups
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Hand groups are kept so that callers written against G1 resolve,
            # but H2's hands carry no actuated joints.
            "left_hand": {"joints": [], "groups": []},
            "right_hand": {"joints": [], "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            # Full body groups
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms", "head"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        # Define joint name mapping from generic types to robot-specific names
        joint_name_mapping = {
            # Waist joints
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            # Head joints (no G1 equivalent)
            "head_pitch": "head_pitch_joint",
            "head_yaw": "head_yaw_joint",
            # Shoulder joints
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            # Elbow joints
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            # Wrist joints
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
        }

        root_frame_name = "pelvis"

        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

        calibration_joint_q = {"elbow_pitch": {"left": 0.0, "right": 0.0}}

        # 90° Y-axis rotation: aligns hand-tracking frame (palm-forward) to robot wrist frame.
        # Carried over from G1 -- H2's wrist chain is roll/pitch/yaw in the same
        # order and sense, but this has not been checked against H2 hardware.
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # HIGH: arms raised with elbows bent (hands near shoulder height)
        # LOW: H2's own standing pose, so the IK default agrees with the
        # default joint angles the policy is trained around (H2_CFG.init_state).
        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
            }
        else:  # ElbowPose.LOW
            default_joint_q = {
                "shoulder_pitch": {"left": 0.2, "right": 0.2},
                "shoulder_roll": {"left": 0.2, "right": -0.2},
                "elbow_pitch": {"left": 0.6, "right": 0.6},
            }

        teleop_upper_body_motion_scale = 1.0

        # Configure joint groups based on waist location
        modified_joint_groups = joint_groups.copy()
        if waist_location == WaistLocation.UPPER_BODY:
            # Move waist from lower_body to upper_body_no_hands
            modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "head", "waist"],
            }
        elif waist_location == WaistLocation.LOWER_AND_UPPER_BODY:
            # Add waist to upper_body_no_hands while keeping it in lower_body
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "head", "waist"],
            }
        # For LOWER_BODY, keep default joint_groups as is

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=modified_joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            calibration_joint_q=calibration_joint_q,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )
