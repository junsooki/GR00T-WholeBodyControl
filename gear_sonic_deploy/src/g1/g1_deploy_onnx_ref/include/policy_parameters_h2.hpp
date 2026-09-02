/**
 * @file policy_parameters_h2.hpp
 * @brief Motor constants, PID gains, joint mappings, action scales, and default
 *        standing angles for the Unitree H2 31-DOF policy.
 *
 * H2 counterpart of policy_parameters.hpp. Same conventions throughout: every
 * array below is in **MuJoCo order** and the two permutation arrays convert to
 * and from IsaacLab order.
 *
 * Derived from gear_sonic/envs/manager_env/robots/h2.py (the config the shipped
 * checkpoint was trained with) and cross-checked against
 * gear_sonic/data/assets/robot_description/mjcf/h2.xml -- the orderings here are
 * a breadth-first walk of that MJCF's body tree, which is what IsaacLab uses.
 *
 * ## Differences from G1 that are easy to get wrong
 *
 *  - **31 DOF / 32 bodies**, not 29 / 30.
 *  - **Ankle order is reversed.** H2's leg chain is knee -> ankle_roll ->
 *    ankle_pitch; G1's is knee -> ankle_pitch -> ankle_roll. So H2's distal
 *    body (the one carrying the foot) is *_ankle_pitch_link, and MuJoCo joint
 *    index 4 is ankle_roll where G1's is ankle_pitch.
 *  - **Effort limits are per joint, not per motor class.** G1 can use four
 *    EFFORT_LIMIT_* constants because each motor class has one torque rating.
 *    H2's URDF gives ankle_roll 19 Nm against ankle_pitch's 66.88 Nm despite
 *    both being 5020-class, so a per-class constant would be wrong.
 *  - **H2 has a 2-DOF head** (head_pitch, head_yaw); G1 has none.
 *
 * ## PID Gain Computation
 *
 * Identical to G1: stiffness = armature x w^2, damping = 2 x zeta x armature x w,
 * with w = 10 Hz x 2pi and zeta = 2.0. Armature values are G1's -- see the note
 * in h2.py: no published source gives H2's rotor inertia, so these are carried
 * over rather than invented.
 *
 * ## Action Scaling
 *
 * action_scale = 0.25 x effort_limit / stiffness, target = action x action_scale
 * + default_angle. Same as G1.
 */

#ifndef POLICY_PARAMETERS_H2_HPP
#define POLICY_PARAMETERS_H2_HPP

#include <array>
#include <vector>

const double ONE_DEGREE_H2 = 0.0174533;

// Motor armature constants. Carried over from G1 -- h2.urdf carries link
// inertias only and Unitree publish torque but not rotor inertia.
const double H2_ARMATURE_5020 = 0.003609725;
const double H2_ARMATURE_7520_14 = 0.010177520;
const double H2_ARMATURE_7520_22 = 0.025101925;
const double H2_ARMATURE_4010 = 0.00425;

const double H2_NATURAL_FREQ = 10 * 2.0 * 3.1415926535; // 10Hz
const double H2_DAMPING_RATIO = 2;

const double H2_STIFFNESS_5020 = H2_ARMATURE_5020 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_7520_14 = H2_ARMATURE_7520_14 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_7520_22 = H2_ARMATURE_7520_22 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_4010 = H2_ARMATURE_4010 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;

const double H2_DAMPING_5020 = 2.0 * H2_DAMPING_RATIO * H2_ARMATURE_5020 * H2_NATURAL_FREQ;
const double H2_DAMPING_7520_14 = 2.0 * H2_DAMPING_RATIO * H2_ARMATURE_7520_14 * H2_NATURAL_FREQ;
const double H2_DAMPING_7520_22 = 2.0 * H2_DAMPING_RATIO * H2_ARMATURE_7520_22 * H2_NATURAL_FREQ;
const double H2_DAMPING_4010 = 2.0 * H2_DAMPING_RATIO * H2_ARMATURE_4010 * H2_NATURAL_FREQ;

// Per-joint effort limits (Nm), MuJoCo order, from h2.urdf. See the header note:
// these are per joint because H2's 5020-class joints do not share a rating.
const std::array<double, 31> h2_effort_limit = {
    360.0,  // 0  left_hip_pitch
    360.0,  // 1  left_hip_roll
    360.0,  // 2  left_hip_yaw
    360.0,  // 3  left_knee
    19.0,   // 4  left_ankle_roll   <-- roll before pitch on H2
    66.88,  // 5  left_ankle_pitch
    360.0,  // 6  right_hip_pitch
    360.0,  // 7  right_hip_roll
    360.0,  // 8  right_hip_yaw
    360.0,  // 9  right_knee
    19.0,   // 10 right_ankle_roll
    66.88,  // 11 right_ankle_pitch
    120.0,  // 12 waist_yaw
    180.0,  // 13 waist_roll
    180.0,  // 14 waist_pitch
    50.0,   // 15 head_pitch
    50.0,   // 16 head_yaw
    120.0,  // 17 left_shoulder_pitch
    54.0,   // 18 left_shoulder_roll
    54.0,   // 19 left_shoulder_yaw
    54.0,   // 20 left_elbow
    54.0,   // 21 left_wrist_roll
    25.0,   // 22 left_wrist_pitch
    25.0,   // 23 left_wrist_yaw
    120.0,  // 24 right_shoulder_pitch
    54.0,   // 25 right_shoulder_roll
    54.0,   // 26 right_shoulder_yaw
    54.0,   // 27 right_elbow
    54.0,   // 28 right_wrist_roll
    25.0,   // 29 right_wrist_pitch
    25.0,   // 30 right_wrist_yaw
};

// PID position gains (Kp), MuJoCo order.
const std::array<float, 31> h2_kps = {
    H2_STIFFNESS_7520_22,       // 0  left_hip_pitch
    H2_STIFFNESS_7520_22,       // 1  left_hip_roll
    H2_STIFFNESS_7520_14,       // 2  left_hip_yaw
    H2_STIFFNESS_7520_22,       // 3  left_knee
    2.0f * H2_STIFFNESS_5020,   // 4  left_ankle_roll
    2.0f * H2_STIFFNESS_5020,   // 5  left_ankle_pitch
    H2_STIFFNESS_7520_22,       // 6  right_hip_pitch
    H2_STIFFNESS_7520_22,       // 7  right_hip_roll
    H2_STIFFNESS_7520_14,       // 8  right_hip_yaw
    H2_STIFFNESS_7520_22,       // 9  right_knee
    2.0f * H2_STIFFNESS_5020,   // 10 right_ankle_roll
    2.0f * H2_STIFFNESS_5020,   // 11 right_ankle_pitch
    H2_STIFFNESS_7520_14,       // 12 waist_yaw
    2.0f * H2_STIFFNESS_5020,   // 13 waist_roll
    2.0f * H2_STIFFNESS_5020,   // 14 waist_pitch
    2.0f * H2_STIFFNESS_5020,   // 15 head_pitch
    2.0f * H2_STIFFNESS_5020,   // 16 head_yaw
    H2_STIFFNESS_5020,          // 17 left_shoulder_pitch
    H2_STIFFNESS_5020,          // 18 left_shoulder_roll
    H2_STIFFNESS_5020,          // 19 left_shoulder_yaw
    H2_STIFFNESS_5020,          // 20 left_elbow
    H2_STIFFNESS_5020,          // 21 left_wrist_roll
    H2_STIFFNESS_4010,          // 22 left_wrist_pitch
    H2_STIFFNESS_4010,          // 23 left_wrist_yaw
    H2_STIFFNESS_5020,          // 24 right_shoulder_pitch
    H2_STIFFNESS_5020,          // 25 right_shoulder_roll
    H2_STIFFNESS_5020,          // 26 right_shoulder_yaw
    H2_STIFFNESS_5020,          // 27 right_elbow
    H2_STIFFNESS_5020,          // 28 right_wrist_roll
    H2_STIFFNESS_4010,          // 29 right_wrist_pitch
    H2_STIFFNESS_4010,          // 30 right_wrist_yaw
};

// PID derivative gains (Kd), MuJoCo order. Same class per index as h2_kps.
const std::array<float, 31> h2_kds = {
    H2_DAMPING_7520_22,       // 0  left_hip_pitch
    H2_DAMPING_7520_22,       // 1  left_hip_roll
    H2_DAMPING_7520_14,       // 2  left_hip_yaw
    H2_DAMPING_7520_22,       // 3  left_knee
    2.0f * H2_DAMPING_5020,   // 4  left_ankle_roll
    2.0f * H2_DAMPING_5020,   // 5  left_ankle_pitch
    H2_DAMPING_7520_22,       // 6  right_hip_pitch
    H2_DAMPING_7520_22,       // 7  right_hip_roll
    H2_DAMPING_7520_14,       // 8  right_hip_yaw
    H2_DAMPING_7520_22,       // 9  right_knee
    2.0f * H2_DAMPING_5020,   // 10 right_ankle_roll
    2.0f * H2_DAMPING_5020,   // 11 right_ankle_pitch
    H2_DAMPING_7520_14,       // 12 waist_yaw
    2.0f * H2_DAMPING_5020,   // 13 waist_roll
    2.0f * H2_DAMPING_5020,   // 14 waist_pitch
    2.0f * H2_DAMPING_5020,   // 15 head_pitch
    2.0f * H2_DAMPING_5020,   // 16 head_yaw
    H2_DAMPING_5020,          // 17 left_shoulder_pitch
    H2_DAMPING_5020,          // 18 left_shoulder_roll
    H2_DAMPING_5020,          // 19 left_shoulder_yaw
    H2_DAMPING_5020,          // 20 left_elbow
    H2_DAMPING_5020,          // 21 left_wrist_roll
    H2_DAMPING_4010,          // 22 left_wrist_pitch
    H2_DAMPING_4010,          // 23 left_wrist_yaw
    H2_DAMPING_5020,          // 24 right_shoulder_pitch
    H2_DAMPING_5020,          // 25 right_shoulder_roll
    H2_DAMPING_5020,          // 26 right_shoulder_yaw
    H2_DAMPING_5020,          // 27 right_elbow
    H2_DAMPING_5020,          // 28 right_wrist_roll
    H2_DAMPING_4010,          // 29 right_wrist_pitch
    H2_DAMPING_4010,          // 30 right_wrist_yaw
};

// action_scale = 0.25 * effort_limit / stiffness, MuJoCo order.
const std::array<double, 31> h2_action_scale = {
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 0  left_hip_pitch
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 1  left_hip_roll
    0.25 * 360.0 / H2_STIFFNESS_7520_14,        // 2  left_hip_yaw
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 3  left_knee
    0.25 * 19.0 / (2.0 * H2_STIFFNESS_5020),    // 4  left_ankle_roll
    0.25 * 66.88 / (2.0 * H2_STIFFNESS_5020),   // 5  left_ankle_pitch
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 6  right_hip_pitch
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 7  right_hip_roll
    0.25 * 360.0 / H2_STIFFNESS_7520_14,        // 8  right_hip_yaw
    0.25 * 360.0 / H2_STIFFNESS_7520_22,        // 9  right_knee
    0.25 * 19.0 / (2.0 * H2_STIFFNESS_5020),    // 10 right_ankle_roll
    0.25 * 66.88 / (2.0 * H2_STIFFNESS_5020),   // 11 right_ankle_pitch
    0.25 * 120.0 / H2_STIFFNESS_7520_14,        // 12 waist_yaw
    0.25 * 180.0 / (2.0 * H2_STIFFNESS_5020),   // 13 waist_roll
    0.25 * 180.0 / (2.0 * H2_STIFFNESS_5020),   // 14 waist_pitch
    0.25 * 50.0 / (2.0 * H2_STIFFNESS_5020),    // 15 head_pitch
    0.25 * 50.0 / (2.0 * H2_STIFFNESS_5020),    // 16 head_yaw
    0.25 * 120.0 / H2_STIFFNESS_5020,           // 17 left_shoulder_pitch
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 18 left_shoulder_roll
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 19 left_shoulder_yaw
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 20 left_elbow
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 21 left_wrist_roll
    0.25 * 25.0 / H2_STIFFNESS_4010,            // 22 left_wrist_pitch
    0.25 * 25.0 / H2_STIFFNESS_4010,            // 23 left_wrist_yaw
    0.25 * 120.0 / H2_STIFFNESS_5020,           // 24 right_shoulder_pitch
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 25 right_shoulder_roll
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 26 right_shoulder_yaw
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 27 right_elbow
    0.25 * 54.0 / H2_STIFFNESS_5020,            // 28 right_wrist_roll
    0.25 * 25.0 / H2_STIFFNESS_4010,            // 29 right_wrist_pitch
    0.25 * 25.0 / H2_STIFFNESS_4010,            // 30 right_wrist_yaw
};

// Default standing angles (rad), MuJoCo order, from H2_CFG.init_state.joint_pos.
const std::array<double, 31> h2_default_angles = {
    -0.312, // 0  left_hip_pitch
     0.0,   // 1  left_hip_roll
     0.0,   // 2  left_hip_yaw
     0.669, // 3  left_knee
     0.0,   // 4  left_ankle_roll
    -0.363, // 5  left_ankle_pitch
    -0.312, // 6  right_hip_pitch
     0.0,   // 7  right_hip_roll
     0.0,   // 8  right_hip_yaw
     0.669, // 9  right_knee
     0.0,   // 10 right_ankle_roll
    -0.363, // 11 right_ankle_pitch
     0.0,   // 12 waist_yaw
     0.0,   // 13 waist_roll
     0.0,   // 14 waist_pitch
     0.0,   // 15 head_pitch
     0.0,   // 16 head_yaw
     0.2,   // 17 left_shoulder_pitch
     0.2,   // 18 left_shoulder_roll
     0.0,   // 19 left_shoulder_yaw
     0.6,   // 20 left_elbow
     0.0,   // 21 left_wrist_roll
     0.0,   // 22 left_wrist_pitch
     0.0,   // 23 left_wrist_yaw
     0.2,   // 24 right_shoulder_pitch
    -0.2,   // 25 right_shoulder_roll
     0.0,   // 26 right_shoulder_yaw
     0.6,   // 27 right_elbow
     0.0,   // 28 right_wrist_roll
     0.0,   // 29 right_wrist_pitch
     0.0,   // 30 right_wrist_yaw
};

// --------------------------------------------------------------------------
// Joint / body index maps. Every array below was generated from a breadth-first
// walk of h2.xml and asserted equal to the corresponding array in h2.py.
// --------------------------------------------------------------------------

// mujoco order in isaaclab index
const std::array<int, 31> h2_isaaclab_to_mujoco = {
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11,
    16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30};
// isaaclab order in mujoco index
const std::array<int, 31> h2_mujoco_to_isaaclab = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10,
    16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30};

// Upper body joints (waist, head, arms) -- 19 on H2 against G1's 17: H2 adds
// head_pitch and head_yaw.
const std::vector<int> h2_upper_body_joint_mujoco_order_in_isaaclab_index = {
    2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30};
const std::vector<int> h2_upper_body_joint_mujoco_order_in_mujoco_index = {
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
const std::vector<int> h2_upper_body_joint_isaaclab_order_in_isaaclab_index = {
    2, 5, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
const std::vector<int> h2_upper_body_joint_isaaclab_order_in_mujoco_index = {
    12, 13, 14, 15, 17, 24, 16, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30};

// Wrist joints. These are the indices sonic_h2.yaml sets as
// wrist_mujoco_dof_indices for wrist pose augmentation.
const std::vector<int> h2_wrist_joint_mujoco_order_in_isaaclab_index = {25, 27, 29, 26, 28, 30};
const std::vector<int> h2_wrist_joint_mujoco_order_in_mujoco_index = {21, 22, 23, 28, 29, 30};
const std::vector<int> h2_wrist_joint_isaaclab_order_in_isaaclab_index = {25, 26, 27, 28, 29, 30};
const std::vector<int> h2_wrist_joint_isaaclab_order_in_mujoco_index = {21, 28, 22, 29, 23, 30};

// Lower body joints (12, same count as G1, but ankle roll/pitch are swapped).
const std::vector<int> h2_lower_body_joint_mujoco_order_in_isaaclab_index = {
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20};
const std::vector<int> h2_lower_body_joint_mujoco_order_in_mujoco_index = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const std::vector<int> h2_lower_body_joint_isaaclab_order_in_isaaclab_index = {
    0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 19, 20};
const std::vector<int> h2_lower_body_joint_isaaclab_order_in_mujoco_index = {
    0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

// Body (not joint) indices, IsaacLab order.
// 3-point: left wrist, right wrist, torso.
const std::array<int, 3> h2_vr_3point_index = {30, 31, 9};
// 5-point: left wrist, right wrist, pelvis, left foot, right foot. The feet are
// *_ankle_pitch_link -- *_ankle_roll_link resolves on H2 but is a mid-ankle stub.
const std::array<int, 5> h2_vr_5point_index = {30, 31, 0, 20, 21};

#endif // POLICY_PARAMETERS_H2_HPP
