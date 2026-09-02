/**
 * @file robot_parameters.hpp
 * @brief Hardware-level constants, data structures, and joint indices for the
 *        Unitree G1 humanoid robot.
 *
 * This header defines:
 *  - DDS topic names for the Unitree SDK (low-level command / state channels).
 *  - Motor count and per-joint MotorCommand structure.
 *  - HeadingState – a compact struct bundling the captured IMU quaternion
 *    with a user-controlled delta-heading offset.
 *  - OperatorState – high-level start / stop / play flags.
 *  - Control mode enum (series vs. parallel ankle actuation).
 *  - G1JointIndex enum mapping symbolic joint names to hardware motor indices.
 *
 * All indices in this file use the **hardware / URDF ordering** (not the
 * IsaacLab training ordering – see `policy_parameters.hpp` for the mapping).
 */

#ifndef ROBOT_PARAMETERS_HPP
#define ROBOT_PARAMETERS_HPP

#include <array>
#include <string>

#include "robot_common.hpp"

/// Total number of actuated joints on the G1 (29-DOF configuration).
const int G1_NUM_MOTOR = 29;

/**
 * @brief Per-joint motor command sent to the low-level controller.
 *
 * Each field is an array of size G1_NUM_MOTOR (29), indexed by hardware joint
 * index (see G1JointIndex).
 */
struct MotorCommand {
    std::array<float, G1_NUM_MOTOR> q_target = {};   ///< Target position (rad).
    std::array<float, G1_NUM_MOTOR> dq_target = {};  ///< Target velocity (rad/s).
    std::array<float, G1_NUM_MOTOR> kp = {};          ///< Position gain (Nm/rad).
    std::array<float, G1_NUM_MOTOR> kd = {};          ///< Velocity gain (Nm*s/rad).
    std::array<float, G1_NUM_MOTOR> tau_ff = {};      ///< Feed-forward torque (Nm).
};

/**
 * @brief Symbolic names for G1 hardware joint indices (0–28).
 *
 * Ankle joints have dual names because they can be addressed in either
 * series (Pitch/Roll) or parallel (A/B) mode.  Joints marked "INVALID"
 * are not present on the 23-DOF or waist-locked 29-DOF variants.
 */
enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14, // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20, // NOTE INVALID for g1 23dof
  LeftWristYaw = 21, // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27, // NOTE INVALID for g1 23dof
  RightWristYaw = 28 // NOTE INVALID for g1 23dof
};

/// Joint identifiers in hardware order, matching g1_29dof.xml's actuator names.
/// Single source of truth -- this was previously duplicated in error_monitor.hpp
/// and twice in g1_deploy_onnx_ref.cpp.
static const std::array<std::string, G1_NUM_MOTOR> G1_JOINT_NAMES = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw"};

/// Human-readable names, spoken by the TTS high-temperature warning and printed
/// in the motor temperature report. Kept separate from the identifiers above so
/// those can match the MJCF exactly without the TTS saying "leftankleroll".
static const std::array<std::string, G1_NUM_MOTOR> G1_JOINT_DISPLAY_NAMES = {
    "Left Hip Pitch", "Left Hip Roll", "Left Hip Yaw", "Left Knee",
    "Left Ankle Pitch", "Left Ankle Roll", "Right Hip Pitch", "Right Hip Roll",
    "Right Hip Yaw", "Right Knee", "Right Ankle Pitch", "Right Ankle Roll",
    "Waist Yaw", "Waist Roll", "Waist Pitch", "Left Shoulder Pitch",
    "Left Shoulder Roll", "Left Shoulder Yaw", "Left Elbow", "Left Wrist Roll",
    "Left Wrist Pitch", "Left Wrist Yaw", "Right Shoulder Pitch", "Right Shoulder Roll",
    "Right Shoulder Yaw", "Right Elbow", "Right Wrist Roll", "Right Wrist Pitch",
    "Right Wrist Yaw"};

#endif // ROBOT_PARAMETERS_HPP
