/**
 * @file robot_parameters_h2.hpp
 * @brief Hardware-level constants and joint indices for the Unitree H2 humanoid.
 *
 * H2 counterpart of robot_parameters.hpp. Same conventions: indices here are
 * **hardware / URDF ordering**, which for H2 is also the MuJoCo actuator order;
 * see policy_parameters_h2.hpp for the mapping to IsaacLab order.
 *
 * Differences from G1 that this header exists to capture:
 *
 *  - **31 motors, not 29.** H2_NUM_MOTOR is used as a compile-time array size,
 *    so this is not a value that can be switched at runtime.
 *  - **The ankle chain is reversed.** H2 is knee -> ankle_roll -> ankle_pitch,
 *    where G1 is knee -> ankle_pitch -> ankle_roll. Index 4 is *roll* on H2 and
 *    *pitch* on G1, so a G1-derived index list silently swaps the two.
 *  - **H2 has a 2-DOF head** (head_pitch, head_yaw) at indices 15-16, which G1
 *    does not have at all. Every arm index is therefore shifted by two relative
 *    to G1.
 *  - **No A/B ankle aliases.** G1 exposes its coupled ankle in either series
 *    (pitch/roll) or parallel (A/B) mode, hence its duplicate enumerators. The
 *    H2 joints are addressed directly, so there is one name per index and no
 *    Mode PR/AB distinction to carry.
 *
 * Verified against gear_sonic/data/assets/robot_description/mjcf/h2.xml: the
 * enumerator order below is that file's actuator order.
 */

#ifndef ROBOT_PARAMETERS_H2_HPP
#define ROBOT_PARAMETERS_H2_HPP

#include <array>
#include <string>

#include "robot_common.hpp"

/// Total number of actuated joints on H2.
const int H2_NUM_MOTOR = 31;

/**
 * @brief Per-joint motor command sent to the low-level controller.
 *
 * Indexed by hardware joint index (see H2JointIndex).
 */
struct H2MotorCommand {
    std::array<float, H2_NUM_MOTOR> q_target = {};   ///< Target position (rad).
    std::array<float, H2_NUM_MOTOR> dq_target = {};  ///< Target velocity (rad/s).
    std::array<float, H2_NUM_MOTOR> kp = {};         ///< Position gain (Nm/rad).
    std::array<float, H2_NUM_MOTOR> kd = {};         ///< Velocity gain (Nm*s/rad).
    std::array<float, H2_NUM_MOTOR> tau_ff = {};     ///< Feed-forward torque (Nm).
};

/**
 * @brief H2 hardware (DDS motor) joint indices, 0-30.
 *
 * Taken from the Unitree SDK's own corrected H2JointIndex
 * (unitree_sdk2_python example/h2/low_level/h2_ankle_swing_example.py, fixed in
 * commit 65691c8 "Correct the order and spelling errors in the H2 joint index").
 *
 * IMPORTANT: unlike G1, H2's hardware order is NOT its MJCF actuator order.
 * G1's two orders coincide, which makes it easy to assume the same holds here.
 * It does not -- 17 of 31 joints sit at a different index:
 *
 *   waist    hardware roll, pitch, yaw   vs   MJCF yaw, roll, pitch
 *   wrists   hardware yaw, pitch, roll   vs   MJCF roll, pitch, yaw
 *   head     hardware 29-30 (last)       vs   MJCF 15-16 (after the waist)
 *
 * Only the twelve leg joints agree. Anything indexed by a DDS motor index must
 * use this enum; anything indexed by MJCF/policy order must not. Convert with
 * H2_HARDWARE_TO_MUJOCO / H2_MUJOCO_TO_HARDWARE below.
 *
 * The SDK also exposes WaistA/WaistB and AnkleA/AnkleB aliases, so H2 does have
 * coupled mechanisms addressable in parallel mode; this port only ever drives
 * them in series, which is why no A/B aliases are defined here.
 */
enum H2JointIndex {
  H2_LeftHipPitch = 0,
  H2_LeftHipRoll = 1,
  H2_LeftHipYaw = 2,
  H2_LeftKnee = 3,
  H2_LeftAnkleRoll = 4,
  H2_LeftAnklePitch = 5,
  H2_RightHipPitch = 6,
  H2_RightHipRoll = 7,
  H2_RightHipYaw = 8,
  H2_RightKnee = 9,
  H2_RightAnkleRoll = 10,
  H2_RightAnklePitch = 11,
  H2_WaistRoll = 12,
  H2_WaistPitch = 13,
  H2_WaistYaw = 14,
  H2_LeftShoulderPitch = 15,
  H2_LeftShoulderRoll = 16,
  H2_LeftShoulderYaw = 17,
  H2_LeftElbow = 18,
  H2_LeftWristYaw = 19,
  H2_LeftWristPitch = 20,
  H2_LeftWristRoll = 21,
  H2_RightShoulderPitch = 22,
  H2_RightShoulderRoll = 23,
  H2_RightShoulderYaw = 24,
  H2_RightElbow = 25,
  H2_RightWristYaw = 26,
  H2_RightWristPitch = 27,
  H2_RightWristRoll = 28,
  H2_HeadPitch = 29,
  H2_HeadYaw = 30,
};

/// hardware index -> MJCF actuator index.
static const std::array<int, H2_NUM_MOTOR> H2_HARDWARE_TO_MUJOCO = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 13, 14, 12, 17,
    18, 19, 20, 23, 22, 21, 24, 25,
    26, 27, 30, 29, 28, 15, 16};

/// MJCF actuator index -> hardware index.
static const std::array<int, H2_NUM_MOTOR> H2_MUJOCO_TO_HARDWARE = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 14, 12, 13, 29,
    30, 15, 16, 17, 18, 21, 20, 19,
    22, 23, 24, 25, 28, 27, 26};

/// Joint identifiers in HARDWARE order (see the enum note above).
static const std::array<std::string, H2_NUM_MOTOR> H2_JOINT_NAMES = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_roll", "left_ankle_pitch", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_roll", "right_ankle_pitch",
    "waist_roll", "waist_pitch", "waist_yaw", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_yaw",
    "left_wrist_pitch", "left_wrist_roll", "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow", "right_wrist_yaw", "right_wrist_pitch",
    "right_wrist_roll", "head_pitch", "head_yaw"};

/// Human-readable names in HARDWARE order, spoken by the TTS high-temperature
/// warning. See the note on G1_JOINT_DISPLAY_NAMES.
static const std::array<std::string, H2_NUM_MOTOR> H2_JOINT_DISPLAY_NAMES = {
    "Left Hip Pitch", "Left Hip Roll", "Left Hip Yaw", "Left Knee",
    "Left Ankle Roll", "Left Ankle Pitch", "Right Hip Pitch", "Right Hip Roll",
    "Right Hip Yaw", "Right Knee", "Right Ankle Roll", "Right Ankle Pitch",
    "Waist Roll", "Waist Pitch", "Waist Yaw", "Left Shoulder Pitch",
    "Left Shoulder Roll", "Left Shoulder Yaw", "Left Elbow", "Left Wrist Yaw",
    "Left Wrist Pitch", "Left Wrist Roll", "Right Shoulder Pitch", "Right Shoulder Roll",
    "Right Shoulder Yaw", "Right Elbow", "Right Wrist Yaw", "Right Wrist Pitch",
    "Right Wrist Roll", "Head Pitch", "Head Yaw"};

#endif // ROBOT_PARAMETERS_H2_HPP
