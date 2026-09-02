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

// ---------------------------------------------------------------------------
// Unitree SDK DDS topic names. Same as G1 -- both robots speak the hg IDL.
// ---------------------------------------------------------------------------
static const std::string H2_HG_CMD_TOPIC = "rt/lowcmd";
static const std::string H2_HG_IMU_TORSO = "rt/secondary_imu";
static const std::string H2_HG_STATE_TOPIC = "rt/lowstate";

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
 * @brief Symbolic names for H2 hardware joint indices (0-30).
 *
 * Note LeftAnkleRoll = 4 and LeftAnklePitch = 5 -- the reverse of G1.
 */
enum H2JointIndex {
  H2_LeftHipPitch = 0,
  H2_LeftHipRoll = 1,
  H2_LeftHipYaw = 2,
  H2_LeftKnee = 3,
  H2_LeftAnkleRoll = 4,   // roll before pitch on H2
  H2_LeftAnklePitch = 5,
  H2_RightHipPitch = 6,
  H2_RightHipRoll = 7,
  H2_RightHipYaw = 8,
  H2_RightKnee = 9,
  H2_RightAnkleRoll = 10,
  H2_RightAnklePitch = 11,
  H2_WaistYaw = 12,
  H2_WaistRoll = 13,
  H2_WaistPitch = 14,
  H2_HeadPitch = 15,      // no G1 equivalent
  H2_HeadYaw = 16,        // no G1 equivalent
  H2_LeftShoulderPitch = 17,
  H2_LeftShoulderRoll = 18,
  H2_LeftShoulderYaw = 19,
  H2_LeftElbow = 20,
  H2_LeftWristRoll = 21,
  H2_LeftWristPitch = 22,
  H2_LeftWristYaw = 23,
  H2_RightShoulderPitch = 24,
  H2_RightShoulderRoll = 25,
  H2_RightShoulderYaw = 26,
  H2_RightElbow = 27,
  H2_RightWristRoll = 28,
  H2_RightWristPitch = 29,
  H2_RightWristYaw = 30
};

/// Joint names in hardware order, for logging and for asserting against the MJCF.
static const std::array<std::string, H2_NUM_MOTOR> H2_JOINT_NAMES = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_roll", "left_ankle_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_roll", "right_ankle_pitch",
    "waist_yaw", "waist_roll", "waist_pitch",
    "head_pitch", "head_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"};

#endif // ROBOT_PARAMETERS_H2_HPP
