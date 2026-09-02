/**
 * @file robot_common.hpp
 * @brief Definitions shared by every supported robot.
 *
 * Split out of robot_parameters.hpp, which mixed genuinely G1-specific things
 * (motor count, joint enum, command struct) with things that are identical for
 * any humanoid this stack drives (DDS topic names, operator/heading state).
 * Keeping the shared half here lets a second robot be added without either
 * duplicating these or dragging in G1's motor count.
 */

#ifndef ROBOT_COMMON_HPP
#define ROBOT_COMMON_HPP

#include <array>
#include <string>

// ---------------------------------------------------------------------------
// Unitree SDK DDS topic names. Identical across G1 and H2 -- both speak the hg
// IDL; only the motor count inside the message differs.
// ---------------------------------------------------------------------------
static const std::string HG_CMD_TOPIC = "rt/lowcmd";        ///< Low-level motor command topic.
static const std::string HG_IMU_TORSO = "rt/secondary_imu"; ///< Secondary (torso) IMU topic.
static const std::string HG_STATE_TOPIC = "rt/lowstate";    ///< Low-level motor / sensor state topic.

/**
 * @brief Bundled heading state for thread-safe access via DataBuffer.
 */
struct HeadingState {
    std::array<double, 4> init_base_quat;  ///< Captured IMU base quaternion (w,x,y,z) at init.
    double delta_heading;                  ///< Cumulative heading offset (radians).

    HeadingState(const std::array<double, 4>& quat = {1.0, 0.0, 0.0, 0.0}, double delta = 0.0)
        : init_base_quat(quat), delta_heading(delta) {}
};

/**
 * @brief High-level operator signals (set by input interfaces, read by control loop).
 */
struct OperatorState {
  bool stop = false;   ///< Emergency stop requested.
  bool start = false;  ///< Control-system start requested.
  bool play = false;   ///< Motion playback active.
};

/**
 * @brief Ankle actuation mode.
 *
 * Only meaningful on G1, whose ankle is a coupled 2-DOF mechanism addressable
 * in series (pitch/roll) or parallel (A/B). H2 drives its ankle joints directly
 * and is always PR; the enum is shared so the control loop needs no #ifdef.
 */
enum class Mode {
  PR = 0, ///< Series control for Pitch / Roll joints.
  AB = 1  ///< Parallel control for A / B motors.
};

#endif // ROBOT_COMMON_HPP
