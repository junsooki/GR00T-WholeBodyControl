/**
 * @file robot_config.hpp
 * @brief Single include that selects the robot this binary is built for.
 *
 * Include this instead of robot_parameters.hpp / policy_parameters.hpp. It pulls
 * in exactly one robot's parameter headers and re-exports them under
 * robot-neutral names, so the control code carries no robot-specific symbol.
 *
 * Selected at compile time with -DROBOT_H2=1 (see CMakeLists.txt, which builds
 * a separate target per robot). Compile time rather than runtime because
 * NUM_MOTOR is a std::array bound in sixteen places, including every member of
 * MotorCommand, which is copied through a DataBuffer on the 500 Hz command
 * path -- making that dynamic would mean heap allocation in the real-time loop.
 *
 * One binary drives one robot over one DDS interface, so there is nothing to be
 * gained from supporting both at once, and a compile-time split makes the
 * checkpoint-dimension guard in control_policy.hpp impossible to bypass.
 */

#ifndef ROBOT_CONFIG_HPP
#define ROBOT_CONFIG_HPP

#include "robot_common.hpp"

#if defined(ROBOT_H2) && ROBOT_H2

  #include "robot_parameters_h2.hpp"
  #include "policy_parameters_h2.hpp"

  constexpr int NUM_MOTOR = H2_NUM_MOTOR;                 // 31
  constexpr int NUM_BODIES = 32;                          // pelvis + 31 links
  constexpr int NUM_UPPER_BODY_JOINTS = 19;               // waist 3 + head 2 + arms 14
  constexpr int NUM_LOWER_BODY_JOINTS = 12;
  constexpr int NUM_WRIST_JOINTS = 6;

  using MotorCommand = H2MotorCommand;

  inline const auto& JOINT_NAMES        = H2_JOINT_NAMES;
  inline const auto& kps                = h2_kps;
  inline const auto& kds                = h2_kds;
  inline const auto& action_scale       = h2_action_scale;
  inline const auto& default_angles     = h2_default_angles;
  inline const auto& isaaclab_to_mujoco = h2_isaaclab_to_mujoco;
  inline const auto& mujoco_to_isaaclab = h2_mujoco_to_isaaclab;
  inline const auto& vr_3point_index    = h2_vr_3point_index;
  inline const auto& vr_5point_index    = h2_vr_5point_index;

  inline const auto& upper_body_joint_mujoco_order_in_isaaclab_index   = h2_upper_body_joint_mujoco_order_in_isaaclab_index;
  inline const auto& upper_body_joint_mujoco_order_in_mujoco_index     = h2_upper_body_joint_mujoco_order_in_mujoco_index;
  inline const auto& upper_body_joint_isaaclab_order_in_isaaclab_index = h2_upper_body_joint_isaaclab_order_in_isaaclab_index;
  inline const auto& upper_body_joint_isaaclab_order_in_mujoco_index   = h2_upper_body_joint_isaaclab_order_in_mujoco_index;
  inline const auto& lower_body_joint_mujoco_order_in_isaaclab_index   = h2_lower_body_joint_mujoco_order_in_isaaclab_index;
  inline const auto& lower_body_joint_mujoco_order_in_mujoco_index     = h2_lower_body_joint_mujoco_order_in_mujoco_index;
  inline const auto& lower_body_joint_isaaclab_order_in_isaaclab_index = h2_lower_body_joint_isaaclab_order_in_isaaclab_index;
  inline const auto& lower_body_joint_isaaclab_order_in_mujoco_index   = h2_lower_body_joint_isaaclab_order_in_mujoco_index;
  inline const auto& wrist_joint_mujoco_order_in_isaaclab_index        = h2_wrist_joint_mujoco_order_in_isaaclab_index;
  inline const auto& wrist_joint_mujoco_order_in_mujoco_index          = h2_wrist_joint_mujoco_order_in_mujoco_index;
  inline const auto& wrist_joint_isaaclab_order_in_isaaclab_index      = h2_wrist_joint_isaaclab_order_in_isaaclab_index;
  inline const auto& wrist_joint_isaaclab_order_in_mujoco_index        = h2_wrist_joint_isaaclab_order_in_mujoco_index;

  /// H2 drives its ankles directly; there is no coupled PR/AB mechanism.
  constexpr bool ROBOT_HAS_COUPLED_ANKLE = false;
  /// No H2 planner checkpoint exists, and the planner's qpos stride is
  /// (NUM_MOTOR + 7) -- 38 for H2, which no released model emits.
  constexpr bool ROBOT_HAS_PLANNER = false;
  /// Dex3 hands are a G1 accessory on their own DDS topics.
  constexpr bool ROBOT_HAS_DEX3_HANDS = false;
  constexpr const char* ROBOT_NAME = "h2";
  constexpr const char* ROBOT_MJCF = "gear_sonic/data/assets/robot_description/mjcf/h2.xml";

#else

  #include "robot_parameters.hpp"
  #include "policy_parameters.hpp"

  constexpr int NUM_MOTOR = G1_NUM_MOTOR;                 // 29
  constexpr int NUM_BODIES = 30;
  constexpr int NUM_UPPER_BODY_JOINTS = 17;
  constexpr int NUM_LOWER_BODY_JOINTS = 12;
  constexpr int NUM_WRIST_JOINTS = 6;

  inline const auto& JOINT_NAMES  = G1_JOINT_NAMES;
  inline const auto& action_scale = g1_action_scale;
  // kps, kds, default_angles, isaaclab_to_mujoco, mujoco_to_isaaclab,
  // vr_*point_index and the {upper,lower,wrist}_body_joint_* vectors are
  // already declared under neutral names by policy_parameters.hpp.

  constexpr bool ROBOT_HAS_COUPLED_ANKLE = true;
  constexpr bool ROBOT_HAS_PLANNER = true;
  constexpr bool ROBOT_HAS_DEX3_HANDS = true;
  constexpr const char* ROBOT_NAME = "g1";
  constexpr const char* ROBOT_MJCF = "g1/g1_29dof.xml";

#endif

static_assert(NUM_UPPER_BODY_JOINTS + NUM_LOWER_BODY_JOINTS == NUM_MOTOR,
              "upper + lower body joint counts must cover every motor");
static_assert(NUM_BODIES == NUM_MOTOR + 1,
              "every body but the root carries exactly one joint");

#endif // ROBOT_CONFIG_HPP
