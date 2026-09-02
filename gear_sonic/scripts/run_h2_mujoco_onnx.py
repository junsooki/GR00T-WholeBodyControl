#!/usr/bin/env python3
"""Run the H2 SONIC ONNX policy in MuJoCo against a reference motion.

Pure ``mujoco`` + ``onnxruntime``: no Isaac Lab, no TensorRT, no C++ deploy
binary. The script rebuilds the observation layout that
``gear_sonic/utils/inference_helpers.py`` bakes into the fused per-mode ONNX
heads, so an exported ``*_g1.onnx`` can be driven directly.

Observation layout for the ``g1`` head (1670 = 680 reference + 990 proprioception)::

    command_multi_future_nonflat        620   10 future frames of 31 joint pos,
                                              then 10 frames of 31 joint vel
    motion_anchor_ori_heading_mf_nonflat 60   10 frames x 6D reference root
                                              orientation, heading-normalised
    -- proprioception, each term a 10-frame history, oldest first --
    base_ang_vel                         30
    joint_pos (relative to default)     310
    joint_vel                           310
    actions (raw, pre-scaling)          310
    gravity_dir                          30

The proprioception term order comes from the field order of
``PolicyCfg`` in ``gear_sonic/envs/manager_env/mdp/observations.py`` (Isaac Lab
concatenates in declaration order, not in the order of the Hydra defaults list)
and is independently confirmed by ``gear_sonic_deploy/policy/*/observation_config.yaml``.

Reference sources:

``static``
    Hold one fixed pose. Needs no motion data at all: ``sonic_h2.yaml`` trains
    with ``freeze_frame_aug: true``, so a frozen reference frame is in
    distribution and means "hold this pose and stay balanced". Use this first --
    it exercises the observation layout, action scaling and PD gains without
    depending on a dataset being correct.

``motion``
    A motion-library PKL as written by
    ``gear_sonic/data_process/convert_h2_csv_to_motion_lib.py``.

Usage::

    .venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py --seconds 10
    .venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py --viewer
    .venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py \
        --reference motion --motion-file data/h2_motions/robot.pkl --video out.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H2_XML = os.path.join(REPO_ROOT, "gear_sonic", "data", "assets", "robot_description", "mjcf", "h2.xml")

# Control rates, from the exported env_config: sim_dt 0.005, decimation 4.
SIM_DT = 0.005
DECIMATION = 4
HISTORY_LEN = 10          # actor_prop_history_length / actor_actions_history_length
NUM_FUTURE_FRAMES = 10    # commands.motion.num_future_frames
DT_FUTURE_REF = 0.1       # commands.motion.dt_future_ref_frames
ACTION_CLIP = 20.0        # env_config.action_clip_value
NUM_DOF = 31
INIT_HEIGHT = 1.04        # H2_CFG.init_state.pos[2]

# --------------------------------------------------------------------------
# Actuator model. Mirrors gear_sonic/envs/manager_env/robots/h2.py -- same
# armature constants, same natural frequency, same effort limits (which that
# file takes from h2.urdf). Kept as a table here because h2.py imports Isaac
# Lab, which this script deliberately does not depend on.
# --------------------------------------------------------------------------
ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

# joint suffix -> (motor class, gain multiplier, effort limit [Nm])
ACTUATOR_TABLE = {
    "hip_pitch": ("7520_22", 1.0, 360.0),
    "hip_roll": ("7520_22", 1.0, 360.0),
    "hip_yaw": ("7520_14", 1.0, 360.0),
    "knee": ("7520_22", 1.0, 360.0),
    "ankle_pitch": ("5020", 2.0, 66.88),
    "ankle_roll": ("5020", 2.0, 19.0),
    "waist_yaw": ("7520_14", 1.0, 120.0),
    "waist_roll": ("5020", 2.0, 180.0),
    "waist_pitch": ("5020", 2.0, 180.0),
    "head_pitch": ("5020", 2.0, 50.0),
    "head_yaw": ("5020", 2.0, 50.0),
    "shoulder_pitch": ("5020", 1.0, 120.0),
    "shoulder_roll": ("5020", 1.0, 54.0),
    "shoulder_yaw": ("5020", 1.0, 54.0),
    "elbow": ("5020", 1.0, 54.0),
    "wrist_roll": ("5020", 1.0, 54.0),
    "wrist_pitch": ("4010", 1.0, 25.0),
    "wrist_yaw": ("4010", 1.0, 25.0),
}

# H2_CFG.init_state.joint_pos, keyed the same way (side-specific entries win).
DEFAULT_JOINT_POS = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": 0.6,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.2,
}

# h2.py's H2_MUJOCO_TO_ISAACLAB_DOF. Despite the name it maps an Isaac Lab DOF
# index to the MuJoCo DOF index, i.e. il_to_mj_dof[il_index] = mujoco_index.
# Verified at load time against a breadth-first walk of the MJCF body tree.
IL_TO_MJ_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10,
    16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30,
]


# --------------------------------------------------------------------------
# Quaternion helpers. Isaac Lab and MuJoCo both use wxyz.
# --------------------------------------------------------------------------
def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_apply(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def heading_quat(q):
    """Yaw-only quaternion, matching torch_transform.get_heading_q."""
    out = np.array([q[0], 0.0, 0.0, q[3]])
    n = np.linalg.norm(out)
    return out / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def rot6d(q):
    """First two columns of the rotation matrix, flattened row-wise.

    Matches ``matrix_from_quat(q)[..., :2].reshape(-1)`` in commands.py.
    """
    return quat_to_mat(q)[:, :2].reshape(-1)


# --------------------------------------------------------------------------
# Robot description
# --------------------------------------------------------------------------
class H2Spec:
    """Joint ordering, gains and defaults, derived from the MJCF and verified."""

    def __init__(self, model, mujoco):
        self.mujoco = mujoco
        nm = lambda o, i: mujoco.mj_id2name(model, o, i)
        self.mj_joints = [nm(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        if len(self.mj_joints) != NUM_DOF:
            raise RuntimeError(f"expected {NUM_DOF} actuators, found {len(self.mj_joints)}")

        # Isaac Lab orders joints by a breadth-first walk of the body tree.
        children = defaultdict(list)
        for b in range(1, model.nbody):
            children[model.body_parentid[b]].append(b)
        bfs, queue = [], list(children[0])
        while queue:
            b = queue.pop(0)
            bfs.append(b)
            queue.extend(children[b])
        il_joints = []
        for b in bfs:
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                    il_joints.append(nm(mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix("_joint"))
        self.il_joints = il_joints

        il_to_mj = [self.mj_joints.index(j) for j in il_joints]
        if il_to_mj != IL_TO_MJ_DOF:
            raise RuntimeError(
                "Isaac Lab joint order derived from the MJCF does not match h2.py.\n"
                f"  derived: {il_to_mj}\n  h2.py:   {IL_TO_MJ_DOF}"
            )
        self.il_to_mj = np.asarray(il_to_mj)          # mujoco index of isaaclab dof i
        self.mj_to_il = np.argsort(self.il_to_mj)     # isaaclab index of mujoco dof i

        # Per-joint gains, effort limits, action scale and defaults, in MuJoCo order.
        kp, kd, arm, eff, scale, default = [], [], [], [], [], []
        for j in self.mj_joints:
            suffix = j.removeprefix("left_").removeprefix("right_")
            motor, mult, effort = ACTUATOR_TABLE[suffix]
            a = ARMATURE[motor] * mult
            k = a * NATURAL_FREQ**2
            kp.append(k)
            kd.append(2.0 * DAMPING_RATIO * a * NATURAL_FREQ)
            arm.append(a)
            eff.append(effort)
            scale.append(0.25 * effort / k)
            default.append(DEFAULT_JOINT_POS.get(j, DEFAULT_JOINT_POS.get(suffix, 0.0)))
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.armature = np.array(arm)
        self.effort_limit = np.array(eff)
        self.action_scale_mj = np.array(scale)
        self.default_mj = np.array(default)
        # Isaac Lab ordered copies, for building observations and applying actions.
        self.action_scale_il = self.action_scale_mj[self.il_to_mj]
        self.default_il = self.default_mj[self.il_to_mj]
        self.head_mj = np.array([self.mj_joints.index(j) for j in ("head_pitch", "head_yaw")])

    def mj_to_il_vec(self, v):
        return np.asarray(v)[self.il_to_mj]

    def il_to_mj_vec(self, v):
        out = np.empty(NUM_DOF)
        out[self.il_to_mj] = np.asarray(v)
        return out


def build_scene(mujoco, add_armature=True):
    """Load h2.xml with a ground plane, lighting and the training timestep.

    The MJCF ships without an <option> block or a floor, and its meshdir is
    relative to its own directory, so the compiler path is rewritten to an
    absolute one rather than copying the file elsewhere.
    """
    tree = ET.parse(H2_XML)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "")
    compiler.set("meshdir", os.path.normpath(os.path.join(os.path.dirname(H2_XML), meshdir)))

    ET.SubElement(root, "option", timestep=str(SIM_DT), integrator="implicitfast")

    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="_sky", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="512")
    ET.SubElement(asset, "texture", name="_grid", type="2d", builtin="checker",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.15 0.2", width="512", height="512")
    ET.SubElement(asset, "material", name="_grid", texture="_grid",
                  texrepeat="16 16", reflectance="0.05")

    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="_floor", type="plane", size="0 0 0.05",
                  material="_grid", condim="3")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    spec = H2Spec(model, mujoco)
    if add_armature:
        # Isaac Lab applies the actuator armature as reflected rotor inertia; the
        # MJCF sets none, so without this the legs are noticeably easier to move
        # here than in training.
        for mj_i, joint in enumerate(spec.mj_joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint + "_joint")
            model.dof_armature[model.jnt_dofadr[jid]] = spec.armature[mj_i]
    return model, spec


# --------------------------------------------------------------------------
# Reference sources. Each returns, for the current step, NUM_FUTURE_FRAMES of
# reference joint positions and velocities (Isaac Lab order) plus the reference
# root orientation per frame (wxyz, world).
# --------------------------------------------------------------------------
class StaticReference:
    """A frozen reference: hold one pose, upright, aligned with the robot heading.

    Needs no motion data. ``sonic_h2.yaml`` sets ``freeze_frame_aug: true``, so
    a constant reference frame is something the policy saw during training; it
    reads as "hold this pose and stay balanced".
    """

    name = "static"

    def __init__(self, spec, joint_pos_il=None):
        self.joint_pos = spec.default_il.copy() if joint_pos_il is None else np.asarray(joint_pos_il)
        self.duration = float("inf")

    def sample(self, t, anchor_heading_quat):
        jp = np.tile(self.joint_pos, (NUM_FUTURE_FRAMES, 1))
        jv = np.zeros((NUM_FUTURE_FRAMES, NUM_DOF))
        # Upright and facing wherever the robot faces -> the heading-normalised
        # difference is identity every frame.
        quat = np.tile(anchor_heading_quat, (NUM_FUTURE_FRAMES, 1))
        return jp, jv, quat


class MotionLibReference:
    """Reference frames from a motion-library PKL.

    Accepts the output of ``gear_sonic/data_process/convert_h2_csv_to_motion_lib.py``:
    ``dof`` (T, 31) in MuJoCo order and radians, ``root_rot`` (T, 4) in *xyzw*
    (that converter matches the G1 one and writes xyzw, not the wxyz the docs
    describe), and ``fps``.
    """

    name = "motion"

    def __init__(self, spec, path, motion_key=None):
        import joblib

        data = joblib.load(path)
        keys = list(data.keys())
        if motion_key is None:
            motion_key = keys[0]
        elif motion_key not in data:
            raise SystemExit(f"motion '{motion_key}' not in {path}. Available: {keys[:20]}")
        self.key = motion_key
        motion = data[motion_key]

        dof_mj = np.asarray(motion["dof"], dtype=np.float64)
        if dof_mj.shape[1] != NUM_DOF:
            raise SystemExit(
                f"'{motion_key}' has {dof_mj.shape[1]} DOF, expected {NUM_DOF}. "
                "G1 motions (29 DOF) cannot drive H2 -- they need retargeting first."
            )
        self.fps = int(motion.get("fps", 30))
        self.joint_pos = dof_mj[:, spec.il_to_mj]                    # -> Isaac Lab order
        self.joint_vel = np.gradient(self.joint_pos, 1.0 / self.fps, axis=0)
        root_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
        self.root_quat = root_xyzw[:, [3, 0, 1, 2]]                  # xyzw -> wxyz
        self.num_frames = len(self.joint_pos)
        self.duration = self.num_frames / self.fps
        self.available = keys

    def sample(self, t, anchor_heading_quat):
        base = t * self.fps
        idx = np.clip(
            np.round(base + np.arange(NUM_FUTURE_FRAMES) * DT_FUTURE_REF * self.fps).astype(int),
            0, self.num_frames - 1,
        )
        return self.joint_pos[idx], self.joint_vel[idx], self.root_quat[idx]


class TeleopReference:
    """Upper body from 3-point VR targets, lower body from a frozen standing pose.

    This is the reference for the ``teleop`` head, and it needs no motion data:
    the legs read a constant standing reference (in distribution thanks to
    ``freeze_frame_aug``) while the arms follow three targets -- left hand, right
    hand, head -- exactly as they would from a headset and two controllers.

    Targets are expressed in the reference anchor (pelvis) frame. The defaults
    are the robot's own default pose, computed by forward kinematics, so with no
    input the command reads "stand as you are".

    ``target_fn(t) -> dict`` optionally overrides targets per step, with keys
    ``left``/``right``/``head`` holding a 3-vector position offset (metres, in
    the pelvis frame) and optionally ``*_quat`` for a wxyz orientation. This is
    the hook a live Pico feed plugs into.
    """

    name = "teleop"
    SIZE = 6 + 240 + 9 + 12

    # commands/terms/motion.yaml: vr_3point_body and vr_3point_body_offset.
    BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    OFFSETS = np.array([[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]])

    def __init__(self, spec, model, mujoco, target_fn=None):
        self.spec = spec
        self.target_fn = target_fn
        self.duration = float("inf")

        # Lower body reference: the 12 leg joints of the default pose, held.
        # commands.py indexes the Isaac Lab ordered dof vector with
        # isaaclab_to_mujoco_dof[:12], i.e. the legs in MuJoCo order.
        self.lower_idx = np.asarray([spec.mj_to_il[i] for i in range(12)])
        lower_pos = spec.default_il[self.lower_idx]
        self.lower_block = np.concatenate([
            np.tile(lower_pos, NUM_FUTURE_FRAMES),          # 120 positions
            np.zeros(NUM_FUTURE_FRAMES * 12),               # 120 velocities
        ])

        # Default 3-point targets, by forward kinematics on the default pose.
        data = mujoco.MjData(model)
        data.qpos[:3] = [0.0, 0.0, INIT_HEIGHT]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = spec.default_mj
        mujoco.mj_forward(model, data)
        anchor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        anchor_pos, anchor_quat = data.xpos[anchor].copy(), data.xquat[anchor].copy()
        inv = quat_inv(anchor_quat)

        pos, orn = [], []
        for body, offset in zip(self.BODIES, self.OFFSETS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            world = data.xpos[bid] + quat_to_mat(data.xquat[bid]) @ offset
            pos.append(quat_apply(inv, world - anchor_pos))
            orn.append(quat_mul(inv, data.xquat[bid].copy()))
        self.default_pos = np.array(pos)     # (3, 3), pelvis frame
        self.default_orn = np.array(orn)     # (3, 4), wxyz

    def reference_block(self, t, anchor_heading_quat):
        pos = self.default_pos.copy()
        orn = self.default_orn.copy()
        if self.target_fn is not None:
            override = self.target_fn(t) or {}
            for i, key in enumerate(("left", "right", "head")):
                if key in override:
                    pos[i] = pos[i] + np.asarray(override[key], dtype=np.float64)
                if f"{key}_quat" in override:
                    orn[i] = np.asarray(override[f"{key}_quat"], dtype=np.float64)
        # Reference anchor is upright and aligned with the robot heading, so the
        # heading-normalised anchor orientation is identity.
        anchor_ori = rot6d(np.array([1.0, 0.0, 0.0, 0.0]))
        return np.concatenate([anchor_ori, self.lower_block, pos.reshape(-1), orn.reshape(-1)])


def head_target(t):
    """Commanded (head_pitch, head_yaw) in radians. Zero is level and forward.

    Replace this with the headset's pitch and yaw to have the robot's head follow
    the operator.
    """
    return np.array([0.0, 0.0])


def wave_targets(t):
    """A scripted stand-in for a live VR feed: raise both hands, then wave."""
    if t < 1.5:
        return {}
    lift = min((t - 1.5) / 2.0, 1.0)
    swing = 0.10 * math.sin(2.0 * math.pi * 0.5 * max(t - 3.5, 0.0))
    return {
        "left": [0.05 * lift, 0.0, 0.30 * lift + swing],
        "right": [0.05 * lift, 0.0, 0.30 * lift - swing],
    }


# --------------------------------------------------------------------------
# Live teleoperation from a PICO headset
# --------------------------------------------------------------------------
class PicoSource:
    """3-point targets from a PICO headset and controllers, via XRoboToolkit.

    Supplies both hooks the runner exposes: ``targets(t)`` feeds
    ``TeleopReference.target_fn`` and ``head(t)`` replaces ``head_target``.

    Requires the XRoboToolkit PC service to be running and ``xrobotoolkit_sdk``
    importable. The SDK reports poses as ``[x, y, z, qx, qy, qz, qw]`` -- note
    the quaternion is *xyzw*, while everything else here is wxyz.

    Frames: OpenXR is Y-up with -Z forward, the robot is Z-up with +X forward,
    so positions and orientations are both changed of basis by ``XR_TO_ROBOT``.

    Nothing is commanded until the operator zeroes the pose with the A button.
    Targets are deltas from that zero, so the robot holds its default stance
    until you move, regardless of where you are standing in the play space.
    """

    # robot_x = -xr_z (forward), robot_y = -xr_x (left), robot_z = +xr_y (up)
    XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # H2 head joint ranges from h2.xml, less a small margin.
    HEAD_PITCH_RANGE = (-0.50, 0.80)
    HEAD_YAW_RANGE = (-1.70, 1.70)

    def __init__(self, position_gain=1.0, max_offset=0.6, track_head=True):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise SystemExit(
                "xrobotoolkit_sdk is not importable.\n"
                "  Build it for this interpreter with:\n"
                "    uv pip install --python .venv/bin/python pybind11\n"
                "    cmake -S external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \\\n"
                "          -B build/xrsdk -Dpybind11_DIR=$(.venv/bin/python -c "
                "'import pybind11;print(pybind11.get_cmake_dir())')\n"
                "    cmake --build build/xrsdk -j8\n"
                "  then copy the resulting .so into .venv/lib/python3.11/site-packages/."
            ) from exc
        self.xrt = xrt
        self.gain = position_gain
        self.max_offset = max_offset
        self.track_head = track_head
        self.zero = None
        self.live = False
        self._warned = False
        self._prev_a = False
        xrt.init()

    @staticmethod
    def _is_live(pose):
        """True if the SDK is actually reporting a pose.

        With no headset connected the service returns all zeros, including a
        zero quaternion. Treating that as a real pose would mean a device that
        never connected -- or one that drops out mid-session -- silently
        commands the arms, so it is rejected here instead.
        """
        return abs(np.linalg.norm(np.asarray(pose, dtype=np.float64)[3:]) - 1.0) < 0.1

    @classmethod
    def _to_robot(cls, pose):
        """SDK pose -> (position in robot frame, rotation matrix in robot frame)."""
        pose = np.asarray(pose, dtype=np.float64)
        pos = cls.XR_TO_ROBOT @ pose[:3]
        wxyz = np.array([pose[6], pose[3], pose[4], pose[5]])  # xyzw -> wxyz
        rot = cls.XR_TO_ROBOT @ quat_to_mat(wxyz) @ cls.XR_TO_ROBOT.T
        return pos, rot

    def _read(self):
        raw = {
            "left": self.xrt.get_left_controller_pose(),
            "right": self.xrt.get_right_controller_pose(),
            "head": self.xrt.get_headset_pose(),
        }
        self.live = all(self._is_live(v) for v in raw.values())
        return {k: self._to_robot(v) for k, v in raw.items()}

    @staticmethod
    def _pitch_yaw(rot):
        forward = rot[:, 0]
        yaw = math.atan2(forward[1], forward[0])
        pitch = -math.asin(np.clip(forward[2], -1.0, 1.0))
        return pitch, yaw

    def set_zero(self):
        """Capture the current pose as the origin. Call with arms in the robot's stance."""
        current = self._read()
        if not self.live:
            print("  [pico] no tracking data -- is the headset connected and streaming?")
            return False
        self.zero = {k: (p.copy(), self._pitch_yaw(r)) for k, (p, r) in current.items()}
        return True

    def poll(self):
        """Refresh state. Returns True once a zero has been captured.

        The A button (re)zeroes, on the press edge rather than while held.
        """
        pressed = bool(self.xrt.get_A_button())
        if pressed and not self._prev_a and self.set_zero():
            print("  [pico] zeroed")
        self._prev_a = pressed
        return self.zero is not None

    def targets(self, t):
        if self.zero is None:
            return {}
        current = self._read()
        if not self.live:
            # Tracking dropped: hold the default stance rather than acting on
            # whatever the service last put in the buffer.
            if not self._warned:
                print("  [pico] tracking lost -- holding default stance")
                self._warned = True
            return {}
        self._warned = False
        out = {}
        for key in ("left", "right"):
            delta = (current[key][0] - self.zero[key][0]) * self.gain
            out[key] = np.clip(delta, -self.max_offset, self.max_offset)
        return out

    def head(self, t):
        """(head_pitch, head_yaw) in radians, relative to the zeroed pose."""
        if self.zero is None or not self.track_head:
            return np.zeros(2)
        head_rot = self._read()["head"][1]
        if not self.live:
            return np.zeros(2)
        pitch, yaw = self._pitch_yaw(head_rot)
        zero_pitch, zero_yaw = self.zero["head"][1]
        return np.array([
            np.clip(pitch - zero_pitch, *self.HEAD_PITCH_RANGE),
            np.clip(math.atan2(math.sin(yaw - zero_yaw), math.cos(yaw - zero_yaw)),
                    *self.HEAD_YAW_RANGE),
        ])

    def close(self):
        with contextlib.suppress(Exception):
            self.xrt.close()


# --------------------------------------------------------------------------
# Observation assembly
# --------------------------------------------------------------------------
class History:
    """Fixed-length history, flattened oldest-frame-first.

    Matches Isaac Lab's ObservationManager, which appends to a circular buffer
    and flattens it as (num_envs, history_length * dim) with the oldest entry
    first. On reset the buffer is filled with the first observation.
    """

    def __init__(self, dim, length=HISTORY_LEN):
        self.dim = dim
        self.length = length
        self.buf = None

    def reset(self, value):
        self.buf = deque([np.asarray(value, dtype=np.float64)] * self.length, maxlen=self.length)

    def append(self, value):
        self.buf.append(np.asarray(value, dtype=np.float64))

    def flat(self):
        return np.concatenate(self.buf)


class ObservationBuilder:
    """Builds the 1670-dim input of the fused ``g1`` head.

    Term order is fixed by two things, neither of which is the Hydra defaults
    list: the tokenizer half follows ``env_config.obs.group_obs_names.tokenizer``
    filtered to the terms this head needs, and the proprioception half follows
    the *field order of PolicyCfg* -- Isaac Lab concatenates in declaration
    order. The resulting proprioception order (ang vel, joint pos, joint vel,
    actions, gravity) is the same one the C++ deployment configs list.
    """

    def __init__(self, spec):
        self.spec = spec
        self.h_ang_vel = History(3)
        self.h_joint_pos = History(NUM_DOF)
        self.h_joint_vel = History(NUM_DOF)
        self.h_actions = History(NUM_DOF)
        self.h_gravity = History(3)

    @staticmethod
    def _current(spec, data):
        pelvis_quat = data.qpos[3:7].copy()             # MuJoCo free joint: wxyz
        ang_vel_b = data.qvel[3:6].copy()               # already in the body frame
        gravity = quat_apply(quat_inv(pelvis_quat), np.array([0.0, 0.0, -1.0]))
        joint_pos_il = spec.mj_to_il_vec(data.qpos[7:]) - spec.default_il
        joint_vel_il = spec.mj_to_il_vec(data.qvel[6:])
        return pelvis_quat, ang_vel_b, gravity, joint_pos_il, joint_vel_il

    def reset(self, data, last_action):
        _, ang_vel, gravity, jp, jv = self._current(self.spec, data)
        self.h_ang_vel.reset(ang_vel)
        self.h_joint_pos.reset(jp)
        self.h_joint_vel.reset(jv)
        self.h_actions.reset(last_action)
        self.h_gravity.reset(gravity)

    def update(self, data, last_action):
        _, ang_vel, gravity, jp, jv = self._current(self.spec, data)
        self.h_ang_vel.append(ang_vel)
        self.h_joint_pos.append(jp)
        self.h_joint_vel.append(jv)
        self.h_actions.append(last_action)
        self.h_gravity.append(gravity)

    def proprioception(self):
        return np.concatenate([
            self.h_ang_vel.flat(),    # 30
            self.h_joint_pos.flat(),  # 310
            self.h_joint_vel.flat(),  # 310
            self.h_actions.flat(),    # 310
            self.h_gravity.flat(),    # 30
        ])

    @staticmethod
    def reference(ref_joint_pos, ref_joint_vel, ref_root_quat, anchor_heading_quat):
        """command_multi_future_nonflat (620) then motion_anchor_ori_heading_mf_nonflat (60).

        ``command_multi_future`` concatenates all future joint positions and then
        all future joint velocities -- it is not interleaved per frame, and the
        ``_nonflat`` reshape to (10, 62) is applied to that same flat vector.
        """
        heading_inv = quat_inv(anchor_heading_quat)
        ori = np.concatenate([rot6d(quat_mul(heading_inv, q)) for q in ref_root_quat])
        return np.concatenate([
            np.asarray(ref_joint_pos).reshape(-1),  # 310
            np.asarray(ref_joint_vel).reshape(-1),  # 310
            ori,                                    # 60
        ])


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
def run(args):
    import mujoco
    import onnxruntime as ort

    model, spec = build_scene(mujoco, add_armature=not args.no_armature)
    data = mujoco.MjData(model)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    expected = session.get_inputs()[0].shape[-1]

    pico = None
    if args.reference == "static":
        reference = StaticReference(spec)
    elif args.reference == "teleop":
        target_fn = None
        if args.pico:
            pico = PicoSource(position_gain=args.pico_gain, track_head=not args.pico_no_head)
            target_fn = pico.targets
        elif args.wave:
            target_fn = wave_targets
        reference = TeleopReference(spec, model, mujoco, target_fn=target_fn)
    else:
        if not args.motion_file:
            raise SystemExit("--reference motion requires --motion-file")
        reference = MotionLibReference(spec, args.motion_file, args.motion_key)

    obs = ObservationBuilder(spec)

    # Reset to the training initial state.
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, args.height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = spec.default_mj
    mujoco.mj_forward(model, data)

    action = np.zeros(NUM_DOF)
    obs.reset(data, action)

    ref_size = TeleopReference.SIZE if args.reference == "teleop" else 680
    proprio = obs.proprioception()
    if proprio.size + ref_size != expected:
        raise SystemExit(
            f"observation size mismatch: built {proprio.size + ref_size}, "
            f"{os.path.basename(args.onnx)} expects {expected}.\n"
            f"  --reference {args.reference} needs the "
            f"{'teleop' if args.reference == 'teleop' else 'g1'} head; pass the matching --onnx."
        )
    print(f"model      {os.path.basename(args.onnx)}  input {expected}  "
          f"(reference {ref_size} + proprioception {proprio.size})")
    print(f"reference  {reference.name}" + (f"  '{reference.key}'" if args.reference == "motion" else ""))
    print("head       " + ("following the headset" if (pico and not args.pico_no_head)
                            else "commanded level and forward"))
    if pico is not None:
        print()
        print("  Stand in the robot's stance -- arms relaxed, facing forward -- and press A")
        print("  on the right controller to zero. Press A again at any time to re-zero.")
        print("  Nothing is commanded until you do.")
        print()
    print(f"armature   {'applied' if not args.no_armature else 'off'}   "
          f"control 1/{DECIMATION} of {1 / SIM_DT:.0f} Hz = {1 / (SIM_DT * DECIMATION):.0f} Hz")

    control_dt = SIM_DT * DECIMATION
    # With the viewer open and no explicit --seconds, run until the window is
    # closed: capping at 30 s meant the window vanished on its own.
    if args.seconds:
        horizon = args.seconds
    elif args.viewer:
        horizon = float("inf")
    else:
        horizon = min(reference.duration, 30.0)
    n_control = None if math.isinf(horizon) else int(horizon / control_dt)
    print(f"horizon    {'until the viewer window is closed' if n_control is None else '%.1f s' % horizon}"
          + ("   (real time)" if args.viewer else ""))

    frames = []
    heights, fell_at = [], None

    with contextlib.ExitStack() as stack:
        renderer = None
        if args.video:
            renderer = stack.enter_context(
                mujoco.Renderer(model, height=args.render_height, width=args.render_width))
        if pico is not None:
            stack.callback(pico.close)
        viewer_ctx = None
        if args.viewer:
            import mujoco.viewer

            # Entered as a context manager: closing it by hand and letting the
            # object be collected afterwards tears the GL context down twice and
            # segfaults on exit.
            viewer_ctx = stack.enter_context(mujoco.viewer.launch_passive(model, data))

        wall_start = time.perf_counter()
        step = -1
        while n_control is None or step + 1 < n_control:
            step += 1
            t = step * control_dt
            if pico is not None:
                pico.poll()
            heading = heading_quat(data.qpos[3:7])
            if args.reference == "teleop":
                ref_block = reference.reference_block(t, heading)
            else:
                ref_jp, ref_jv, ref_quat = reference.sample(t, heading)
                ref_block = obs.reference(ref_jp, ref_jv, ref_quat, heading)

            model_in = np.concatenate([ref_block, obs.proprioception()]).astype(np.float32)[None, :]
            action = session.run(None, {in_name: model_in})[0][0].astype(np.float64)
            action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)

            target_mj = spec.il_to_mj_vec(action * spec.action_scale_il + spec.default_il)
            # The head is unconstrained by the reward set, so the policy parks
            # head_yaw around 1.0 rad and the robot stares 58 degrees off-axis.
            # Nothing downstream depends on it -- the head drives no tracked body
            # and carries no load -- so the target is always commanded directly.
            # head_target is where a headset orientation goes in a teleop loop.
            target_mj[spec.head_mj] = pico.head(t) if pico is not None else head_target(t)
            for _ in range(DECIMATION):
                torque = spec.kp * (target_mj - data.qpos[7:]) - spec.kd * data.qvel[6:]
                data.ctrl[:] = np.clip(torque, -spec.effort_limit, spec.effort_limit)
                mujoco.mj_step(model, data)

            obs.update(data, action)
            heights.append(data.qpos[2])
            if fell_at is None and data.qpos[2] < 0.4:
                fell_at = t

            if renderer is not None and step % args.render_every == 0:
                renderer.update_scene(data, camera=-1)
                frames.append(renderer.render())
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                # Pace to wall clock. Without this the whole run finishes in a
                # couple of seconds and the window flashes past.
                behind = (t + control_dt) - (time.perf_counter() - wall_start)
                if behind > 0:
                    time.sleep(behind)

    heights = np.asarray(heights)
    print()
    print(f"steps      {len(heights)} control steps ({len(heights) * SIM_DT * DECIMATION:.1f} s)")
    print(f"height     start {heights[0]:.3f}  mean {heights.mean():.3f}  "
          f"min {heights.min():.3f}  end {heights[-1]:.3f}")
    print(f"outcome    {'FELL at %.1f s' % fell_at if fell_at is not None else 'stayed up'}")

    if frames:
        import imageio

        fps = round(1.0 / (SIM_DT * DECIMATION * args.render_every))
        imageio.mimsave(args.video, frames, fps=fps)
        print(f"video      {args.video} ({len(frames)} frames @ {fps} fps)")
    return 0 if fell_at is None else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", default=os.path.join(REPO_ROOT, "h2_policy", "onnx",
                                                  "model_step_100000_g1.onnx"),
                   help="fused per-mode ONNX head; the g1 head is the motion-tracking one")
    p.add_argument("--reference", choices=["static", "motion", "teleop"], default="static")
    p.add_argument("--pico", action="store_true",
                   help="teleop only: drive the hand and head targets from a PICO headset via "
                        "XRoboToolkit (the PC service must be running)")
    p.add_argument("--pico-gain", type=float, default=1.0,
                   help="scale from operator hand travel to robot hand travel")
    p.add_argument("--pico-no-head", action="store_true",
                   help="hold the head level and forward instead of following the headset")
    p.add_argument("--wave", action="store_true",
                   help="teleop only: drive the hand targets with a scripted lift-and-wave "
                        "instead of holding the default pose")
    p.add_argument("--motion-file", help="motion-library PKL from convert_h2_csv_to_motion_lib.py")
    p.add_argument("--motion-key", help="motion name inside the PKL (default: the first)")
    p.add_argument("--seconds", type=float, default=0.0, help="0 = the reference's own length")
    p.add_argument("--height", type=float, default=INIT_HEIGHT, help="initial pelvis height")
    p.add_argument("--no-armature", action="store_true",
                   help="skip applying Isaac Lab's actuator armature to the MuJoCo model")
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    p.add_argument("--video", help="write an mp4 here")
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--render-every", type=int, default=2, help="render every Nth control step")
    args = p.parse_args(argv)

    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX not found: {args.onnx}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
