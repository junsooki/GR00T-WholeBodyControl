#!/usr/bin/env python3
"""Run the trained H2 SONIC policy (ONNX) against the MuJoCo sim over DDS.

This is the real Terminal 2: it replaces the dummy controller's scripted
targets with the exported `model_step_100000_g1.onnx` full-body tracking
policy from junsooki/h2_checkpoints, fed a synthetic "stand at the default
pose" reference — so the robot should balance and stand in place.

Observation contract (derived from the checkpoint's model_config.yaml and the
upstream export/env code; see docs/TERMINAL2_H2.md):

    obs_dict (1, 1670) =
      [ command_multi_future_nonflat (620)   ref joint pos (10f x 31, IsaacLab
                                             order) then ref joint vel (10f x 31)
      | motion_anchor_ori_heading_mf (60)    ref root ori vs robot heading, 6D
                                             rotmat first-2-columns, 10 frames
      | actor_obs (990) = 10-frame histories (oldest first), term order per
        PolicyCfg FIELD order (not the yaml defaults order!):
          base_ang_vel (3)     IMU gyro (pelvis frame)
          joint_pos_rel (31)   q - default   (IsaacLab order)
          joint_vel_rel (31)   qd            (IsaacLab order)
          actions (31)         previous raw policy outputs
          gravity_dir (3)      gravity in pelvis frame, from IMU quat
      ]

Action contract: action (31, IsaacLab order), clipped to +/-20, then
    target = default + action * action_scale,  action_scale = 0.25*effort/kp
with the *training* PD gains (never scaled).

Usage:
    python tools/h2_policy_runner.py \
        --config GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml \
        --onnx ~/.cache/huggingface/hub/models--junsooki--h2_checkpoints/snapshots/<hash>/onnx/model_step_100000_g1.onnx
    (or --hf junsooki/h2_checkpoints to resolve the ONNX automatically)
"""

import argparse
import sys
import os
import collections
import time

import numpy as np
import yaml

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

try:
    from unitree_sdk2py.utils.crc import CRC
    _crc = CRC()
except Exception:  # pragma: no cover
    _crc = None

CONTROL_DT = 0.02  # 50 Hz policy, matching training decimation
HISTORY = 10
FUTURE_FRAMES = 10
ACTION_CLIP = 20.0

# IsaacLab <-> MuJoCo DOF mappings, verbatim from gear_sonic robots/h2.py
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11,
    16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30,
])
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10,
    16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30,
])

# Per-joint gains come from h2_gains, which parses robots/h2.py. This file used
# to carry its own transcribed KIND_TABLE; it went stale against the corrected
# effort limits, and its single "ankle" entry could not represent ankle roll and
# pitch having different limits (19 vs 66.88 N.m) at all.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2_gains import joint_kind, load  # noqa: E402

W = 2.0 * np.pi * 10.0


def action_scales_mj(joint_names_mj):
    """action_scale = 0.25 * effort / stiffness, in MuJoCo joint order."""
    gains, consts = load(joint_names_mj)
    out = np.zeros(len(joint_names_mj), dtype=np.float32)
    for i, name in enumerate(joint_names_mj):
        family, effort, mult = gains[joint_kind(name)]
        kp = mult * consts[f"ARMATURE_{family}"] * W * W
        out[i] = 0.25 * effort / kp
    return out


def quat_rotate_inv(q_wxyz, v):
    """Rotate vector v by the inverse of quaternion (w,x,y,z)."""
    w, x, y, z = q_wxyz
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class H2PolicyRunner:
    WAVE_JOINTS_IL = {16: 0.5, 12: 0.25, 13: 0.25, 23: 0.35, 24: 0.35}
    WAVE_FREQ = 0.25       # Hz
    FUTURE_DT = 0.1        # dt between reference future frames (training config)

    def __init__(self, cfg, onnx_path, ramp_s=3.0, kp_scale=1.0, kd_scale=1.0,
                 newest_first=False, mode="stand", anchor_pitch_deg=0.0):
        import onnxruntime as ort

        self.n = cfg["NUM_MOTORS"]
        assert self.n == 31
        joint_names_mj = list(cfg["WeakMotorJointIndex"].keys())
        self.kp_mj = kp_scale * np.array(cfg["MOTOR_KP"], dtype=np.float32)
        self.kd_mj = kd_scale * np.array(cfg["MOTOR_KD"], dtype=np.float32)
        self.default_mj = np.array(cfg["DEFAULT_DOF_ANGLES"], dtype=np.float32)
        self.default_il = self.default_mj[MUJOCO_TO_ISAACLAB]
        self.mode_machine = cfg["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
        self.mode_pr = cfg["UNITREE_LEGGED_CONST"]["MODE_PR"]
        self.ramp_s = ramp_s
        self.newest_first = newest_first
        self.mode = mode
        self.anchor_pitch = np.deg2rad(anchor_pitch_deg)

        self.action_scale_il = action_scales_mj(joint_names_mj)[MUJOCO_TO_ISAACLAB]

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        in_meta = self.sess.get_inputs()[0]
        assert in_meta.shape[-1] == 1670, f"expected g1-mode ONNX (1670 inputs), got {in_meta.shape}"
        self.input_name = in_meta.name

        # synthetic standing reference: default pose, zero velocity, identity ori
        self.cmd_block = self.make_cmd_block(0.0)
        # reference root orientation vs robot heading: identity, optionally
        # pitched forward by anchor_pitch (6D = first two rotmat columns,
        # row-major: [m00,m01, m10,m11, m20,m21])
        cp, sp = np.cos(self.anchor_pitch), np.sin(self.anchor_pitch)
        six_d = np.array([cp, 0, 0, 1, -sp, 0], dtype=np.float32)
        self.anchor_block = np.tile(six_d, FUTURE_FRAMES)

        self.hist = {k: collections.deque(maxlen=HISTORY)
                     for k in ["grav", "gyro", "qrel", "qd", "act"]}
        self.last_action = np.zeros(self.n, dtype=np.float32)

        import threading
        self.state_lock = threading.Lock()
        self.low_state = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

    def _on_state(self, msg):
        with self.state_lock:
            self.low_state = msg
            self.state_seq = getattr(self, "state_seq", 0) + 1

    def read_state(self):
        with self.state_lock:
            s = self.low_state
            if s is None:
                return None
            q_mj = np.array([s.motor_state[i].q for i in range(self.n)], dtype=np.float32)
            dq_mj = np.array([s.motor_state[i].dq for i in range(self.n)], dtype=np.float32)
            quat = np.array(s.imu_state.quaternion, dtype=np.float32)  # (w,x,y,z)
            gyro = np.array(s.imu_state.gyroscope, dtype=np.float32)
        return q_mj, dq_mj, quat, gyro

    def _flat_hist(self, dq):
        arr = np.asarray(dq, dtype=np.float32)
        if self.newest_first:
            arr = arr[::-1]
        return arr.reshape(-1)

    def push_obs(self, q_mj, dq_mj, quat, gyro, action):
        self.hist["grav"].append(quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0])))
        self.hist["gyro"].append(gyro.copy())
        self.hist["qrel"].append(q_mj[MUJOCO_TO_ISAACLAB] - self.default_il)
        self.hist["qd"].append(dq_mj[MUJOCO_TO_ISAACLAB])
        self.hist["act"].append(action.copy())

    def build_obs(self):
        # term order = PolicyCfg field order: ang_vel, joint_pos, joint_vel,
        # actions, gravity (see observations.py PolicyCfg); histories oldest-first
        order = ["gyro", "qrel", "qd", "act", "grav"]
        actor = np.concatenate([
            self._flat_hist(self.hist[k]) for k in order
        ])
        obs = np.concatenate([self.cmd_block, self.anchor_block, actor]).astype(np.float32)
        assert obs.shape[0] == 1670, obs.shape[0]
        return obs[None, :]

    def make_cmd_block(self, t):
        """Reference command for 10 future frames at 0.1 s spacing.

        mode=stand: default pose, zero velocity. mode=wave: sinusoids on
        head yaw / shoulder pitches / elbows layered on the default pose,
        with consistent reference velocities.
        """
        pos = np.tile(self.default_il, (FUTURE_FRAMES, 1)).astype(np.float32)
        vel = np.zeros((FUTURE_FRAMES, self.n), dtype=np.float32)
        if self.mode == "wave":
            w = 2 * np.pi * self.WAVE_FREQ
            for k in range(FUTURE_FRAMES):
                ph = w * (t + k * self.FUTURE_DT)
                for j, amp in self.WAVE_JOINTS_IL.items():
                    pos[k, j] += amp * np.sin(ph)
                    vel[k, j] = amp * w * np.cos(ph)
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)]).astype(np.float32)

    def make_cmd(self, q_target_mj):
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_machine = self.mode_machine
        cmd.mode_pr = self.mode_pr
        for i in range(self.n):
            m = cmd.motor_cmd[i]
            m.mode = 1
            m.q = float(q_target_mj[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(self.kp_mj[i])
            m.kd = float(self.kd_mj[i])
        if _crc is not None:
            cmd.crc = _crc.Crc(cmd)
        return cmd

    def run(self, duration=None):
        print("waiting for rt/lowstate ...")
        while self.read_state() is None:
            time.sleep(0.05)
        q0 = self.read_state()[0]
        print(f"lowstate up; ramping to default pose over {self.ramp_s}s, then engaging policy")

        # phase 1: PD ramp to the default pose with temporarily boosted gains,
        # so the policy engages NEAR the reference (training episodes start on
        # it). The boost applies to the ramp only; the policy always runs with
        # the true training gains.
        kp_save, kd_save = self.kp_mj.copy(), self.kd_mj.copy()
        self.kp_mj, self.kd_mj = 4.0 * kp_save, 2.0 * kd_save
        t0 = time.time()
        while time.time() - t0 < self.ramp_s:
            a = min((time.time() - t0) / self.ramp_s, 1.0)
            self.pub.Write(self.make_cmd((1 - a) * q0 + a * self.default_mj))
            time.sleep(CONTROL_DT)
        self.kp_mj, self.kd_mj = kp_save, kd_save

        # seed histories with the current state and zero actions
        q_mj, dq_mj, quat, gyro = self.read_state()
        for _ in range(HISTORY):
            self.push_obs(q_mj, dq_mj, quat, gyro, np.zeros(self.n, dtype=np.float32))

        print("policy engaged (standing reference). Press 9 in the viewer to release the band.")
        t0 = time.time()
        last_print = 0.0
        n_ticks = 0
        last_seq, stale_ticks = -1, 0
        while duration is None or time.time() - t0 < duration:
            tick = time.monotonic()
            with self.state_lock:
                seq = getattr(self, "state_seq", 0)
            stale_ticks = stale_ticks + 1 if seq == last_seq else 0
            last_seq = seq
            if stale_ticks > 25:  # >0.5 s without a fresh lowstate
                print("ERROR: rt/lowstate stopped updating -- the sim likely "
                      "crashed or went unstable. Stopping (telemetry above this "
                      "point may be stale).")
                return
            q_mj, dq_mj, quat, gyro = self.read_state()
            if self.mode == "wave":
                self.cmd_block = self.make_cmd_block(time.time() - t0)
            obs = self.build_obs()
            action = self.sess.run(None, {self.input_name: obs})[0][0]
            action = np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
            self.push_obs(q_mj, dq_mj, quat, gyro, action)

            q_target_il = self.default_il + action * self.action_scale_il
            q_target_mj = q_target_il[ISAACLAB_TO_MUJOCO]
            self.pub.Write(self.make_cmd(q_target_mj))

            n_ticks += 1
            if n_ticks <= 5:
                print(f"tick {n_ticks}: |action| max {np.abs(action).max():6.2f} "
                      f"mean {np.abs(action).mean():5.2f}  "
                      f"argmax il-joint {int(np.abs(action).argmax())}")
            t = time.time() - t0
            if t - last_print > 2.0:
                print(f"t={t:5.1f}s  |action| max {np.abs(action).max():5.2f}  "
                      f"grav_z {self.hist['grav'][-1][2]:+.2f}  "
                      f"pose err {np.abs(q_mj - self.default_mj).max():.2f} rad")
                last_print = t
            elapsed = time.monotonic() - tick
            if CONTROL_DT - elapsed > 0:
                time.sleep(CONTROL_DT - elapsed)


def resolve_onnx(args):
    if args.onnx:
        return args.onnx
    from huggingface_hub import hf_hub_download
    return hf_hub_download(args.hf, "onnx/model_step_100000_g1.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to h2_31dof_sonic.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--onnx", help="path to model_step_100000_g1.onnx")
    g.add_argument("--hf", help="HF repo id to fetch the g1-mode ONNX from")
    ap.add_argument("--interface", default="lo")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="leave at 1.0: the policy is trained for the raw gains")
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--newest-first", action="store_true",
                    help="debug: flip history stacking to newest-first")
    ap.add_argument("--mode", choices=["stand", "wave"], default="stand",
                    help="reference motion the policy tracks: stand still, or wave "
                         "head/arms while balancing")
    ap.add_argument("--anchor-pitch", type=float, default=0.0,
                    help="reference root pitch in degrees (positive = lean forward). "
                         "Tune +-2..6 if the robot consistently tips one way on release")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    onnx_path = resolve_onnx(args)
    print(f"policy: {onnx_path}")
    if args.interface == "default":
        ChannelFactoryInitialize(args.domain_id)
    else:
        ChannelFactoryInitialize(args.domain_id, args.interface)
    H2PolicyRunner(cfg, onnx_path, ramp_s=args.ramp, kp_scale=args.kp_scale,
                   kd_scale=args.kd_scale, newest_first=args.newest_first,
                   mode=args.mode, anchor_pitch_deg=args.anchor_pitch
                   ).run(duration=args.duration)


if __name__ == "__main__":
    main()
