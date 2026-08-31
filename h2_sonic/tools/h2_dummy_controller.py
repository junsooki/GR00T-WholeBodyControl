#!/usr/bin/env python3
"""Terminal-2 stand-in for the H2: a dummy low-level controller.

Speaks the exact protocol the real C++ deploy stack (gear_sonic_deploy) uses:
subscribes to `rt/lowstate` (unitree_hg LowState_) and publishes `rt/lowcmd`
(unitree_hg LowCmd_) at 50 Hz over DDS. Instead of running a trained policy it
sends scripted joint targets, which is enough to plumbing-test the whole
sim <-> controller loop (joint count, ordering, gains, DDS topics) before an
H2 SONIC checkpoint exists.

Run the H2 MuJoCo sim first (Terminal 1):
    python gear_sonic/scripts/run_sim_loop.py --wbc-version sonic_h2
then (Terminal 2):
    python tools/h2_dummy_controller.py --config <path-to-h2_31dof_sonic.yaml>

Modes:
    hold  - ramp from the current pose to DEFAULT_DOF_ANGLES and hold
    wave  - hold, plus sinusoids on head yaw + shoulder pitches + elbows so
            command flow is visible at a glance

Requires: unitree_sdk2py (the repo's vendored copy or upstream), pyyaml, numpy.
"""

import argparse
import threading
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

try:  # CRC is required by the real robot; the MuJoCo bridge ignores it
    from unitree_sdk2py.utils.crc import CRC
    _crc = CRC()
except Exception:  # pragma: no cover
    _crc = None

CONTROL_DT = 0.02  # 50 Hz, same as the C++ deploy control loop

# mujoco-order indices of the joints animated in wave mode
WAVE_JOINTS = {
    16: 0.6,   # head_yaw_joint: +/-0.6 rad
    17: 0.3,   # left_shoulder_pitch_joint
    24: 0.3,   # right_shoulder_pitch_joint
    20: 0.4,   # left_elbow_joint
    27: 0.4,   # right_elbow_joint
}
WAVE_FREQ_HZ = 0.25  # default; override with --wave-freq


class H2DummyController:
    def __init__(self, cfg, mode="hold", ramp_s=2.0, kp_scale=1.0, kd_scale=1.0,
                 wave_freq=WAVE_FREQ_HZ, wave_amp=1.0):
        self.n = cfg["NUM_MOTORS"]
        assert self.n == 31, f"expected 31 motors in config, got {self.n}"
        self.kp = kp_scale * np.array(cfg["MOTOR_KP"], dtype=float)
        self.kd = kd_scale * np.array(cfg["MOTOR_KD"], dtype=float)
        self.q_default = np.array(cfg["DEFAULT_DOF_ANGLES"], dtype=float)
        self.mode_machine = cfg["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
        self.mode_pr = cfg["UNITREE_LEGGED_CONST"]["MODE_PR"]
        self.mode = mode
        self.ramp_s = ramp_s
        self.wave_freq = wave_freq
        self.wave_amp = wave_amp

        self.state_lock = threading.Lock()
        self.low_state = None
        self.state_count = 0

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_handler, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

    def _state_handler(self, msg):
        with self.state_lock:
            self.low_state = msg
            self.state_count += 1

    def wait_for_state(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.state_lock:
                if self.low_state is not None:
                    return np.array(
                        [self.low_state.motor_state[i].q for i in range(self.n)]
                    )
            time.sleep(0.05)
        raise TimeoutError(
            "no rt/lowstate received -- is the H2 sim running on this interface?"
        )

    def target_at(self, t, q_start):
        """Joint target (mujoco order) at time t since controller start."""
        alpha = min(t / self.ramp_s, 1.0)
        q = (1 - alpha) * q_start + alpha * self.q_default
        if self.mode == "wave" and alpha >= 1.0:
            phase = 2 * np.pi * self.wave_freq * (t - self.ramp_s)
            for idx, amp in WAVE_JOINTS.items():
                q[idx] = self.q_default[idx] + self.wave_amp * amp * np.sin(phase)
        return q

    def make_cmd(self, q_target):
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_machine = self.mode_machine
        cmd.mode_pr = self.mode_pr
        for i in range(self.n):
            m = cmd.motor_cmd[i]
            m.mode = 1
            m.q = float(q_target[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(self.kp[i])
            m.kd = float(self.kd[i])
        if _crc is not None:
            cmd.crc = _crc.Crc(cmd)
        return cmd

    def run(self, duration=None, verbose=True):
        q_start = self.wait_for_state()
        if verbose:
            print(f"lowstate up ({self.n} motors); ramping to default over {self.ramp_s}s, mode={self.mode}")
        t0 = time.time()
        last_print = 0.0
        while duration is None or time.time() - t0 < duration:
            t = time.time() - t0
            q_target = self.target_at(t, q_start)
            self.pub.Write(self.make_cmd(q_target))
            if verbose and t - last_print > 2.0:
                with self.state_lock:
                    q_now = np.array(
                        [self.low_state.motor_state[i].q for i in range(self.n)]
                    )
                err = np.abs(q_now - q_target).max()
                print(f"t={t:5.1f}s  max |q - q_target| = {err:.3f} rad")
                last_print = t
            time.sleep(CONTROL_DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to h2_31dof_sonic.yaml")
    ap.add_argument("--interface", default="lo",
                    help="network interface (default lo = sim; pass 'default' to let DDS choose)")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--mode", choices=["hold", "wave"], default="wave")
    ap.add_argument("--duration", type=float, default=None, help="seconds to run (default: forever)")
    ap.add_argument("--ramp", type=float, default=2.0, help="seconds to ramp to default pose")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="scale PD stiffness (SIM DEMOS ONLY -- never on hardware). The raw "
                         "training gains are too soft to hold poses against gravity without a "
                         "policy compensating; ~4 gives a crisp hold/wave in the sim")
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="scale PD damping (sim demos only); ~2 pairs well with --kp-scale 4")
    ap.add_argument("--wave-freq", type=float, default=WAVE_FREQ_HZ,
                    help="wave frequency in Hz (lower = calmer, less body swing)")
    ap.add_argument("--wave-amp", type=float, default=1.0,
                    help="scale wave amplitude (smaller = less reaction on the hanging body)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.interface == "default":
        ChannelFactoryInitialize(args.domain_id)
    else:
        ChannelFactoryInitialize(args.domain_id, args.interface)
    H2DummyController(
        cfg, mode=args.mode, ramp_s=args.ramp, kp_scale=args.kp_scale, kd_scale=args.kd_scale,
        wave_freq=args.wave_freq, wave_amp=args.wave_amp,
    ).run(duration=args.duration)


if __name__ == "__main__":
    main()
