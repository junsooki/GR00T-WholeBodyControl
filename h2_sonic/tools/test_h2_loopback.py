#!/usr/bin/env python3
"""End-to-end plumbing test: H2 MuJoCo sim <-> dummy controller over DDS.

Launches the (patched) upstream sim headless in a subprocess, runs the dummy
controller in-process in "wave" mode on the loopback interface, and asserts:

  1. the sim publishes rt/lowstate with 31 valid motor states
  2. the controller's rt/lowcmd commands are received and actuated by the sim
     (the head-yaw joint follows the commanded sinusoid with visible range
      of motion, proving cmd -> torque -> physics -> state round trip)

This is the same wire protocol the real C++ deploy stack uses, so passing here
means the sim side is ready for an H2 policy runner.

Usage:
    python tools/test_h2_loopback.py --upstream <path-to-patched-checkout>
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import yaml

SIM_RUNNER = r"""
import sys, yaml
sys.path.insert(0, {upstream!r})
cfg = yaml.safe_load(open({cfg_path!r}))
cfg["ENABLE_ONSCREEN"] = False
cfg["ENABLE_OFFSCREEN"] = False
cfg["verbose"] = False
cfg["INTERFACE"] = None  # container-safe: let DDS pick the interface
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
sim = BaseSimulator(cfg, env_name="default", onscreen=False, offscreen=False)
print("SIM_READY", flush=True)
sim.start()
"""

WAVE_JOINT = 16  # head_yaw_joint (mujoco order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    up = os.path.abspath(args.upstream)
    cfg_path = os.path.join(up, "gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    # 1. launch the sim headless
    sim_proc = subprocess.Popen(
        [sys.executable, "-u", "-c", SIM_RUNNER.format(upstream=up, cfg_path=cfg_path)],
        cwd=up, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        t0 = time.time()
        ready = False
        for line in sim_proc.stdout:
            if "SIM_READY" in line:
                ready = True
                break
            print(f"  [sim] {line.rstrip()}")
            if time.time() - t0 > 60:
                break
        if not ready:
            raise RuntimeError("sim subprocess exited or timed out before SIM_READY")
        print("sim subprocess up")

        # 2. run the dummy controller against it
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from h2_dummy_controller import H2DummyController

        ChannelFactoryInitialize(0)  # default interface (container-safe)
        ctrl = H2DummyController(cfg, mode="wave", ramp_s=2.0)
        q0 = ctrl.wait_for_state(timeout=15.0)
        assert q0.shape == (31,) and np.all(np.isfinite(q0)), "bad initial lowstate"
        print(f"lowstate OK: 31 motors, q[hip..] = {np.round(q0[:3], 3)}")

        # drive and record the wave joint
        t_start = time.time()
        times, q_meas, q_cmd = [], [], []
        while time.time() - t_start < args.seconds:
            t = time.time() - t_start
            target = ctrl.target_at(t, q0)
            ctrl.pub.Write(ctrl.make_cmd(target))
            with ctrl.state_lock:
                q_now = ctrl.low_state.motor_state[WAVE_JOINT].q
            if t > 4.0:  # past ramp, wave active
                times.append(t)
                q_meas.append(q_now)
                q_cmd.append(target[WAVE_JOINT])
            time.sleep(0.02)

        q_meas, q_cmd = np.array(q_meas), np.array(q_cmd)
        span = q_meas.max() - q_meas.min()
        corr = float(np.corrcoef(q_meas, q_cmd)[0, 1])
        rmse = float(np.sqrt(np.mean((q_meas - q_cmd) ** 2)))
        print(f"head_yaw: commanded span {q_cmd.max()-q_cmd.min():.2f} rad, "
              f"measured span {span:.2f} rad, corr {corr:.3f}, rmse {rmse:.3f} rad")
        assert span > 0.5, "sim joint barely moved -- commands not actuated?"
        assert corr > 0.9, "sim motion does not follow commanded sinusoid"
        print("LOOPBACK TEST PASSED: sim <-> controller round trip works")
    finally:
        sim_proc.terminate()
        try:
            sim_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sim_proc.kill()


if __name__ == "__main__":
    main()
