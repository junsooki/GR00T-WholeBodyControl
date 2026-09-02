#!/usr/bin/env python3
"""Stress the H2 teleop policy with synthetic target motions and report stability.

Runs the teleop head against scripted hand/head targets that reproduce the ways
a real operator (and a flaky tracker) drive it, and reports whether the robot
stayed up. No headset required, so a change can be evaluated before anyone puts
one on.

The scenarios are chosen around what actually destabilises the policy: the legs
track a frozen standing reference and only the arms follow the operator, so
every failure mode here is the arms moving the centre of mass faster than the
balance controller can answer.

Usage:
    .venv/bin/python gear_sonic/scripts/test_h2_teleop_scenarios.py
    .venv/bin/python gear_sonic/scripts/test_h2_teleop_scenarios.py --seconds 20
"""

import argparse
import importlib.util
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "gear_sonic", "scripts", "run_h2_mujoco_onnx.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("h2runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- scenarios -------------------------------------------------------------
# Each returns left/right hand offsets in metres, in the pelvis frame.

def hold(t):
    return {}


def slow_reach(t):
    """A calm reach forward and up: the well-behaved case."""
    a = min(t / 4.0, 1.0)
    return {"left": [0.25 * a, 0.0, 0.25 * a], "right": [0.25 * a, 0.0, 0.25 * a]}


def fast_wave(t):
    """Both arms oscillating at 1 Hz -- sustained centre-of-mass disturbance."""
    s = 0.30 * math.sin(2 * math.pi * 1.0 * t)
    return {"left": [0.10, 0.0, 0.25 + s], "right": [0.10, 0.0, 0.25 - s]}


def step_jump(t):
    """Instantaneous 0.4 m jump every 2 s: what an unfiltered tracker glitch does."""
    on = int(t // 2) % 2 == 1
    v = 0.40 if on else 0.0
    return {"left": [v, 0.0, v], "right": [v, 0.0, v]}


def dropout(t):
    """Reach out, then lose tracking for 2 s, repeatedly.

    Reproduces the real failure: tracking drops and the targets either hold or
    snap back to the default pose. Returning {} is exactly what PicoSource does
    on a lost frame.
    """
    if 4.0 < t < 6.0 or 10.0 < t < 12.0:
        return {}
    a = min(t / 3.0, 1.0)
    return {"left": [0.30 * a, 0.0, 0.30 * a], "right": [0.30 * a, 0.0, 0.30 * a]}


def extreme_reach(t):
    """Push to PicoSource's clamp in every axis at once, both arms opposed.

    This is the case that set the clamp: a sweep showed the robot topples at
    0.40 m and above and holds at 0.35, so PicoSource.max_offset is 0.35.
    """
    a = min(t / 3.0, 1.0)
    c = 0.35
    return {"left": [c * a, c * 0.67 * a, c * a],
            "right": [c * a, -c * 0.67 * a, -c * 0.67 * a]}


def asymmetric(t):
    """One arm far out to the side: the worst lateral centre-of-mass case."""
    a = min(t / 3.0, 1.0)
    return {"left": [0.2 * a, 0.55 * a, 0.35 * a], "right": [0.0, 0.0, 0.0]}


SCENARIOS = [
    ("hold default", hold),
    ("slow reach", slow_reach),
    ("fast wave 1Hz", fast_wave),
    ("step jump 0.4m", step_jump),
    ("tracking dropout", dropout),
    ("extreme reach", extreme_reach),
    ("asymmetric reach", asymmetric),
]


def run_one(mod, mujoco, ort, model, spec, session, in_name, target_fn, seconds):
    data = mujoco.MjData(model)
    reference = mod.TeleopReference(spec, model, mujoco, target_fn=target_fn)
    obs = mod.ObservationBuilder(spec)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, mod.INIT_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = spec.default_mj
    mujoco.mj_forward(model, data)
    action = np.zeros(mod.NUM_DOF)
    obs.reset(data, action)

    heights, fell_at, max_action = [], None, 0.0
    for step in range(int(seconds / (mod.SIM_DT * mod.DECIMATION))):
        t = step * mod.SIM_DT * mod.DECIMATION
        heading = mod.heading_quat(data.qpos[3:7])
        model_in = np.concatenate([
            reference.reference_block(t, heading), obs.proprioception()
        ]).astype(np.float32)[None, :]
        action = np.clip(session.run(None, {in_name: model_in})[0][0].astype(np.float64),
                         -mod.ACTION_CLIP, mod.ACTION_CLIP)
        max_action = max(max_action, float(np.abs(action).max()))
        target = spec.il_to_mj_vec(action * spec.action_scale_il + spec.default_il)
        target[spec.head_mj] = mod.head_target(t)
        for _ in range(mod.DECIMATION):
            torque = spec.kp * (target - data.qpos[7:]) - spec.kd * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -spec.effort_limit, spec.effort_limit)
            mujoco.mj_step(model, data)
        obs.update(data, action)
        heights.append(data.qpos[2])
        if fell_at is None and data.qpos[2] < 0.4:
            fell_at = t
            break
    h = np.asarray(heights)
    return h, fell_at, max_action


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", default=os.path.join(REPO, "h2_policy", "onnx",
                                                  "model_step_100000_teleop.onnx"))
    p.add_argument("--seconds", type=float, default=15.0)
    args = p.parse_args(argv)

    import mujoco
    import onnxruntime as ort

    mod = _load_runner()
    model, spec = mod.build_scene(mujoco)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    print(f"{'scenario':<20}{'min h':>8}{'end h':>8}{'|a|max':>9}  outcome")
    print("-" * 60)
    failures = 0
    for name, fn in SCENARIOS:
        h, fell, amax = run_one(mod, mujoco, ort, model, spec, session, in_name,
                                fn, args.seconds)
        ok = fell is None
        failures += 0 if ok else 1
        outcome = "ok" if ok else f"FELL at {fell:.1f}s"
        print(f"{name:<20}{h.min():>8.3f}{h[-1]:>8.3f}{amax:>9.2f}  {outcome}")
    print("-" * 60)
    print(f"{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios stayed up")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
