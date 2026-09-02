#!/usr/bin/env python3
"""Teleoperate H2 in MuJoCo from a VR headset, with the robot's view in the headset.

The XRoboToolkit path (``run_h2_mujoco_onnx.py --pico``) sends tracking one way:
poses come from the headset, and the operator watches a monitor. Its SDK has no
video functions at all, so there is no way to show the robot's view through it.

TeleVuer solves both halves. It serves a WebXR page the headset's browser opens,
pushes images to it, and reads head and controller poses back -- so the operator
sees H2's head camera and drives it with the same device.

    display_mode="immersive"  the robot's first-person view fills the headset
    display_mode="ego"        that view in a centre window, real world around it
    display_mode="pass-through"  real world only, no robot view

WebXR requires a secure context, so this serves HTTPS with a self-signed
certificate. The headset will warn about it once; accept and continue.

Usage:
    .venv_televuer/bin/python gear_sonic/scripts/run_h2_televuer.py
    .venv_televuer/bin/python gear_sonic/scripts/run_h2_televuer.py --display-mode ego

Then on the headset, open https://<this machine's IP>:8012 and press "Enter VR".
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "gear_sonic", "scripts", "run_h2_mujoco_onnx.py")

# Stereo pair, side by side. TeleVuer takes (height, width) for the combined image.
EYE_W, EYE_H = 640, 480
# Horizontal separation between the two eye cameras. Roughly human interpupillary
# distance so the stereo depth cue reads correctly.
IPD = 0.063


def _load_runner():
    spec = importlib.util.spec_from_file_location("h2runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TeleVuerSource:
    """3-point targets from TeleVuer, matching PicoSource's interface.

    Deliberately mirrors PicoSource: the same zeroing, the same clamp and rate
    limits, so the safety work done there is not silently lost by swapping the
    input device. TeleVuer reports poses as 4x4 SE(3) matrices already converted
    to the robot convention, so no change of basis is needed here -- unlike the
    XRoboToolkit path, which delivers raw OpenXR.
    """

    def __init__(self, wrapper, mod, position_gain=1.0, max_offset=0.30, track_head=True):
        self.tv = wrapper
        self.mod = mod
        self.gain = position_gain
        self.max_offset = max_offset
        self.track_head = track_head
        self.zero = None
        self.live = False
        self._last = {}
        self._last_head = np.zeros(2)
        self.duration = float("inf")

    @staticmethod
    def _pos(mat):
        return np.asarray(mat)[:3, 3]

    @staticmethod
    def _pitch_yaw(mat):
        fwd = np.asarray(mat)[:3, 0]
        import math

        return (-math.asin(np.clip(fwd[2], -1.0, 1.0)), math.atan2(fwd[1], fwd[0]))

    def _read(self):
        d = self.tv.get_tele_data()
        # A headset that has not connected yet reports identity matrices.
        self.live = d is not None and d.head_pose is not None
        return d

    def set_zero(self):
        d = self._read()
        if not self.live:
            print("  [televuer] no tracking data -- is the headset in VR?")
            return False
        self.zero = {
            "left": self._pos(d.left_wrist_pose).copy(),
            "right": self._pos(d.right_wrist_pose).copy(),
            "head": self._pitch_yaw(d.head_pose),
        }
        return True

    def poll(self):
        d = self._read()
        # Either trigger zeroes, so the operator never removes the headset to do it.
        if d is not None and (getattr(d, "right_ctrl_trigger", False)
                              or getattr(d, "left_ctrl_trigger", False)):
            if self.zero is None and self.set_zero():
                print("  [televuer] zeroed")
        return self.zero is not None

    def targets(self, t):
        if self.zero is None:
            return {}
        d = self._read()
        if not self.live:
            return dict(self._last)
        step = self.mod.PicoSource.MAX_TARGET_SPEED * (self.mod.SIM_DT * self.mod.DECIMATION)
        out = {}
        for key, mat in (("left", d.left_wrist_pose), ("right", d.right_wrist_pose)):
            delta = (self._pos(mat) - self.zero[key]) * self.gain
            delta = np.clip(delta, -self.max_offset, self.max_offset)
            prev = self._last.get(key)
            if prev is not None:
                move = delta - prev
                dist = float(np.linalg.norm(move))
                if dist > step:
                    delta = prev + move * (step / dist)
            out[key] = delta
        self._last = out
        return dict(out)

    def head(self, t):
        if self.zero is None or not self.track_head:
            return np.zeros(2)
        d = self._read()
        if not self.live:
            return self._last_head.copy()
        pitch, yaw = self._pitch_yaw(d.head_pose)
        zp, zy = self.zero["head"]
        import math

        target = np.array([
            np.clip(pitch - zp, *self.mod.PicoSource.HEAD_PITCH_RANGE),
            np.clip(math.atan2(math.sin(yaw - zy), math.cos(yaw - zy)),
                    *self.mod.PicoSource.HEAD_YAW_RANGE),
        ])
        step = self.mod.PicoSource.MAX_HEAD_SPEED * (self.mod.SIM_DT * self.mod.DECIMATION)
        move = target - self._last_head
        dist = float(np.linalg.norm(move))
        if dist > step:
            target = self._last_head + move * (step / dist)
        self._last_head = target
        return target


def stereo_frame(renderer, model, data, mujoco, cam_id):
    """Render the head camera twice, offset horizontally, as a side-by-side pair.

    MuJoCo has one camera in the model, so the two eyes are produced by nudging
    the camera body position between renders rather than by defining two cameras
    -- a scene cannot add a camera to an included body.
    """
    eyes = []
    base = model.cam_pos[cam_id].copy()
    for dx in (-IPD / 2.0, IPD / 2.0):
        model.cam_pos[cam_id] = base + np.array([0.0, dx, 0.0])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        eyes.append(renderer.render())
    model.cam_pos[cam_id] = base
    return np.hstack(eyes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", default=os.path.join(REPO, "h2_policy", "onnx",
                                                  "model_step_100000_teleop.onnx"))
    p.add_argument("--display-mode", choices=["immersive", "ego", "pass-through"],
                   default="immersive")
    p.add_argument("--cert", default=os.path.join(REPO, "certs", "cert.pem"))
    p.add_argument("--key", default=os.path.join(REPO, "certs", "key.pem"))
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--no-head", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0)
    args = p.parse_args(argv)

    import mujoco
    import onnxruntime as ort
    from televuer import TeleVuerWrapper

    mod = _load_runner()
    model, spec = mod.build_scene(mujoco)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head")
    renderer = mujoco.Renderer(model, height=EYE_H, width=EYE_W)

    tv = TeleVuerWrapper(
        use_hand_tracking=False,            # controllers, matching the Pico path
        binocular=True,
        img_shape=(EYE_H, EYE_W * 2),
        display_fps=30.0,
        display_mode=args.display_mode,
        zmq=args.display_mode != "pass-through",
        cert_file=args.cert,
        key_file=args.key,
    )
    source = TeleVuerSource(tv, mod, position_gain=args.gain,
                            track_head=not args.no_head)
    reference = mod.TeleopReference(spec, model, mujoco, target_fn=source.targets)
    obs = mod.ObservationBuilder(spec)

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, mod.INIT_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = spec.default_mj
    mujoco.mj_forward(model, data)
    action = np.zeros(mod.NUM_DOF)
    obs.reset(data, action)

    print(f"display    {args.display_mode}")
    print(f"serving    https://<this machine>:8012   (self-signed; accept the warning)")
    print()
    print("  On the headset: open that URL, press \"Enter VR\", stand in the robot's")
    print("  stance and squeeze either trigger to zero. Nothing is commanded until")
    print("  you do.")
    print()

    control_dt = mod.SIM_DT * mod.DECIMATION
    n = None if not args.seconds else int(args.seconds / control_dt)
    step = -1
    try:
        while n is None or step + 1 < n:
            step += 1
            t = step * control_dt
            source.poll()
            heading = mod.heading_quat(data.qpos[3:7])
            model_in = np.concatenate([
                reference.reference_block(t, heading), obs.proprioception()
            ]).astype(np.float32)[None, :]
            action = np.clip(session.run(None, {in_name: model_in})[0][0].astype(np.float64),
                             -mod.ACTION_CLIP, mod.ACTION_CLIP)
            target = spec.il_to_mj_vec(action * spec.action_scale_il + spec.default_il)
            target[spec.head_mj] = source.head(t)
            for _ in range(mod.DECIMATION):
                torque = spec.kp * (target - data.qpos[7:]) - spec.kd * data.qvel[6:]
                data.ctrl[:] = np.clip(torque, -spec.effort_limit, spec.effort_limit)
                mujoco.mj_step(model, data)
            obs.update(data, action)

            # The headset renders at 30 Hz; the control loop runs at 50.
            if args.display_mode != "pass-through" and step % 2 == 0:
                tv.render_to_xr(stereo_frame(renderer, model, data, mujoco, cam_id))

            if data.qpos[2] < 0.4:
                print(f"FELL at {t:.1f}s")
                break
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        renderer.close()
        tv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
