# H2 MuJoCo sim — design notes

Everything here is derived from upstream commit `a0732b64` (see `upstream.lock`).

## Joint order (the thing that will bite you)

The WBC yaml's per-joint lists are in **H2 MJCF kinematic-tree order**, which is
also the actuator order in `h2.xml` (verified by `tools/check_h2_sim.py`), so
`MOTOR2JOINT`/`JOINT2MOTOR` are identity maps of length 31:

| idx | joint | idx | joint |
| --- | --- | --- | --- |
| 0–5 | L hip pitch/roll/yaw, knee, **ankle ROLL, ankle PITCH** | 17–23 | L shoulder p/r/y, elbow, wrist r/p/y |
| 6–11 | R leg (same order) | 24–30 | R arm (same order) |
| 12–14 | waist yaw/roll/pitch | | |
| 15–16 | head pitch/yaw | | |

Differences from G1 (29 DOF):

- **Ankle order is roll-then-pitch** — G1 is pitch-then-roll.
- **Two head joints** (pitch, yaw) sit between the waist and the arms. Upper
  body = waist 3 + head 2 + arms 14 = **19** (`NUM_UPPER_BODY_JOINTS`,
  `ref_upper_dof_pos`), vs 17 on G1.
- No hands in the MJCF → `NUM_HAND_MOTORS: 0` (G1's scene_43dof has 2×7
  Dex3 hand joints).
- `waist_pitch` is index 14 on both robots, so the sim-to-real KD tweak in
  `configs.py::override_wbc_config` (index 14, real robot only) stays correct.

## Gain provenance

`MOTOR_KP/KD` mirror the Isaac Lab implicit-actuator gains in
`gear_sonic/envs/manager_env/robots/h2.py`: stiffness = armature × (2π·10 Hz)²,
damping = 2 · ζ · armature · (2π·10 Hz) with ζ = 2, per actuator group
(7520-22 hips/knees, 7520-14 hip-yaw/waist-yaw, 2×5020 ankles/waist-r-p/head,
5020 arms, 4010 wrist pitch/yaw). They are intentionally the **training** gains:
an H2 SONIC policy will be trained against these, so the sim-side PD must match.
They're much softer than G1's hand-tuned deploy KPs (e.g. arm KP ≈ 14 vs 100),
so a bare PD hold sags visibly under gravity — that's expected, not a bug.

- Position limits: from `h2.xml` joint `range` (checked exactly by the smoke test).
- Effort limits: from `h2.xml` `actuatorfrcrange` — what MuJoCo actually
  enforces. Note training uses different (higher, motor-datasheet) values in
  `robots/h2.py`; if you want the sim to clip like Isaac Lab, raise the MJCF
  ranges instead of editing the yaml.
- Velocity limits: from `robots/h2.py` `velocity_limit_sim`.
- `DEFAULT_DOF_ANGLES`: from `H2_CFG.init_state` (hip −0.312, knee 0.669,
  ankle-pitch −0.363, elbow 0.6, shoulder pre-pose ±0.2).

Regenerate the yaml with
`python tools/generate_h2_wbc_yaml.py --upstream <checkout>` rather than
editing numbers by hand.

## Why each upstream patch hunk exists

1. **`h2.xml` meshdir** — upstream says `meshdir="meshes/"` but `mjcf/meshes/`
   doesn't exist; the STLs are at `../urdf/h2/meshes/`. MuJoCo resolves
   `meshdir` relative to the *main* xml's directory, which is why
   `scene_h2.xml` lives next to `h2.xml` in `mjcf/` (unlike G1, whose scene
   sits in `robot_model/model_data/g1/` beside its own robot xml).
2. **`base_sim.py` joint filter** — `body_joint_index` is built from joint-name
   substrings (`hip/knee/ankle/waist/shoulder/elbow/wrist`); H2's
   `head_pitch/head_yaw` didn't match, tripping
   `assert len(body_joint_index) == NUM_JOINTS`. Added `"head"`.
3. **`base_sim.py` elastic band** — attachment body was chosen by
   `"g1" in ROBOT_TYPE` / `"h1" in ROBOT_TYPE` / else `base_link`; H2 has no
   `base_link` → KeyError. Added an `"h2"` branch attaching at `pelvis`
   (the root body, same choice as G1-with-waist).
4. **`configs.py`** — `sonic_h2` added to `WBC_VERSIONS` and to
   `load_wbc_yaml()`.
5. **`run_sim_loop.py`** — `instantiate_g1_robot_model()` builds a G1 pinocchio
   model; `SimWrapper` stores it but the MuJoCo sim never reads it. For H2 we
   pass `None`. If Python-side WBC/IK for H2 is ever needed, write
   `gear_sonic/data/robot_model/instantiation/h2.py` mirroring `g1.py` instead.

## Still needed for full H2 teleop (out of scope here)

- H2 SONIC checkpoint (training in progress elsewhere) exported to ONNX/TensorRT.
- Terminal-2 deploy stack (`gear_sonic_deploy`): the C++ controller assumes 29
  motors and G1 observation dims; needs an H2 config (31 motors,
  `ref_upper_dof_pos: 19`, this yaml's joint order) and the H2 policy paths.
- Retargeted H2 motion data if mimic modes are wanted (`mimic_models` is empty).

## Smoke-test expectations

`tools/check_h2_sim.py` output on a good setup: all structural checks OK, and a
finite 3 s PD-hold rollout. The robot **falls** in that rollout (pelvis height
≈ 0.1 m at the end) because nothing balances it — in the interactive viewer the
elastic band (`ENABLE_ELASTIC_BAND: True`, keys 9/8/7) holds it up instead.
