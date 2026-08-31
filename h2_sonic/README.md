# H2 SONIC — Unitree H2 in the GR00T-WholeBodyControl MuJoCo stack

**Status: the trained H2 SONIC policy stands and balances in MuJoCo.**
The GR00T-WholeBodyControl stack shipped G1-only; this work (merged from
[spark318/H2_SONIC](https://github.com/spark318/H2_SONIC)) ports the simulator
side to the H2 (31 DOF) and adds a Python policy runner that drives the sim
with the [junsooki/h2_checkpoints](https://huggingface.co/junsooki/h2_checkpoints)
SONIC policy — the first Isaac-Lab-to-MuJoCo validation of that checkpoint.

The sim-side changes live directly in the tree (`gear_sonic/`): the H2 scene
(`scene_h2.xml`), the generated WBC config (`h2_31dof_sonic.yaml`, registered
as `--wbc-version sonic_h2`), and fixes to `h2.xml` (meshdir, joint armature),
`base_sim.py`, the DDS bridge, and a G1-only import. This directory holds the
controller-side tools, deploy data, and docs.

## Quick start

From the repo root:

```bash
pip install --user mujoco numpy scipy pyyaml tyro onnxruntime huggingface_hub
pip install --user external_dependencies/unitree_sdk2_python
# (if the SDK install fights your pip, PYTHONPATH the source dir instead — see docs)

# If the H2 meshes are git-lfs pointers, fetch them first:
git lfs pull --include "gear_sonic/data/assets/robot_description/urdf/h2/meshes/**"
```

Run (two terminals, both from the repo root):

```bash
# Terminal 1 — MuJoCo sim with viewer
python3 gear_sonic/scripts/run_sim_loop.py --wbc-version sonic_h2

# Terminal 2 — trained SONIC policy (downloads the ONNX from HF on first run)
python3 h2_sonic/tools/h2_policy_runner.py \
  --config gear_sonic/utils/mujoco_sim/wbc_configs/h2_31dof_sonic.yaml \
  --hf junsooki/h2_checkpoints            # add --mode wave to wave while balancing
```

The robot ramps to its default crouch, then the policy takes over tracking a
synthetic standing reference. Viewer keys: `9` toggles the elastic band
(virtual gantry), `8`/`7` raise/lower it. Ease the band down (`7`) until the
feet take weight before releasing (`9`).

No trained policy needed? `h2_sonic/tools/h2_dummy_controller.py` drives the
sim with scripted poses over the same DDS protocol (plumbing test / demo), and
`h2_sonic/tools/test_h2_loopback.py --upstream .` runs a full headless
sim↔controller round-trip test with pass/fail assertions.

## What was done, in order

1. **H2 in the MuJoCo sim.** The sim was hardwired to G1. Added an H2 scene
   (`scene_h2.xml`) and a generated WBC config (`h2_31dof_sonic.yaml` — joint
   order/limits read from the H2 MJCF, PD gains from the Isaac Lab training
   config), registered as `--wbc-version sonic_h2`.
2. **Five sim bugs fixed** (now applied directly in-tree): broken `meshdir`
   in `h2.xml`; the joint name filter in `base_sim.py` missing H2's head
   joints; the elastic band having no H2 attachment body; the DDS bridge
   rejecting non-G1 robot types; a G1-only import forcing pinocchio on H2
   runs.
3. **Terminal-2 groundwork.** A dummy controller speaking the exact deploy
   protocol (rt/lowstate in, rt/lowcmd out, 50 Hz, unitree_hg messages),
   an automated loopback test, and `h2_sonic/deploy/h2_policy_parameters.hpp`
   — the H2 analog of the C++ deploy stack's robot data (mappings/gains/action
   scales, verified numerically identical to the sim config).
4. **The policy runner** (`h2_sonic/tools/h2_policy_runner.py`).
   Reverse-engineered the checkpoint's observation contract from its
   `model_config.yaml` plus the export/env source: a 1670-dim vector =
   reference joint pos+vel (10 future frames) | reference root orientation
   (6D, 10 frames) | proprioception (10-frame histories of gyro, joint pos,
   joint vel, actions, gravity — in that order, IsaacLab joint ordering).
   Actions: clip ±20, `target = default + action × (0.25·effort/kp)`, sent
   with unscaled training gains.
5. **Two sim-to-sim fixes found by live debugging** (the policy initially
   exploded, then fell):
   - actor-obs term order follows the Python config *class field order*,
     not the yaml — gravity goes last, not first;
   - the H2 MJCF shipped with **no joint armature**, while training simulates
     the rotor inertias the gains are derived from — added armature matching
     `robots/h2.py` plus the `implicitfast` integrator. Sim PD-hold error
     dropped 0.84→0.25 rad; the policy then balanced 60+ s.

## Current state & known limits

- **Works:** standing balance under the policy; `--mode wave` tracks a waving
  reference while balancing; full DDS pipeline identical to hardware's.
- **Fragile:** free-standing after releasing the elastic band can still fall —
  the standing reference is *synthetic* (default pose + perfectly upright
  root), a pose the training data may not contain. `--anchor-pitch <deg>`
  tunes the commanded lean. The proper fix is loading a real H2-retargeted
  standing clip as the reference — next item below.
- Sim-only throughout; nothing here has touched hardware.

## Next steps

1. **Real reference motions**: a loader for H2-retargeted clips (standing/idle
   clip: 31 joint positions + root quat time series). Same loader then replays
   any retargeted motion.
2. **Teleop mode**: the checkpoint ships a teleop encoder (3-point VR input);
   the runner can grow a teleop obs path.
3. **C++ deploy port** for hardware: checklist with file/line pointers in
   `h2_sonic/docs/TERMINAL2_H2.md`; `h2_sonic/deploy/h2_policy_parameters.hpp`
   is the hard part done.

## Directory map

| Path | What |
| --- | --- |
| `tools/h2_policy_runner.py` | **Terminal 2**: runs the trained SONIC ONNX against the sim |
| `tools/h2_dummy_controller.py` | Scripted-pose stand-in for plumbing tests |
| `tools/test_h2_loopback.py` | Automated headless end-to-end test (`--upstream .` from repo root) |
| `tools/check_h2_sim.py` | Model/config consistency smoke test (`--upstream .`) |
| `tools/generate_h2_wbc_yaml.py`, `tools/generate_h2_policy_parameters.py` | Generators — regenerate instead of hand-editing |
| `tools/inspect_h2_checkpoint.py` | Prints ONNX signatures / checkpoint contents from HF |
| `deploy/h2_policy_parameters.hpp` | H2 robot data for the future C++ deploy port |
| `docs/H2_SIM_NOTES.md`, `docs/TERMINAL2_H2.md` | Design notes and the C++ port checklist |
