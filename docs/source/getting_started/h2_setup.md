# H2 Setup (Simulation and Deployment)

Bring-up guide for the Unitree **H2** (31 DOF, 32 bodies). H2 support is newer
than G1's and the two paths below are at different stages: the MuJoCo path runs
today, the C++/TensorRT deployment path is a work in progress.

```{admonition} Which path do I want?
:class: tip
- **Just want to see the policy run, or validate a checkpoint?** Use
  [MuJoCo + ONNX](#path-a-mujoco--onnx). Minutes to set up, no TensorRT, no CUDA.
- **Deploying to real H2 hardware, or using the VR teleop stack?** You need
  [the C++ deployment](#path-b-ctensorrt-deployment), which is still being ported.
```

## H2 vs G1: what actually differs

Worth reading before touching either path — several of these resolve silently
rather than erroring.

| | G1 | H2 |
|---|---|---|
| Actuated joints | 29 | **31** |
| Bodies | 30 | **32** |
| Head | none | **2 DOF** (`head_pitch`, `head_yaw`) |
| Leg chain | knee → ankle_pitch → ankle_roll | knee → **ankle_roll → ankle_pitch** |
| Distal foot body | `*_ankle_roll_link` | **`*_ankle_pitch_link`** |
| Effort limits | one per motor class | **one per joint** |

The reversed ankle order is the dangerous one. `*_ankle_roll_link` exists on both
robots, so G1-derived config resolves cleanly on H2 while pointing at a 364-vertex
mid-ankle stub instead of the actual foot. Both ankles are also 5020-class but
rated 19 Nm (roll) against 66.88 Nm (pitch), so a per-class effort limit is wrong
by a factor of three on the joint that resists sideways tipping.

## Getting the H2 policy

The H2 checkpoint is not part of the NVIDIA release — it is trained separately,
since the released v1.1 weights are 29-DOF G1 and cannot initialise H2. Fetch
your own export into `h2_policy/`:

```bash
huggingface-cli download <your-org>/h2_checkpoints --include 'onnx/*' --local-dir h2_policy
```

You should end up with five fused per-mode heads plus the training config:

| File | Input | Meaning |
|---|---|---|
| `model_step_100000_g1.onnx` | 1670 | motion tracking — reference joint trajectories |
| `model_step_100000_teleop.onnx` | 1257 | VR 3-point upper body + lower-body reference |
| `model_step_100000_smpl.onnx` | 1830 | SMPL human pose (robot-agnostic input) |
| `model_step_100000_encoder.onnx` | 1791 | all encoders, mode-selected |
| `model_step_100000_decoder.onnx` | 1054 | decoder only |

Each fused head takes `[reference terms | proprioception(990)]` and returns 31
actions directly.

## Path A: MuJoCo + ONNX

Runs the policy with nothing but MuJoCo and onnxruntime — no Isaac Lab, no
TensorRT, no CUDA, no C++ build.

### Install

```bash
python3.11 -m venv .venv
.venv/bin/pip install mujoco onnxruntime numpy joblib imageio imageio-ffmpeg
```

### Run

The default reference is a **frozen pose**, which needs no motion data at all —
`sonic_h2.yaml` trains with `freeze_frame_aug: true`, so a constant reference
frame is in distribution and reads as "hold this pose and stay balanced". Start
here: it exercises the observation layout, joint permutation, action scaling and
PD gains without depending on a dataset being correct.

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py --seconds 30
```

A converged policy holds pelvis height near its 1.04 m start and reports
`outcome  stayed up`. To watch it:

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py --viewer
```

To record instead (useful over SSH — set `MUJOCO_GL=egl` for headless rendering):

```bash
MUJOCO_GL=egl .venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py --seconds 12 --video h2.mp4
```

### Teleoperation

The `teleop` head takes three targets — left hand, right hand, head — plus a
lower-body reference, so the legs read a frozen standing pose while the arms
follow the targets. Like the static case this needs **no motion data**:

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py \
    --reference teleop --onnx h2_policy/onnx/model_step_100000_teleop.onnx --viewer
```

With no input the targets sit at the robot's own default pose, so it stands as
it is. `--wave` drives them with a scripted lift-and-wave to confirm the arms
respond; commanding a 0.30 m hand lift produces a 0.30 m lift at the wrist.

`TeleopReference` takes a `target_fn(t)` returning `left`/`right`/`head` position
offsets in the pelvis frame, plus optional `*_quat` orientations.

#### Live PICO teleoperation

`PicoSource` drives those targets from a PICO headset and controllers through
XRoboToolkit. One-time build of the SDK for this interpreter (the vendored copy
ships the headers and `libPXREARobotSDK.so`, so nothing is downloaded):

```bash
uv pip install --python .venv/bin/python pybind11
cmake -S external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 -B build/xrsdk \
      -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())') \
      -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python -DCMAKE_BUILD_TYPE=Release
cmake --build build/xrsdk -j8
cp build/xrsdk/xrobotoolkit_sdk.cpython-311-*.so .venv/lib/python3.11/site-packages/
```

With the XRoboToolkit PC service running and the headset connected:

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py \
    --reference teleop --onnx h2_policy/onnx/model_step_100000_teleop.onnx --pico --viewer
```

Press **A** on the right controller to engage: it zeroes and starts commanding.
Stand in the robot's stance — arms relaxed, facing forward — and press **A** on
the right controller to zero. Targets are deltas from that zero, so where you
stand in the play space does not matter, and nothing is commanded until you
press it. Press **A** again any time to re-zero. `--pico-gain` scales operator
hand travel to robot hand travel; `--pico-no-head` keeps the head level instead
of following the headset.

```{note}
The SDK reports poses as `[x, y, z, qx, qy, qz, qw]` — the quaternion is
**xyzw**, while the rest of the codebase is wxyz. OpenXR is also Y-up with -Z
forward against the robot's Z-up, +X forward, so positions and orientations are
both changed of basis. `PicoSource.XR_TO_ROBOT` does this and is unit-tested
against the three axes.
```

#### Seeing what the robot sees

`--camera head` views from a camera on the robot's head instead of the free
orbit camera, and works with both `--viewer` and `--video`:

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py \\
    --reference teleop --onnx h2_policy/onnx/model_step_100000_teleop.onnx \\
    --pico --camera head --viewer
```

```{note}
This renders the robot's point of view **to the desktop window**. It is not
streamed back into the headset — there is no video path to the PICO here. The
shipped stack does that through the deployment pipeline (`run_sim_loop.py`
`--enable-image-publish`, `sensor_server.py`, and the ZMQ camera topic), which is
part of the C++ deployment port and not wired up for H2 yet.
```

```{warning}
With no headset connected the service returns all-zero poses, quaternion
included. `PicoSource` rejects those rather than treating them as a real pose,
so a device that never connected — or drops out mid-session — leaves the robot
holding its default stance instead of being commanded from a stale buffer.
```

### Playing reference motions

Once you have an H2 motion library, point the runner at it:

```bash
.venv/bin/python gear_sonic/scripts/run_h2_mujoco_onnx.py \
    --reference motion --motion-file data/h2_motions/robot.pkl --viewer
```

```{warning}
The motions must be retargeted **to H2**. Bones-SEED and
`gear_sonic_deploy/reference/example/` both ship **G1** trajectories (29 DOF, 14
bodies indexed into G1's 30) and cannot drive H2. The runner checks the DOF count
and fails loudly rather than transposing joints silently.
```

Build the library from retargeted H2 CSVs with:

```bash
python gear_sonic/data_process/convert_h2_csv_to_motion_lib.py \
    --input /path/to/h2_retargeted_csvs/ --output data/h2_motions/robot \
    --fps 30 --fps_source 120 --individual --num_workers 16
```

### Verifying a change

The observation layout is the part that is easy to get subtly wrong, and a wrong
layout is obvious: swapping any two proprioception blocks makes the robot diverge
within a second, while the correct layout stays up indefinitely. If you change
observation assembly, re-run the 30 s static case as a regression check.

## Known issue: unconstrained head

The v1.1 reward set (`local_feet_acc_energy_5pt`) is entirely body-level — there
is no joint-position tracking term in it at all — so any joint that drives no
tracked body is unconstrained in position. G1 has no head, so this never
surfaced. On H2 it left `head_yaw` free, and on the step-100000 checkpoint it
settles at **+1.02 rad (58 degrees off-axis)** while every other joint stays
within 0.36 rad of default. Nothing pulls it back: `anti_shake_ang_vel`
penalises only the head's *velocity*, `joint_limit` does not fire until 1.745
rad, and once parked the joint costs no energy.

There are two fixes, and you probably want both.

**At inference (works on the checkpoint you already have).** The runner commands
the head joints directly and holds them level and forward. The head drives no
tracked body and carries no load, so overriding it is safe: measured gaze goes
from `[0.33, 0.94, -0.02]` to `[0.99, 0.16, -0.05]` with pelvis height unchanged
(1.024 -> 1.023). This is unconditional. `head_target(t)` in the runner is where
a headset's pitch and yaw go if you want the robot's head to follow the
operator.

**In training (needs a fresh run).** `sonic_h2.yaml` now lists `head_yaw_link`
among the tracked `body_names`, which constrains both head joints through the
existing body position and orientation rewards. Two caveats:

- It only takes effect on a **fresh training run**. Checkpoints trained before
  this change still drift.
- It does **not** change the exported ONNX interface — the encoders read
  joint-level and anchor terms only, so encoder 1791 / decoder 1054 are
  unaffected. The critic observation does change.

## Path B: C++/TensorRT deployment

Required for real hardware and for the shipped VR teleop stack. **This port is
incomplete** — see the status table below before starting.

### Prerequisites

Everything in [Installation (Deployment)](installation_deploy.md) applies:
CUDA Toolkit, TensorRT (10.13 on x86_64 — the exact version matters), and the
Unitree SDK2 C++ headers. Note the TensorRT archive is roughly 10 GB, so check
free space first.

```bash
df -h .
```

### Port status

| Component | Status |
|---|---|
| `policy/sonic_h2/observation_config.yaml` | **done** — encoder 1791 / decoder 1054 |
| `include/policy_parameters_h2.hpp` | **done** — gains, action scale, defaults, index maps |
| `include/robot_parameters.hpp` | todo — `G1_NUM_MOTOR` 29→31, joint enum, DDS topics |
| `src/g1_deploy_onnx_ref.cpp` | todo — observation registry dims (290→310) |
| `include/localmotion_kplanner.hpp` | todo |
| `include/control_policy.hpp`, `error_monitor.hpp` | todo |
| `src/fk.cpp` | todo — H2 link geometry, a rewrite rather than a constant swap |
| MuJoCo sim side (`run_sim_loop.py`) | todo — `instantiate_h2_robot_model`, H2 scene, wbc config |

`G1_NUM_MOTOR` is a compile-time array size (`std::array<float, 29>`), not a
config value, so pointing the existing binary at an H2 ONNX does not work: it
assembles a 1751-dim encoder input for a model that expects 1791.

The numbers in `policy_parameters_h2.hpp` are checked against
`gear_sonic/envs/manager_env/robots/h2.py` — all 31 joints of `action_scale`,
`kps`, `kds` and `default_angles` — and every index array is generated from a
breadth-first walk of `h2.xml`, which is the ordering Isaac Lab uses.
