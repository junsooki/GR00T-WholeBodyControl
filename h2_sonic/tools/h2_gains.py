"""Single source of truth for H2 actuator gains, read from robots/h2.py.

Both the C++ header generator and the Python policy runner need the same
per-joint stiffness, damping and action scale that training used. They used to
keep private copies transcribed by hand. Those copies went stale when the H2
effort limits were corrected against h2.urdf -- ankle_roll 150 -> 19 N.m and
twelve others -- which would have shipped an action scale up to 7.9x too large.
The runner's copy also collapsed ankle roll and pitch into one entry, so it
could not represent their (genuinely different) limits at all.

robots/h2.py cannot simply be imported: it pulls in isaaclab, which needs a
running Isaac Sim kernel, and a build-time code generator should not require
that. So this module reads the file with `ast`. Because h2.py writes stiffness
and damping as symbolic expressions (STIFFNESS_7520_22, 2.0 * STIFFNESS_5020),
the parse yields exactly the expressions the generator emits, and the numeric
constants the runner needs.
"""

import ast
import os
import re

H2_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "gear_sonic", "envs", "manager_env", "robots", "h2.py",
)

# Longest-match-first inside each family, so "hip_pitch" is not shadowed by a
# shorter prefix and ankle/wrist roll-vs-pitch stay distinct. Order matters.
KINDS = [
    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch",
    "waist_yaw", "waist_roll", "waist_pitch", "head_pitch", "head_yaw",
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
]


def joint_kind(name):
    for kind in KINDS:
        if kind in name:
            return kind
    raise KeyError(name)


def _expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return f"{_expr(node.left)} * {_expr(node.right)}"
    raise SystemExit(f"unsupported expression in h2.py: {ast.dump(node)[:120]}")


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _resolve(node, joint):
    """Value of a scalar-or-{regex: value} actuator field, for one joint."""
    if node is None:
        return None
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and re.fullmatch(k.value, joint):
                return _expr(v)
        return None
    return _expr(node)


def _module_constants(tree):
    """Module-level float literals and simple products (ARMATURE_*, etc.)."""
    consts = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        try:
            consts[tgt.id] = eval(  # noqa: S307 - literals from a file we ship
                compile(ast.Expression(node.value), "<h2>", "eval"), {}, dict(consts)
            )
        except Exception:  # noqa: BLE001 - non-numeric assignments are expected
            pass
    return consts


def load(mujoco_joints):
    """Return (gains, constants, velocity).

    gains: {joint_kind: (family, effort_limit, stiffness_multiplier)}
    constants: module-level numeric constants from h2.py (ARMATURE_*, STIFFNESS_*, ...)
    velocity: {joint_name: velocity_limit_sim} -- per joint, not per kind, because
        h2.py gives ankle roll and pitch different limits (100.70 vs 28.61) and a
        kind-keyed table cannot represent that.
    """
    tree = ast.parse(open(H2_PY).read())
    consts = _module_constants(tree)

    groups = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ImplicitActuatorCfg"):
            continue
        names = _kwarg(node, "joint_names_expr")
        if names is None:
            continue
        groups.append(([e.value for e in names.elts],
                       _kwarg(node, "effort_limit_sim"), _kwarg(node, "stiffness"),
                       _kwarg(node, "velocity_limit_sim")))
    if not groups:
        raise SystemExit(f"no ImplicitActuatorCfg found in {H2_PY}")

    gains, vel = {}, {}
    for joint in mujoco_joints:
        for pats, eff_n, stiff_n, vel_n in groups:
            if not any(re.fullmatch(p, joint) for p in pats):
                continue
            eff, stiff = _resolve(eff_n, joint), _resolve(stiff_n, joint)
            if eff is None or stiff is None:
                continue
            m = re.fullmatch(r"(?:([0-9.]+) \* )?STIFFNESS_(\w+)", str(stiff))
            if not m:
                raise SystemExit(f"unexpected stiffness for {joint}: {stiff}")
            cur = (m.group(2), float(eff), float(m.group(1)) if m.group(1) else 1.0)
            kind = joint_kind(joint)
            if kind in gains and gains[kind] != cur:
                raise SystemExit(f"joint kind '{kind}' inconsistent: {gains[kind]} vs {cur}")
            gains[kind] = cur
            v = _resolve(vel_n, joint)
            if v is not None:
                vel[joint] = float(v)
            break
        else:
            raise SystemExit(f"no actuator group in h2.py matches joint '{joint}'")
    return gains, consts, vel
