"""Robot model instantiation helpers, and a robot-type dispatch over them."""

from .g1 import instantiate_g1_robot_model
from .h2 import instantiate_h2_robot_model

_ROBOT_MODEL_FACTORIES = {
    "g1": instantiate_g1_robot_model,
    "g1_29dof": instantiate_g1_robot_model,
    "h2": instantiate_h2_robot_model,
    "h2_31dof": instantiate_h2_robot_model,
}


def instantiate_robot_model(robot_type: str = "g1", **kwargs):
    """Instantiate the RobotModel for a robot type.

    Defaults to G1 so existing call sites are unchanged. Keyword arguments are
    forwarded to the per-robot factory (waist_location, high_elbow_pose).
    """
    key = (robot_type or "g1").lower()
    if key not in _ROBOT_MODEL_FACTORIES:
        raise ValueError(
            f"No robot model for robot type {robot_type!r}. "
            f"Known: {sorted(_ROBOT_MODEL_FACTORIES)}"
        )
    return _ROBOT_MODEL_FACTORIES[key](**kwargs)


__all__ = [
    "instantiate_g1_robot_model",
    "instantiate_h2_robot_model",
    "instantiate_robot_model",
]
