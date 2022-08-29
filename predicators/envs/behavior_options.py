"""Hardcoded options for BehaviorEnv."""
# pylint: disable=import-error

import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random._generator import Generator

from predicators.structs import Array, State

try:
    import pybullet as p
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS, \
        get_aabb, get_aabb_extent
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.utils import sampling_utils
    from igibson.utils.behavior_robot_planning_utils import \
        plan_base_motion_br, plan_hand_motion_br

except (ImportError, ModuleNotFoundError) as e:
    pass


def navigate_to_param_sampler(rng: Generator,
                              objects: Sequence["URDFObject"]) -> Array:
    """Sampler for navigateTo option."""
    assert len(objects) == 1
    # The navigation nsrts are designed such that this is true (the target
    # obj is always last in the params list).
    obj_to_sample_near = objects[0]
    closeness_limit = 0.75
    nearness_limit = 0.5
    distance = nearness_limit + (
        (closeness_limit - nearness_limit) * rng.random())
    yaw = rng.random() * (2 * np.pi) - np.pi
    x = distance * np.cos(yaw)
    y = distance * np.sin(yaw)

    # The below while loop avoids sampling values that are inside
    # the bounding box of the object and therefore will
    # certainly be in collision with the object if the robot
    # tries to move there.
    while (abs(x) <= obj_to_sample_near.bounding_box[0]
           and abs(y) <= obj_to_sample_near.bounding_box[1]):
        distance = closeness_limit * rng.random()
        yaw = rng.random() * (2 * np.pi) - np.pi
        x = distance * np.cos(yaw)
        y = distance * np.sin(yaw)

    return np.array([x, y])


# Sampler for grasp continuous params
def grasp_obj_param_sampler(rng: Generator) -> Array:
    """Sampler for grasp option."""
    x_offset = (rng.random() * 0.4) - 0.2
    y_offset = (rng.random() * 0.4) - 0.2
    z_offset = rng.random() * 0.2
    return np.array([x_offset, y_offset, z_offset])


def place_ontop_obj_pos_sampler(
    obj: Union["URDFObject", "RoomFloor"],
    rng: Optional[Generator] = None,
) -> Array:
    """Sampler for placeOnTop option."""
    if rng is None:
        rng = np.random.default_rng(23)
    # objA is the object the robot is currently holding, and objB
    # is the surface that it must place onto.
    # The BEHAVIOR NSRT's are designed such that objA is the 0th
    # argument, and objB is the last.
    objA = obj[0]
    objB = obj[-1]

    params = {
        "max_angle_with_z_axis": 0.17,
        "bimodal_stdev_fraction": 1e-6,
        "bimodal_mean_fraction": 1.0,
        "max_sampling_attempts": 50,
        "aabb_offset": 0.01,
    }
    aabb = get_aabb(objA.get_body_id())
    aabb_extent = get_aabb_extent(aabb)

    random_seed_int = rng.integers(10000000)
    sampling_results = sampling_utils.sample_cuboid_on_object(
        objB,
        num_samples=1,
        cuboid_dimensions=aabb_extent,
        axis_probabilities=[0, 0, 1],
        refuse_downwards=True,
        random_seed_number=random_seed_int,
        **params,
    )

    if sampling_results[0] is None or sampling_results[0][0] is None:
        # If sampling fails, returns a random set of params
        return np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.3, 1.0)
        ])

    rnd_params = np.subtract(sampling_results[0][0], objB.get_position())
    return rnd_params
