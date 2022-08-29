"""Hardcoded options for BehaviorEnv."""
# pylint: disable=import-error

import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
from numpy.random._generator import Generator

from predicators.behavior_utils.behavior_utils import get_aabb_volume, \
    get_closest_point_on_aabb
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


def create_navigate_policy(
    plan: List[List[float]], original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 3-element lists each containing a series of (x, y, rot)
    waypoints for the robot to pass through."""

    def navigateToOptionPolicy(_state: State,
                               env: "BehaviorEnv") -> Tuple[Array, bool]:
        atol_xy = 1e-2
        atol_theta = 1e-3
        atol_vel = 1e-4

        # 1. Get current position and orientation
        current_pos = list(env.robots[0].get_position()[0:2])
        current_orn = p.getEulerFromQuaternion(
            env.robots[0].get_orientation())[2]

        expected_pos = np.array(plan[0][0:2])
        expected_orn = np.array(plan[0][2])

        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xy) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: navigation policy completed "
                             "execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]),
                np.array(current_pos + [current_orn]), np.array(plan[0]),
                env.action_space.shape)

            # But if the corrective action is 0, take the next action
            if np.allclose(low_level_action,
                           np.zeros((env.action_space.shape[0], 1)),
                           atol=atol_vel):
                low_level_action = get_delta_low_level_base_action(
                    env.robots[0].get_position()[2],
                    tuple(original_orientation[0:2]),
                    np.array(current_pos + [current_orn]), np.array(plan[1]),
                    env.action_space.shape)
                plan.pop(0)

            return low_level_action, False

        if (len(plan) == 1
            ):  # In this case, we're at the final position we wanted
            # to reach
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: navigation policy completed execution!")

        else:
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]), np.array(plan[0]),
                np.array(plan[1]), env.action_space.shape)
            done_bit = False

        plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return navigateToOptionPolicy


def create_navigate_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        _obj_to_nav_to: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a navigation option model function given an RRT
    plan, which is a list of 3-element lists each containing a series of (x, y,
    rot) waypoints for the robot to pass through."""

    def navigateToOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        robot_z = env.robots[0].get_position()[2]
        target_pos = np.array([plan[-1][0], plan[-1][1], robot_z])
        robot_orn = p.getEulerFromQuaternion(env.robots[0].get_orientation())
        target_orn = p.getQuaternionFromEuler(
            np.array([robot_orn[0], robot_orn[1], plan[-1][2]]))
        env.robots[0].set_position_orientation(target_pos, target_orn)
        # this is running a zero action to step simulator so
        # the environment updates to the correct final position
        env.step(np.zeros(env.action_space.shape))

    return navigateToOptionModel


# Sampler for grasp continuous params
def grasp_obj_param_sampler(rng: Generator) -> Array:
    """Sampler for grasp option."""
    x_offset = (rng.random() * 0.4) - 0.2
    y_offset = (rng.random() * 0.4) - 0.2
    z_offset = rng.random() * 0.2
    return np.array([x_offset, y_offset, z_offset])


def create_grasp_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""
    # Set up two booleans to be used as 'memory', as well as
    # a 'reversed' plan to be used for our option that's
    # defined below. Note that the reversed plan makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_closing_gripper = False

    def graspObjectOptionPolicy(_state: State,
                                env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_closing_gripper
        done_bit = False

        atol_xyz = 1e-4
        atol_theta = 0.1
        atol_vel = 5e-3

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_closing_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])
            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True
            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_closing_gripper):
            # Close the gripper to see if you've gotten the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = 1.0
            tried_closing_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: grasp policy completed execution!")

        else:
            # Grasping Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return graspObjectOptionPolicy


def create_grasp_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_grasp: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a grasp option model function given an RRT
    plan, which is a list of 6-element lists containing a series of (x, y, z,
    roll, pitch, yaw) waypoints for the hand to pass through."""

    # NOTE: -1 because there are 25 timesteps that we move along the vector
    # between the hand the object for until finally grasping, and we want
    # just the final orientation.
    hand_i = -1
    rh_final_grasp_postion = plan[hand_i][0:3]
    rh_final_grasp_orn = plan[hand_i][3:6]

    def graspObjectOptionModel(_state: State, env: "BehaviorEnv") -> None:
        nonlocal hand_i
        rh_orig_grasp_postion = env.robots[0].parts["right_hand"].get_position(
        )
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()

        # 1 Teleport Hand to Grasp offset location
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_final_grasp_postion,
            p.getQuaternionFromEuler(rh_final_grasp_orn))

        # 3. Close hand and simulate grasp
        a = np.zeros(env.action_space.shape, dtype=float)
        a[16] = 1.0
        assisted_grasp_action = np.zeros(28, dtype=float)
        assisted_grasp_action[26] = 1.0
        if isinstance(obj_to_grasp.body_id, List):
            grasp_obj_body_id = obj_to_grasp.body_id[0]
        else:
            grasp_obj_body_id = obj_to_grasp.body_id
        # 3.1 Call code that does assisted grasping
        # bypass_force_check is basically a hack we should
        # turn it off for the final system and use a real grasp
        # sampler
        if env.robots[0].parts["right_hand"].object_in_hand is None:
            env.robots[0].parts["right_hand"].trigger_fraction = 0
        env.robots[0].parts["right_hand"].handle_assisted_grasping(
            assisted_grasp_action,
            override_ag_data=(grasp_obj_body_id, -1),
            bypass_force_check=True)
        # 3.2 step the environment a few timesteps to complete grasp
        for _ in range(5):
            env.step(a)

        # 4 Move Hand to Original Location
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_orig_grasp_postion, rh_orig_grasp_orn)
        if env.robots[0].parts["right_hand"].object_in_hand is not None:
            # NOTE: This below line is necessary to update the visualizer.
            # Also, it only works for URDF objects (but if the object is
            # not a URDF object, grasping should have failed)
            obj_to_grasp.force_wakeup()
        # Step a zero-action in the environment to update the visuals of the
        # environment.
        env.step(np.zeros(env.action_space.shape))

    return graspObjectOptionModel


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


def create_place_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

    # Note that the reversed plan code below makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_opening_gripper = False

    def placeOntopObjectOptionPolicy(_state: State,
                                     env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_opening_gripper

        done_bit = False
        atol_xyz = 0.1
        atol_theta = 0.1
        atol_vel = 2.5

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_opening_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])

            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True

            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_opening_gripper):
            # Open the gripper to see if you've released the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = -1.0
            tried_opening_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: place policy completed execution!")

        else:
            # Placing Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return placeOntopObjectOptionPolicy


def create_place_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a place option model function given an RRT
    plan, which is a list of 6-element lists containing a series of (x, y, z,
    roll, pitch, yaw) waypoints for the hand to pass through."""

    def placeOntopObjectOptionModel(_init_state: State,
                                    env: "BehaviorEnv") -> None:
        released_obj_bid = env.robots[0].parts["right_hand"].object_in_hand
        rh_orig_grasp_postion = env.robots[0].parts["right_hand"].get_position(
        )
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()
        target_pos = plan[-1][0:3]
        target_orn = plan[-1][3:6]
        env.robots[0].parts["right_hand"].set_position_orientation(
            target_pos, p.getQuaternionFromEuler(target_orn))
        env.robots[0].parts["right_hand"].force_release_obj()
        obj_to_place.force_wakeup()
        # this is running a zero action to step simulator
        env.step(np.zeros(env.action_space.shape))
        # reset the released object to zero velocity so it doesn't
        # fly away because of residual warp speeds from teleportation!
        p.resetBaseVelocity(
            released_obj_bid,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
        )
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_orig_grasp_postion, rh_orig_grasp_orn)
        # this is running a series of zero action to step simulator
        # to let the object fall into its place
        for _ in range(15):
            env.step(np.zeros(env.action_space.shape))

    return placeOntopObjectOptionModel
