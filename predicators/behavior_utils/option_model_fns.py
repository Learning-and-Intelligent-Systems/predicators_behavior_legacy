"""Functions that consume a plan for a BEHAVIOR robot and return an option
model for that plan."""

import logging
from typing import Callable, List

import numpy as np
import pybullet as p

from predicators.structs import State

try:
    from igibson.external.pybullet_tools.utils import get_aabb, get_aabb_extent
    from igibson.utils import sampling_utils
    from igibson import object_states
    from igibson.object_states.utils import sample_kinematics
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.objects.articulated_object import \
        URDFObject  # pylint: disable=unused-import
except (ImportError, ModuleNotFoundError) as e:
    pass


def create_navigate_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        _obj_to_nav_to: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a navigation option model function given a
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


def create_grasp_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_grasp: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a grasp option model function given a plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

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


def create_place_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a place option model function given a plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

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


def create_open_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_open: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an open option model given a dummy plan."""
    del plan

    def openObjectOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        if np.linalg.norm(
                np.array(obj_to_open.get_position()) -
                np.array(env.robots[0].get_position())) < 2:
            if hasattr(obj_to_open,
                       "states") and object_states.Open in obj_to_open.states:
                obj_to_open.states[object_states.Open].set_value(True)
            else:
                logging.info("PRIMITIVE open failed, cannot be opened")
        else:
            logging.info("PRIMITIVE open failed, too far")
        obj_to_open.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return openObjectOptionModel


def create_close_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_close: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an close option model given a dummy plan."""
    del plan

    def closeObjectOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        if np.linalg.norm(
                np.array(obj_to_close.get_position()) -
                np.array(env.robots[0].get_position())) < 2:
            if hasattr(obj_to_close,
                       "states") and object_states.Open in obj_to_close.states:
                obj_to_close.states[object_states.Open].set_value(False)
            else:
                logging.info("PRIMITIVE close failed, cannot be opened")
        else:
            logging.info("PRIMITIVE close failed, too far")
        obj_to_close.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return closeObjectOptionModel



def create_place_inside_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an placeInside option model given a dummy plan."""
    del plan

    def placeInsideObjectOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        obj_in_hand = env.scene.get_objects()[env.robots[0].parts["right_hand"].object_in_hand]
        rh_orig_grasp_postion = env.robots[0].parts["right_hand"].get_position(
        )
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()
        if obj_in_hand is not None and obj_in_hand != obj_to_place and isinstance(obj_to_place, URDFObject):
            logging.info("PRIMITIVE:attempt to place {} inside {}".format(obj_in_hand.name, obj_to_place.name))
            if np.linalg.norm(np.array(obj_to_place.get_position()) - np.array(env.robots[0].get_position())) < 2:
                if (
                    hasattr(obj_to_place, "states")
                    and object_states.Open in obj_to_place.states
                    and obj_to_place.states[object_states.Open].get_value()
                ) or (hasattr(obj_to_place, "states") and not object_states.Open in obj_to_place.states):
                    state = p.saveState()
                    # TODO fix sample_kinematics
                    result = sample_kinematics(
                        "inside",
                        obj_in_hand,
                        obj_to_place,
                        True,
                        use_ray_casting_method=True,
                        max_trials=200,
                    )
                    # # TODO Attempted to just use sample_cuboid but also doesn't work
                    # sampling_results = [[None]]

                    # while sampling_results[0][0] is None:
                    #     objA = obj_in_hand
                    #     objB = obj_to_place

                    #     params = {
                    #         "max_angle_with_z_axis": 0.17,
                    #         "bimodal_stdev_fraction": 0.4,
                    #         "bimodal_mean_fraction": 0.5,
                    #         "max_sampling_attempts": 100,
                    #         "aabb_offset": -0.01,
                    #     }
                    #     aabb = get_aabb(objA.get_body_id())
                    #     aabb_extent = get_aabb_extent(aabb)

                    #     rng = np.random.default_rng(0)
                    #     random_seed_int = rng.integers(10000000)
                    #     sampling_results = sampling_utils.sample_cuboid_on_object(
                    #         objB,
                    #         num_samples=1,
                    #         cuboid_dimensions=aabb_extent,
                    #         axis_probabilities=[0, 0, 1],
                    #         refuse_downwards=True,
                    #         random_seed_number=random_seed_int,
                    #         **params,
                    #     )

                    # import ipdb; ipdb.set_trace()
                    # #
                    if result:
                        logging.info(
                            "PRIMITIVE: place {} inside {} success".format(obj_in_hand.name, obj_to_place.name)
                        )
                        target_pos = obj_in_hand.get_position()
                        target_orn = obj_in_hand.get_orientation()
                        import ipdb; ipdb.set_trace()
                        env.robots[0].parts["right_hand"].set_position_orientation(
                            target_pos, p.getQuaternionFromEuler(target_orn))
                        env.robots[0].parts["right_hand"].force_release_obj()
                        obj_to_place.force_wakeup()
                        # this is running a zero action to step simulator
                        env.step(np.zeros(env.action_space.shape))
                        # reset the released object to zero velocity so it doesn't
                        # fly away because of residual warp speeds from teleportation!
                        p.resetBaseVelocity(
                            obj_in_hand,
                            linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0],
                        )
                        env.robots[0].parts["right_hand"].set_position_orientation(
                            rh_orig_grasp_postion, rh_orig_grasp_orn)
                        # this is running a series of zero action to step simulator
                        # to let the object fall into its place
                        for _ in range(15):
                            env.step(np.zeros(env.action_space.shape))
                    else:
                        logging.info(
                            "PRIMITIVE: place {} inside {} fail, sampling fail".format(
                                obj_in_hand.name, obj_to_place.name
                            )
                        )
                        p.removeState(state)
                else:
                    logging.info(
                        "PRIMITIVE: place {} inside {} fail, need open not open".format(
                            obj_in_hand.name, obj_to_place.name
                        )
                    )
            else:
                logging.info(
                    "PRIMITIVE: place {} inside {} fail, too far".format(obj_in_hand.name, obj_to_place.name)
                )
        else:
            logging.info(
                    "PRIMITIVE: place failed with invalid obj params."
                )

        obj_to_place.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return placeInsideObjectOptionModel