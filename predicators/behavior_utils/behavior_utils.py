"""Utility functions for using iGibson and BEHAVIOR."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pybullet as p

from predicators.structs import Array

try:
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.robots.behavior_robot import \
        BRBody  # pylint: disable=unused-import
    from igibson.robots.robot_base import \
        BaseRobot  # pylint: disable=unused-import
except:
    pass


def get_aabb_volume(lo: Array, hi: Array) -> float:
    """Simple utility function to compute the volume of an aabb.

    lo refers to the minimum values of the bbox in the x, y and z axes,
    while hi refers to the highest values. Both lo and hi must be three-
    dimensional.
    """
    assert np.all(hi >= lo)
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def get_closest_point_on_aabb(xyz: List, lo: Array, hi: Array) -> List[float]:
    """Get the closest point on an aabb from a particular xyz coordinate."""
    assert np.all(hi >= lo)
    closest_point_on_aabb = [0.0, 0.0, 0.0]
    for i in range(3):
        # if the coordinate is between the min and max of the aabb, then
        # use that coordinate directly
        if xyz[i] < hi[i] and xyz[i] > lo[i]:
            closest_point_on_aabb[i] = xyz[i]
        else:
            if abs(xyz[i] - hi[i]) < abs(xyz[i] - lo[i]):
                closest_point_on_aabb[i] = hi[i]
            else:
                closest_point_on_aabb[i] = lo[i]
    return closest_point_on_aabb


def get_scene_body_ids(
    env: "BehaviorEnv",
    include_self: bool = False,
    include_right_hand: bool = False,
) -> List[int]:
    """Function to return a list of body_ids for all objects in the scene for
    collision checking depending on whether navigation or grasping/ placing is
    being done."""
    ids = []
    for obj in env.scene.get_objects():
        if isinstance(obj, URDFObject):
            # We want to exclude the floor since we're always floating and
            # will never practically collide with it, but if we include it
            # in collision checking, we always seem to collide.
            if obj.name != "floors":
                ids.extend(obj.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())
        ids.append(env.robots[0].parts["eye"].get_body_id())
        if not include_right_hand:
            ids.append(env.robots[0].parts["right_hand"].get_body_id())

    return ids


def detect_collision(bodyA: int, object_in_hand: Optional[int] = None) -> bool:
    """Detects collisions between bodyA in the scene (except for the object in
    the robot's hand)"""
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id in [bodyA, object_in_hand]:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot: "BaseRobot") -> bool:
    """Function to detect whether the robot is currently colliding with any
    object in the scene."""
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (detect_collision(robot.parts["body"].body_id)
            or detect_collision(robot.parts["left_hand"].body_id)
            or detect_collision(robot.parts["right_hand"].body_id,
                                object_in_hand))


def reset_and_release_hand(env: "BehaviorEnv") -> None:
    """Resets the state of the right hand."""
    env.robots[0].set_position_orientation(env.robots[0].get_position(),
                                           env.robots[0].get_orientation())
    for _ in range(50):
        env.robots[0].parts["right_hand"].set_close_fraction(0)
        env.robots[0].parts["right_hand"].trigger_fraction = 0
        p.stepSimulation()


def get_delta_low_level_base_action(robot_z: float,
                                    original_orientation: Tuple,
                                    old_xytheta: Array, new_xytheta: Array,
                                    action_space_shape: Tuple) -> Array:
    """Given a base movement plan that is a series of waypoints in world-frame
    position space, convert pairs of these points to a base movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    ret_action = np.zeros(action_space_shape, dtype=np.float32)

    # First, get the old and new position and orientation in the world
    # frame as numpy arrays
    old_pos = np.array([old_xytheta[0], old_xytheta[1], robot_z])
    old_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             old_xytheta[2]]))
    new_pos = np.array([new_xytheta[0], new_xytheta[1], robot_z])
    new_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             new_xytheta[2]]))

    # Then, simply get the delta position and orientation by multiplying the
    # inverse of the old pose by the new pose
    inverted_old_pos, inverted_old_orn_quat = p.invertTransform(
        old_pos, old_orn_quat)
    delta_pos, delta_orn_quat = p.multiplyTransforms(inverted_old_pos,
                                                     inverted_old_orn_quat,
                                                     new_pos, new_orn_quat)

    # Finally, convert the orientation back to euler angles from a quaternion
    delta_orn = p.getEulerFromQuaternion(delta_orn_quat)

    ret_action[0:3] = np.array([delta_pos[0], delta_pos[1], delta_orn[2]])

    return ret_action


def get_delta_low_level_hand_action(
    body: "BRBody",
    old_pos: Union[Sequence[float], Array],
    old_orn: Union[Sequence[float], Array],
    new_pos: Union[Sequence[float], Array],
    new_orn: Union[Sequence[float], Array],
) -> Array:
    """Given a hand movement plan that is a series of waypoints for the hand in
    position space, convert pairs of these points to a hand movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    # First, convert the supplied orientations to quaternions
    old_orn = p.getQuaternionFromEuler(old_orn)
    new_orn = p.getQuaternionFromEuler(new_orn)

    # Next, find the inverted position of the body (which we know shouldn't
    # change, since our actions move either the body or the hand, but not
    # both simultaneously)
    inverted_body_new_pos, inverted_body_new_orn = p.invertTransform(
        body.new_pos, body.new_orn)
    # Use this to compute the new pose of the hand w.r.t the body frame
    new_local_pos, new_local_orn = p.multiplyTransforms(
        inverted_body_new_pos, inverted_body_new_orn, new_pos, new_orn)

    # Next, compute the old pose of the hand w.r.t the body frame
    inverted_body_old_pos = inverted_body_new_pos
    inverted_body_old_orn = inverted_body_new_orn
    old_local_pos, old_local_orn = p.multiplyTransforms(
        inverted_body_old_pos, inverted_body_old_orn, old_pos, old_orn)

    # The delta position is simply given by the difference between these
    # positions
    delta_pos = np.array(new_local_pos) - np.array(old_local_pos)

    # Finally, compute the delta orientation
    inverted_old_local_orn_pos, inverted_old_local_orn_orn = p.invertTransform(
        [0, 0, 0], old_local_orn)
    _, delta_orn = p.multiplyTransforms(
        [0, 0, 0],
        new_local_orn,
        inverted_old_local_orn_pos,
        inverted_old_local_orn_orn,
    )

    delta_trig_frac = 0
    action = np.concatenate(
        [
            np.zeros((10), dtype=np.float32),
            np.array(delta_pos, dtype=np.float32),
            np.array(p.getEulerFromQuaternion(delta_orn), dtype=np.float32),
            np.array([delta_trig_frac], dtype=np.float32),
        ],
        axis=0,
    )

    return action
