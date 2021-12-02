"""Painting domain, which allows for two different grasps on an object
(side or top). Side grasping allows for placing into the shelf, and top
grasping allows for placing into the box. The box has a lid which may
need to be opened; this lid is NOT modeled by any of the given predicates.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
from gym.spaces import Box
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG


class PaintingEnv(BaseEnv):
    """Painting domain.
    """
    # Parameters that aren't important enough to need to clog up settings.py
    table_lb = -10.1
    table_ub = -0.2
    table_height = 0.2
    shelf_l = 2.0 # shelf length
    shelf_lb = 1.
    shelf_ub = shelf_lb + shelf_l - 0.05
    box_s = 0.8  # side length
    box_y = 0.5  # y coordinate
    box_lb = box_y - box_s/10
    box_ub = box_y + box_s/10
    env_lb = min(table_lb, shelf_lb, box_lb)
    env_ub = max(table_ub, shelf_ub, box_ub)
    obj_height = 0.13
    obj_radius = 0.03
    obj_x = 1.65
    obj_z = table_height + obj_height/2
    pick_tol = 1e-2
    color_tol = 1e-2
    wetness_tol = 0.5
    dirtiness_tol = 0.5
    open_fingers = 0.8
    top_grasp_thresh = 0.5 + 1e-2
    side_grasp_thresh = 0.5 - 1e-2
    held_tol = 0.5
    num_objs_train = [3, 4]
    num_objs_test = [5, 6]

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._obj_type = Type("obj", ["pose_x", "pose_y", "pose_z", "dirtiness",
                                      "wetness", "color", "held"])
        self._box_type = Type("box", ["color"])
        self._lid_type = Type("lid", ["is_open"])
        self._shelf_type = Type("shelf", ["color"])
        self._robot_type = Type("robot", ["gripper_rot", "fingers"])
        # Predicates
        self._InBox = Predicate(
            "InBox", [self._obj_type, self._box_type], self._InBox_holds)
        self._InShelf = Predicate(
            "InShelf", [self._obj_type, self._shelf_type], self._InShelf_holds)
        self._IsBoxColor = Predicate(
            "IsBoxColor", [self._obj_type, self._box_type],
            self._IsBoxColor_holds)
        self._IsShelfColor = Predicate(
            "IsShelfColor", [self._obj_type, self._shelf_type],
            self._IsShelfColor_holds)
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds)
        self._OnTable = Predicate(
            "OnTable", [self._obj_type], self._OnTable_holds)
        self._HoldingTop = Predicate(
            "HoldingTop", [self._obj_type, self._robot_type],
            self._HoldingTop_holds)
        self._HoldingSide = Predicate(
            "HoldingSide", [self._obj_type, self._robot_type],
            self._HoldingSide_holds)
        self._Holding = Predicate(
            "Holding", [self._obj_type], self._Holding_holds)
        self._IsWet = Predicate(
            "IsWet", [self._obj_type], self._IsWet_holds)
        self._IsDry = Predicate(
            "IsDry", [self._obj_type], self._IsDry_holds)
        self._IsDirty = Predicate(
            "IsDirty", [self._obj_type], self._IsDirty_holds)
        self._IsClean = Predicate(
            "IsClean", [self._obj_type], self._IsClean_holds)
        # Options
        self._Pick = ParameterizedOption(
            # variables: [robot, object to pick]
            # params: [delta x, delta y, delta z, grasp rotation]
            "Pick", types=[self._robot_type, self._obj_type],
            params_space=Box(
                np.array([-1.0, -1.0, -1.0, -0.01], dtype=np.float32),
                np.array([1.0, 1.0, 1.0, 1.01], dtype=np.float32)),
            _policy=self._Pick_policy,
            _initiable=self._handempty_initiable,
            _terminal=self._onestep_terminal)
        self._Wash = ParameterizedOption(
            # variables: [robot]
            # params: [water level]
            "Wash", types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1,)),
            _policy=self._Wash_policy,
            _initiable=self._holding_initiable,
            _terminal=self._onestep_terminal)
        self._Dry = ParameterizedOption(
            # variables: [robot]
            # params: [heat level]
            "Dry", types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1,)),
            _policy=self._Dry_policy,
            _initiable=self._holding_initiable,
            _terminal=self._onestep_terminal)
        self._Paint = ParameterizedOption(
            # variables: [robot]
            # params: [new color]
            "Paint", types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1,)),
            _policy=self._Paint_policy,
            _initiable=self._holding_initiable,
            _terminal=self._onestep_terminal)
        self._Place = ParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y, absolute z]
            "Place", types=[self._robot_type],
            params_space=Box(
                np.array([self.obj_x - 1e-2, self.env_lb, self.obj_z - 1e-2],
                         dtype=np.float32),
                np.array([self.obj_x + 1e-2, self.env_ub, self.obj_z + 1e-2],
                         dtype=np.float32)),
            _policy=self._Place_policy,
            _initiable=self._holding_initiable,
            _terminal=self._onestep_terminal)
        self._OpenLid = ParameterizedOption(
            # variables: [robot, lid]
            # params: []
            "OpenLid", types=[self._robot_type, self._lid_type],
            params_space=Box(-0.01, 1.01, (0,)),  # no parameters
            _policy=self._OpenLid_policy,
            _initiable=self._handempty_initiable,
            _terminal=self._onestep_terminal)
        # Objects
        self._box = Object("receptacle_box", self._box_type)
        self._lid = Object("box_lid", self._lid_type)
        self._shelf = Object("receptacle_shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        # Infer which transition function to follow
        wash_affinity = 0 if arr[5] > 0.5 else abs(arr[5] - 0.5)
        dry_affinity = 0 if arr[6] > 0.5 else abs(arr[6] - 0.5)
        paint_affinity = min(
            abs(arr[7] - state.get(self._box, "color")),
            abs(arr[7] - state.get(self._shelf, "color")))
        affinities = [
            (abs(1 - arr[4]), self._transition_pick_or_openlid),
            (wash_affinity, self._transition_wash),
            (dry_affinity, self._transition_dry),
            (paint_affinity, self._transition_paint),
            (abs(-1 - arr[4]), self._transition_place),
        ]
        _, transition_fn = min(affinities, key=lambda item: item[0])
        return transition_fn(state, action)

    def _transition_pick_or_openlid(self, state: State, action: Action
                                    ) -> State:
        x, y, z, rot = action.arr[:4]
        next_state = state.copy()
        # Open lid
        if self.box_lb < y < self.box_ub:
            next_state.set(self._lid, "is_open", 1.0)
            return next_state
        held_obj = self._get_held_object(state)
        # Cannot pick if already holding something
        if held_obj is not None:
            return next_state
        # Check if some object is close enough to (x, y, z)
        target_obj = self._get_object_at_xyz(state, x, y, z)
        if target_obj is None:
            return next_state
        # Execute pick
        next_state.set(self._robot, "gripper_rot", rot)
        next_state.set(self._robot, "fingers", 0.0)
        next_state.set(target_obj, "held", 1.0)
        return next_state

    def _transition_wash(self, state: State, action: Action) -> State:
        target_wetness = action.arr[5]
        next_state = state.copy()
        held_obj = self._get_held_object(state)
        # Can only wash if holding obj
        if held_obj is None:
            return next_state
        # Execute wash
        cur_dirtiness = state.get(held_obj, "dirtiness")
        next_dirtiness = max(cur_dirtiness - target_wetness, 0.0)
        next_state.set(held_obj, "wetness", target_wetness)
        next_state.set(held_obj, "dirtiness", next_dirtiness)
        return next_state

    def _transition_dry(self, state: State, action: Action) -> State:
        target_wetness = max(1.0 - action.arr[6], 0.0)
        next_state = state.copy()
        held_obj = self._get_held_object(state)
        # Can only dry if holding obj
        if held_obj is None:
            return next_state
        # Execute dry
        next_state.set(held_obj, "wetness", target_wetness)
        return next_state

    def _transition_paint(self, state: State, action: Action) -> State:
        color = action.arr[7]
        next_state = state.copy()
        # Can only paint if holding obj
        held_obj = self._get_held_object(state)
        if held_obj is None:
            return next_state
        # Can only paint if dry and clean
        if state.get(held_obj, "dirtiness") > self.dirtiness_tol or \
           state.get(held_obj, "wetness") > self.wetness_tol:
            return next_state
        # Execute paint
        next_state.set(held_obj, "color", color)
        return next_state

    def _transition_place(self, state: State, action: Action) -> State:
        # Action args are target pose for held obj
        x, y, z = action.arr[:3]
        next_state = state.copy()
        # Can only place if holding obj
        held_obj = self._get_held_object(state)
        if held_obj is None:
            return next_state
        # Detect shelf vs box place
        if self.shelf_lb < y < self.shelf_ub:
            shelf_or_box = "shelf"
        elif self.box_lb < y < self.box_ub:
            shelf_or_box = "box"
        else:
            # Cannot place outside of shelf or box
            return next_state
        if shelf_or_box == "box" and state.get(self._lid, "is_open") < 0.5:
            # Cannot place in box if lid is not open
            raise EnvironmentFailure("box lid is closed", {self._lid})
        # Detect top grasp vs side grasp
        rot = state.get(self._robot, "gripper_rot")
        if rot > self.top_grasp_thresh:
            top_or_side = "top"
        elif rot < self.side_grasp_thresh:
            top_or_side = "side"
        else:
            # Cannot place in either shelf or box, bad gripper_rot
            return next_state
        # Can only place in shelf if side grasping, box if top grasping
        if (shelf_or_box, top_or_side) not in [("shelf", "side"),
                                               ("box", "top")]:
            return next_state
        # Execute place
        next_state.set(self._robot, "gripper_rot", 0.5)
        next_state.set(self._robot, "fingers", 1.0)
        next_state.set(held_obj, "pose_x", x)
        next_state.set(held_obj, "pose_y", y)
        next_state.set(held_obj, "pose_z", z)
        next_state.set(held_obj, "held", 0.0)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               num_objs_lst=self.num_objs_train,
                               rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               num_objs_lst=self.num_objs_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InBox, self._InShelf, self._IsBoxColor,
                self._IsShelfColor, self._GripperOpen, self._OnTable,
                self._HoldingTop, self._HoldingSide, self._Holding,
                self._IsWet, self._IsDry, self._IsDirty, self._IsClean}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBox, self._InShelf, self._IsBoxColor,
                self._IsShelfColor}

    @property
    def types(self) -> Set[Type]:
        return {self._obj_type, self._box_type, self._lid_type,
                self._shelf_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Wash, self._Dry, self._Paint,
                self._Place, self._OpenLid}

    @property
    def action_space(self) -> Box:
        # Actions are 8-dimensional vectors:
        # [x, y, z, rot, pickplace, water level, heat level, color]
        # Note that pickplace is 1 for pick, -1 for place, and 0 otherwise,
        # while rot, water level, heat level, and color are in [0, 1].
        lowers = np.array([self.obj_x - 1e-2, self.env_lb,
                           self.obj_z - 1e-2, 0.0, -1.0, 0.0, 0.0, 0.0],
                          dtype=np.float32)
        uppers = np.array([self.obj_x + 1e-2, self.env_ub,
                           self.obj_z + 1e-2, 1.0, 1.0, 1.0, 1.0, 1.0],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> List[Image]:
        raise NotImplementedError

    def _get_tasks(self, num_tasks: int, num_objs_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            num_objs = num_objs_lst[i % len(num_objs_lst)]
            data = {}
            # Initialize robot
            data[self._robot] = np.array([0.5, 1.0])  # fingers start off open
            # Sample distinct colors for shelf and box
            color1 = rng.uniform(0.2, 0.4)
            color2 = rng.uniform(0.6, 1.0)
            if rng.choice(2):
                box_color, shelf_color = color1, color2
            else:
                shelf_color, box_color = color1, color2
            # Create box, lid, and shelf objects
            lid_is_open = int(rng.uniform() > 0.7)
            data[self._box] = np.array([box_color], dtype=np.float32)
            data[self._lid] = np.array([lid_is_open], dtype=np.float32)
            data[self._shelf] = np.array([shelf_color], dtype=np.float32)
            # Create moveable objects
            obj_poses: List[Tuple[float, float, float]] = []
            goal = set()
            for j in range(num_objs):
                obj = Object(f"obj{j}", self._obj_type)
                pose = self._sample_initial_object_pose(obj_poses, rng)
                obj_poses.append(pose)
                # Start out wet and clean, dry and dirty, or dry and clean
                choice = rng.choice(3)
                if choice == 0:
                    wetness = 0.0
                    dirtiness = rng.uniform(0.5, 1.)
                elif choice == 1:
                    wetness = rng.uniform(0.5, 1.)
                    dirtiness = 0.0
                else:
                    wetness = 0.0
                    dirtiness = 0.0
                color = 0.0
                held = 0.0
                data[obj] = np.array([pose[0], pose[1], pose[2], dirtiness,
                                      wetness, color, held], dtype=np.float32)
                if j == num_objs-1:  # last object should go into the box
                    goal.add(GroundAtom(self._InBox, [obj, self._box]))
                    goal.add(GroundAtom(self._IsBoxColor, [obj, self._box]))
                else:
                    goal.add(GroundAtom(self._InShelf, [obj, self._shelf]))
                    goal.add(GroundAtom(self._IsShelfColor, [obj, self._shelf]))
            tasks.append(Task(State(data), goal))
        return tasks

    def _sample_initial_object_pose(
            self, existing_poses: List[Tuple[float, float, float]],
            rng: np.random.Generator) -> Tuple[float, float, float]:
        existing_ys = [p[1] for p in existing_poses]
        while True:
            this_y = rng.uniform(self.table_lb, self.table_ub)
            if all(abs(this_y-other_y) > 3.5*self.obj_radius
                   for other_y in existing_ys):
                return (self.obj_x, this_y, self.table_height+self.obj_height/2)

    def _Pick_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        _, obj = objects
        obj_x = state.get(obj, "pose_x")
        obj_y = state.get(obj, "pose_y")
        obj_z = state.get(obj, "pose_z")
        dx, dy, dz, rot = params
        arr = np.array([obj_x + dx, obj_y + dy, obj_z + dz, rot,
                        1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # The addition of dx, dy, and dz could cause the action to go
        # out of bounds, so we clip it back into the bounds for safety.
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _Wash_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        water_level, = params
        water_level = min(max(water_level, 0.0), 1.0)
        arr = np.array([self.obj_x, self.table_lb, self.obj_z,
                        0.0, 0.0, water_level, 0.0, 0.0],
                       dtype=np.float32)
        return Action(arr)

    def _Dry_policy(self, state: State, memory: Dict,
                    objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        heat_level, = params
        heat_level = min(max(heat_level, 0.0), 1.0)
        arr = np.array([self.obj_x, self.table_lb, self.obj_z,
                        0.0, 0.0, 0.0, heat_level, 0.0],
                       dtype=np.float32)
        return Action(arr)

    def _Paint_policy(self, state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        new_color, = params
        new_color = min(max(new_color, 0.0), 1.0)
        arr = np.array([self.obj_x, self.table_lb, self.obj_z,
                        0.0, 0.0, 0.0, 0.0, new_color],
                       dtype=np.float32)
        return Action(arr)

    @staticmethod
    def _Place_policy(state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        x, y, z = params
        arr = np.array([x, y, z, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return Action(arr)

    def _holding_initiable(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        # An initiation function for an option that requires holding an object.
        del memory, objects, params  # unused
        return self._get_held_object(state) is not None

    def _handempty_initiable(self, state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
        # An initiation function for an option that requires holding nothing.
        del memory, objects, params  # unused
        return self._get_held_object(state) is None

    @staticmethod
    def _onestep_terminal(state: State, memory: Dict, objects: Sequence[Object],
                          params: Array) -> bool:
         # A termination function for an option that only lasts 1 timestep.
        del state, memory, objects, params  # unused
        return True  # always 1 timestep

    def _OpenLid_policy(self, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects, params  # unused
        arr = np.array([self.obj_x, (self.box_lb + self.box_ub) / 2, self.obj_z,
                        0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return Action(arr)

    def _InBox_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in box
        if state.get(obj, "held") > 0.5:
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.box_lb < obj_y < self.box_ub

    def _InShelf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in shelf
        if state.get(obj, "held") > 0.5:
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.shelf_lb < obj_y < self.shelf_ub

    def _IsBoxColor_holds(self, state: State, objects: Sequence[Object]
                          ) -> bool:
        obj, box = objects
        return abs(state.get(obj, "color") -
                   state.get(box, "color")) < self.color_tol

    def _IsShelfColor_holds(self, state: State, objects: Sequence[Object]
                            ) -> bool:
        obj, shelf = objects
        return abs(state.get(obj, "color") -
                   state.get(shelf, "color")) < self.color_tol

    def _GripperOpen_holds(self, state: State, objects: Sequence[Object]
                           ) -> bool:
        robot, = objects
        fingers = state.get(robot, "fingers")
        return fingers >= self.open_fingers

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_y = state.get(obj, "pose_y")
        return self.table_lb < obj_y < self.table_ub

    def _HoldingTop_holds(self, state: State, objects: Sequence[Object]
                          ) -> bool:
        obj, robot = objects
        rot = state.get(robot, "gripper_rot")
        if rot < self.top_grasp_thresh:
            return False
        return self._Holding_holds(state, [obj])

    def _HoldingSide_holds(self, state: State, objects: Sequence[Object]
                           ) -> bool:
        obj, robot = objects
        rot = state.get(robot, "gripper_rot")
        if rot > self.side_grasp_thresh:
            return False
        return self._Holding_holds(state, [obj])

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._get_held_object(state) == obj

    def _IsWet_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "wetness") > self.wetness_tol

    def _IsDry_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsWet_holds(state, [obj])

    def _IsDirty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "dirtiness") > self.dirtiness_tol

    def _IsClean_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsDirty_holds(state, [obj])

    def _get_held_object(self, state: State) -> Optional[Object]:
        for obj in state:
            if obj.type != self._obj_type:
                continue
            if state.get(obj, "held") >= self.held_tol:
                return obj
        return None

    def _get_object_at_xyz(self, state: State, x: float, y: float, z: float
                           ) -> Optional[Object]:
        target_obj = None
        for obj in state:
            if obj.type != self._obj_type:
                continue
            if np.allclose([x, y, z], [state.get(obj, "pose_x"),
                                       state.get(obj, "pose_y"),
                                       state.get(obj, "pose_z")],
                           atol=self.pick_tol):
                assert target_obj is None
                target_obj = obj
        return target_obj