" Project Packages "
from Experiments_Engine.Objects_Bases import EnvironmentBase, FunctionApproximatorBase, PolicyBase

" Math packages "
from pylab import random, cos
import numpy as np

from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_dict_else_default, check_attribute_else_default


class MountainCliff(EnvironmentBase):
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float32
    Reward = -1 at every step and -100 when the position is < -1.2

    Summary Name: steps_per_episode
    """

    def __init__(self, config=None, summary=None):

        super().__init__()
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             5000            The max number of actions executed before forcing
                                                                    termination
        save_summary                bool            False           Whether to save a summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', 5000)
        self.save_summary = check_attribute_else_default(config, 'save_summary', False)
        self.summary = summary
        if self.save_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        " Inner state of the environment "
        self.step_count = 0
        self.current_state = self.reset()
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.high = np.array([0.5, 0.07], dtype=np.float32)
        self.low = np.array([-1.2, -0.07], dtype=np.float32)
        self.action_dictionary = {0: -1,    # accelerate backwards
                                   1: 0,    # coast
                                   2: 1}    # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        position = -0.6 + random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float32)
        self.step_count = 0
        return self.current_state

    " Update environment "
    def update(self, A):
        self.step_count += 1

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[A]
        reward = -1.0
        terminate = False

        if self.step_count >= self.max_actions:
            terminate = True

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -0.6 + random() * 0.2
            reward = -100.0
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        if terminate:
            if self.save_summary:
                self.summary['steps_per_episode'].append(self.step_count)
            self.step_count = 0

        self.current_state = np.array((position, velocity), dtype=np.float64)

        return self.current_state, reward, terminate

    " Getters "
    def get_num_actions(self):
        return 3

    def get_current_state(self):
        return self.current_state

    def get_observation_dtype(self):
        return self.current_state.dtype

    def get_state_for_er_buffer(self):
        return self.current_state

    " Utilities "
    def get_surface(self, fa=FunctionApproximatorBase(), granularity=0.01, tpolicy=PolicyBase()):
        # the Granularity defines how many slices to split each dimension, e.g. 0.01 = 100 slices
        position_shift = (self.high[0] - self.low[0]) * granularity
        velocity_shift = (self.high[1] - self.low[1]) * granularity

        current_position = self.low[0]
        current_velocity = self.low[1]

        surface = []
        surface_by_action = [[] for _ in range(self.get_num_actions())]
        surface_x_coordinates = []
        surface_y_coordinates = []

        while current_position < (self.high[0] + position_shift):
            surface_slice = []
            surface_by_action_slice = [[] for _ in range(self.get_num_actions())]
            surface_slice_x_coord = []
            surface_slice_y_coord = []

            while current_velocity < (self.high[1] + velocity_shift):
                current_state = np.array((current_position, current_velocity), dtype=np.float64)

                q_values = np.array(fa.get_next_states_values(current_state))
                for i in range(self.get_num_actions()):
                    surface_by_action_slice[i].append(q_values[i])
                p_values = tpolicy.probability_of_action(q_values, all_actions=True)
                state_value = np.sum(q_values * p_values)

                surface_slice.append(state_value)
                surface_slice_x_coord.append(current_position)
                surface_slice_y_coord.append(current_velocity)
                current_velocity += velocity_shift

            surface.append(surface_slice)
            for i in range(self.get_num_actions()):
                surface_by_action[i].append(surface_by_action_slice[i])
            surface_x_coordinates.append(surface_slice_x_coord)
            surface_y_coordinates.append(surface_slice_y_coord)

            current_velocity = self.low[1]
            current_position += position_shift

        surface = np.array(surface)
        surface_by_action = np.array(surface_by_action)
        surface_x_coordinates = np.array(surface_x_coordinates)
        surface_y_coordinates = np.array(surface_y_coordinates)
        return surface, surface_x_coordinates, surface_y_coordinates, surface_by_action
