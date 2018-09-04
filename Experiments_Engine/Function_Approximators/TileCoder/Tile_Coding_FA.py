from pylab import random, asarray
import numpy as np

from Experiments_Engine.Objects_Bases import FunctionApproximatorBase
from Experiments_Engine.Function_Approximators.TileCoder import IHT, tiles
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class TileCoderFA(FunctionApproximatorBase):

    def __init__(self, config=None):
        super().__init__()
        assert isinstance(config, Config)
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_tilings             int             32                  Number of tilings
        tiling_side_length      int             8                   The length of the tiling side
        num_actions             int             3                   Number of actions
        num_dims                int             2                   Number of dimensions
        alpha                   float           0.1                 Learning rate
        state_space_range       np.array        [1,1]               The range of the state space
        """
        self.num_tilings = check_attribute_else_default(config, 'num_tilings', 32)
        self.tiling_side_length = check_attribute_else_default(config, 'tiling_side_length', 8)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.num_dims = check_attribute_else_default(config, 'num_dims', 2)
        self.alpha = check_attribute_else_default(config, 'alpha', 0.1)
        self.state_space_range = check_attribute_else_default(config, 'state_space_range', np.ones(self.num_dims))

        self.scale_factor = self.tiling_side_length * (1 / self.state_space_range)
        self.tiles_per_tiling = self.tiling_side_length ** self.num_dims
        self.num_tiles = (self.num_tilings * self.tiles_per_tiling)
        self.theta = 0.001 * random(self.num_tiles * self.num_actions)
        self.iht = IHT(self.num_tiles)

    """ Updates the value of the parameters corresponding to the state and action """
    def update(self, state, action, nstep_return):
        current_estimate = self.get_value(state, action)
        value = nstep_return - current_estimate
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)

        tile_indices = asarray(
            tiles(self.iht, self.num_tilings, scaled_state),
            dtype=int) + (action * self.num_tiles)
        self.theta[tile_indices] += self.alpha * value

    """ Return the value of a specific state-action pair """
    def get_value(self, state, action):
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)

        tile_indices = asarray(
            tiles(self.iht, self.num_tilings, scaled_state),
            dtype=int) + (action * self.num_tiles)

        return sum(self.theta[tile_indices])

    """ Returns all the action values of the current state """
    def get_next_states_values(self, state):
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)

        values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            tile_indices = asarray(
                tiles(self.iht, self.num_tilings, scaled_state),
                dtype=int) + (action * self.num_tiles)
            values[action] = np.sum(self.theta[tile_indices])
        return values
