import numpy as np
from Experiments_Engine.Util import check_attribute_else_default, check_dict_else_default
from Experiments_Engine.config import Config


class ReplayBufferAgent:

    def __init__(self, environment, function_approximator, behaviour_policy, er_buffer, config=None, summary=None):
        super().__init__()
        """
        Summary Name: return_per_episode
        """
        self.config = config or Config()
        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions             int             3                   number_of_actions
        initial_rand_steps      int             0                   number of random steps before training starts
        rand_steps_count        int             0                   number of random steps taken so far
        save_summary            bool            False               Save the summary of the agent (return per episode)
        """
        self.num_actions = check_attribute_else_default(self.config, 'num_actions', 3)
        self.initial_rand_steps = check_attribute_else_default(self.config, 'initial_rand_steps', 0)
        check_attribute_else_default(self.config, 'rand_steps_count', 0)
        self.save_summary = check_attribute_else_default(self.config, 'save_summary', False)

        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'return_per_episode', [])

        " Other Parameters "
        # Behaviour and Target Policies
        self.bpolicy = behaviour_policy

        # Experience Replay Buffer: used for storing and retrieving observations. Mainly for Deep RL
        self.er_buffer = er_buffer

        # Function Approximator: used to approximate the Q-Values
        self.fa = function_approximator

        # Environment that the agent is interacting with
        self.env = environment

    def train(self):
        # Record Keeping
        episode_reward_sum = 0
        # Current State, Action, and Q_values
        S = self.env.get_current_state()
        if self.config.rand_steps_count >= self.initial_rand_steps:
            q_values = self.fa.get_next_states_values(S)

            A = self.bpolicy.choose_action(q_values)
            self.bpolicy.anneal()
        else:
            A = np.random.randint(self.num_actions)
            self.config.rand_steps_count += 1
        # Storing in the experience replay buffer
        observation = {"reward": 0, "action": A, "state": self.env.get_state_for_er_buffer(), "terminate": False}
        self.er_buffer.store_observation(observation)

        terminate = False
        while not terminate:
            #Step in the environment
            S, R, terminate = self.env.update(A)
            # Record Keeping
            episode_reward_sum += R

            if self.config.rand_steps_count >= self.initial_rand_steps:
                q_values = self.fa.get_next_states_values(S)
                A = self.bpolicy.choose_action(q_values)
                self.bpolicy.anneal()
                self.fa.update()
            else:
                A = np.random.randint(self.num_actions)
                self.config.rand_steps_count += 1

            observation = {"reward": R, "action": A, "state": self.env.get_state_for_er_buffer(),
                           'terminate': terminate}
            self.er_buffer.store_observation(observation)

        # End of episode
        if self.save_summary:
            self.summary['return_per_episode'].append(episode_reward_sum)
            self.env.reset()
