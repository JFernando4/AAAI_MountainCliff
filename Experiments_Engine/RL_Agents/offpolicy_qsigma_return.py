import numpy as np
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class OffPolicyQSigmaReturnFunction:

    def __init__(self, tpolicy, config=None, bpolicy=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        compute_bprobabilities  bool            False               whether to recompute bprobabilities or used
                                                                    the ones stored in the trajectory. This is the 
                                                                    difference between on-policy and off-policy updates.
        truncate_rho            bool            False               whether to truncate the importance sampling ratio
                                                                    at 1    
        """
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.compute_bprobabilities = check_attribute_else_default(config, 'compute_bprobabilities', False)
        self.truncate_rho = check_attribute_else_default(config, 'truncate_rho', False)

        """
        Other Parameters:
        tpolicy - The target policy
        bpolicy - Behaviour policy. Only required if compute_bprobabilities is True.
        """
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy
        if self.compute_bprobabilities:
            assert self.bpolicy is not None

    def recursive_return_function(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas, step=0):
        if step == self.n:
            raise RecursionError('This case should be impossible!')
        else:
            r = rewards[step]
            T = terminations[step]
            if T:
                return r
            else:
                a = actions[step]
                qv = qvalues[step]
                bprob = bprobabilities[step]
                sig = sigmas[step]
                tprob = self.tpolicy.probability_of_action(q_values=qv, all_actions=True)
                if self.compute_bprobabilities:
                    bprob = self.bpolicy.probability_of_action(q_values=qv, all_actions=True)
                assert bprob[a] > 0
                rho = tprob[a] / bprob[a]
                if self.truncate_rho:
                    rho = min(rho, 1)
                average_action_value = np.sum(np.multiply(tprob, qv))
                if step == self.n -1:
                    next_return = qv[a]
                else:
                    next_return = self.recursive_return_function(rewards, actions, qvalues, terminations,
                                                                  bprobabilities, sigmas, step=step+1)
                return r + self.gamma * (rho * sig + (1-sig) * tprob[a]) * next_return + \
                       self.gamma * (1-sig) * (average_action_value - tprob[a] * qv[a])

    def iterative_return_function(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas):
        trajectory_len = len(rewards)
        estimate_return = qvalues[trajectory_len-1][actions[trajectory_len-1]]

        for i in range(trajectory_len-1, -1, -1):
            if terminations[i]:
                estimate_return = rewards[i]
            else:
                R_t = rewards[i]
                A_t = actions[i]
                Q_t = qvalues[i]
                Sigma_t = sigmas[i]

                tprobs = self.tpolicy.probability_of_action(Q_t, all_actions=True)
                bprobs = bprobabilities[i]
                if self.compute_bprobabilities: bprobs = self.bpolicy.probability_of_action(Q_t, all_actions=True)
                assert bprobs[A_t] != 0

                rho = tprobs[A_t]/bprobs[A_t]
                v_t = np.sum(np.multiply(tprobs, Q_t))
                pi_t = tprobs[A_t]

                estimate_return = R_t + self.gamma * (rho * Sigma_t + (1-Sigma_t) * pi_t) * estimate_return +\
                                  self.gamma * (1-Sigma_t) * (v_t - pi_t * Q_t[A_t])
        return estimate_return

    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas,
                                        batch_size):
        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 0.
            All the terminations indicators after the terminal state are True
            All the bprobabilities and tprobabilities after the terminal state are 1

        :param rewards: expected_shape = [batch_size, n]
        :param actions: expected_shape = [batch_size, n], expected_type = np.uint8, np.uint16, np.uint32, or np.uint64
        :param qvalues: expected_shape = [batch_size, n, num_actions]
        :param terminations: expected_shape = [batch_size, n]
        :param bprobabilities: expected_shape = [batch_size, n, num_actions]
        :param sigmas: expected_shape = [batch_size, n]
        :param selected_qval: expected_shape = [batch_size, n]
        ;param batch_size: dtype = int
        :return: estimated_returns:
        """
        num_actions = self.tpolicy.num_actions
        tprobabilities = np.zeros([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)
        bprobabilities = bprobabilities if not self.compute_bprobabilities \
                         else np.zeros([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)

        for i in range(self.n):
            tprobabilities[:,i] = self.tpolicy.batch_probability_of_action(qvalues[:,i])
            if self.compute_bprobabilities:
                bprobabilities[:, i] = self.bpolicy.batch_probability_of_action(qvalues[:,i])

        selected_qval = qvalues.take(np.arange(actions.size) * num_actions + actions.flatten()).reshape(actions.shape)
        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size)
        one_matrix = np.ones([batch_idxs.size, self.n], dtype=np.uint8)
        term_ind = terminations.astype(np.uint8)
        neg_term_ind = np.subtract(one_matrix, term_ind)
        estimated_Gt = neg_term_ind[:,-1] * selected_qval[:,-1] + term_ind[:,-1] * rewards[:,-1]

        for i in range(self.n-1, -1, -1):
            R_t = rewards[:, i]
            A_t = actions[:, i]
            Q_t = qvalues[:, i, :]
            Sigma_t = sigmas[:, i]
            exec_q = Q_t[batch_idxs, A_t]       # The action-value of the executed actions
            assert np.sum(exec_q == selected_qval[:,i]) == batch_size
            tprob = tprobabilities[:, i, :]     # The probability of the executed actions under the target policy
            exec_tprob = tprob[batch_idxs, A_t]
            bprob = bprobabilities[:, i, :]
            exec_bprob = bprob[batch_idxs, A_t] # The probability of the executed actions under the behaviour policy
            rho = np.divide(exec_tprob, exec_bprob)
            V_t = np.sum(np.multiply(Q_t, tprob), axis=-1)

            G_t = R_t + self.gamma * (rho * Sigma_t + (one_vector - Sigma_t) * exec_tprob) * estimated_Gt +\
                  self.gamma * (one_vector - Sigma_t) * (V_t - exec_tprob * exec_q)
            estimated_Gt = neg_term_ind[:,i] * G_t + term_ind[:,i] * R_t

        return estimated_Gt
