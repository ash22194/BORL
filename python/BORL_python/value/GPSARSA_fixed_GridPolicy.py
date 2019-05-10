from tqdm import trange
import numpy as np 
from ipdb import set_trace

class GPSARSA_fixed_GridPolicy:
    def __init__(self, env, u_limits, sigma0, gamma, epsilon, kernel, numElemsInD, Q_mu=[], simulation_policy=[], explore_policy=[]):
        
        self.env = env
        self.actions = u_limits
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        self.numElemsInD = numElemsInD
        if (not Q_mu):
            Q_mu = lambda s,a: np.zeros(s.shape[1])[:,np.newaxis] 
        self.Q_mu = Q_mu
        if (not explore_policy):
            explore_policy = lambda s: np.repeat(self.actions[:,0][:,np.newaxis], s.shape[1], axis=1) + \
                                     np.repeat((self.actions[:,-1] - self.actions[:,0])[:,np.newaxis], s.shape[1], axis=1)* \
                                     np.random.rand(self.actions.shape[0], s.shape[1])
        self.exploration_policy = explore_policy
        if (not simulation_policy):
            simulation_policy = lambda s: np.repeat(self.actions[:,0][:,np.newaxis], s.shape[1], axis=1) + \
                                     np.repeat((self.actions[:,-1] - self.actions[:,0])[:,np.newaxis], s.shape[1], axis=1)* \
                                     np.random.rand(self.actions.shape[0], s.shape[1])
        self.policy = lambda s,e: simulation_policy(s) if e<np.random.rand() else self.exploration_policy(s)
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.Q_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CQ_D = np.array([[]], dtype=np.float64, order='C')

    def k_(self,x):

        if (len(x.shape)==1):
            x = x[:,np.newaxis]
        assert len(x.shape)==2, "Check state dimensions"

        return self.kernel(self.D, np.repeat(x, self.D.shape[1], axis=1))

    def update(self, state_sequence, action_sequence, reward_sequence):
        """
        Update GP after observing states (state_sequence), 
        actions (action_sequence) and rewards (reward_sequence)
        """
        D = (np.concatenate((self.env.x_limits[0] + np.random.rand(1,self.numElemsInD)*(self.env.x_limits[-1] - self.env.x_limits[0]),\
                             self.env.x_dot_limits[0] + np.random.rand(1,self.numElemsInD)*(self.env.x_dot_limits[-1] - self.env.x_dot_limits[0])), axis=0), \
             np.repeat(self.actions[:,0][:,np.newaxis], self.numElemsInD, axis=1) +\
                  (np.random.rand(self.actions.shape[0], self.numElemsInD) *\
                   np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], self.numElemsInD, axis=1)))
        
        self.Q_D = self.Q_mu(D[0], D[1])
        self.D = np.concatenate((D[0], D[1]), axis=0)
        self.A = np.zeros((self.D.shape[1],1), dtype=np.float64, order='C')
        self.A[-1,0] = 1
        K = self.kernel(self.D, self.D)
        self.K_inv = np.linalg.inv(K)
        self.alpha_ = np.zeros((self.D.shape[1],1), dtype=np.float64, order='C')
        self.C_ = np.zeros((self.D.shape[1],self.D.shape[1]), dtype=np.float64, order='C')
        self.diff_alpha_CQ_D = np.empty((self.D.shape[1],1), dtype=np.float64, order='C')

        for i in range(reward_sequence.shape[0]):
            trajt_1 = np.concatenate((state_sequence[:,i][:,np.newaxis], action_sequence[:,i][:,np.newaxis]))
            trajt = np.concatenate((state_sequence[:,i+1][:,np.newaxis], action_sequence[:,i+1][:,np.newaxis]))

            k_t_1 = self.kernel(self.D, trajt_1)
            k_t = self.kernel(self.D, trajt)
            ktt = self.kernel(trajt, trajt)
            at = np.dot(self.K_inv, k_t)
            delk_t_1 = k_t_1 - self.gamma*k_t
        
            ct = np.dot(self.C_, delk_t_1) - (self.A - self.gamma*at)
            st = self.sigma0**2 - np.dot(ct.T, delk_t_1)

            diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - reward_sequence[i]
            self.alpha_ = self.alpha_ + ct/st*diff_r

            self.C_ = self.C_ + np.dot(ct, ct.T)/st

            self.A = at

            assert (not np.isnan(self.alpha_).any()), "Check alpha for NaN values"

        self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

        self.policy = lambda s,e: self.select_action_fromD(s,e)[1]

    def select_action_fromD(self, state, epsilon=0):
        """
        Select action epsilon-greedily
        Return action and corresponding Q value
        """

        num_actions_to_sample = 10
        actions = np.repeat(self.actions[:,0][:,np.newaxis], num_actions_to_sample, axis=1) +\
                  (np.random.rand(self.actions.shape[0], num_actions_to_sample) *\
                   np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], num_actions_to_sample, axis=1))
        action_explore = self.exploration_policy(state)
        explore = np.random.rand()

        if (self.D.shape[1]==0):
            Q = self.Q_mu(state, action_explore)[0,0]
            action = action_explore

        elif(explore<epsilon):
            traj = np.concatenate((state, action_explore), axis=0)
            Q = self.Q_mu(state, action_explore)[0,0] +\
                np.dot(self.kernel(self.D, traj).T, \
                       self.diff_alpha_CQ_D)[0,0]
            action = action_explore
        else:

            Q = np.empty((num_actions_to_sample, 1), dtype=np.float64, order='C')
            for a in range(num_actions_to_sample):
                action = actions[:,a][:,np.newaxis]
                traj = np.concatenate((state, action), axis=0)
                Q[a,0] = self.Q_mu(state, action)[0,0] + \
                         np.dot(self.kernel(self.D, traj).T, \
                                self.diff_alpha_CQ_D)[0,0]

            action_exploit = np.argmin(Q, axis=0)[0]
            Q = Q[action_exploit, 0]
            action_exploit = actions[:, action_exploit][:,np.newaxis]

        return Q, action

    def build_policy_monte_carlo(self, num_episodes, max_episode_length, update_every=1, states_V_target=()):
        """
        """

        statistics = trange(num_episodes)
        test_error = np.array([])
        current_value_error = 0
        current_pos_error = 0

        state = self.env.reset()
        action = self.policy(state, self.epsilon)
        num_steps = 0
        
        state_sequence = np.empty((state.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
        state_sequence[:,num_steps] = state[:,0]
        action_sequence = np.empty((action.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
        action_sequence[:,num_steps] = action[:,0]
        reward_sequence = np.empty(max_episode_length*(update_every+1), dtype=np.float64, order='C')

        for e in statistics:
            is_terminal = False
            num_steps_epi = 0

            while ((num_steps_epi < max_episode_length) and (not is_terminal)):
                num_steps+=1
                num_steps_epi+=1
                state, reward, is_terminal = self.env.step(action)
                action = self.policy(state, self.epsilon)

                state_sequence[:, num_steps] = state[:,0]
                action_sequence[:, num_steps] = action[:,0]
                reward_sequence[num_steps-1] = reward

            state = self.env.reset()
            action = self.policy(state, self.epsilon)

            if ((e+1)%update_every==0):
                state_sequence = state_sequence[:, 0:(num_steps+1)]
                action_sequence = action_sequence[:, 0:(num_steps+1)]
                reward_sequence = reward_sequence[0:num_steps]
                self.update(state_sequence, action_sequence, reward_sequence)
                _,_,pos_error = self.test_policy(num_episodes=10, max_episode_length=max_episode_length)
                current_pos_error = np.mean(np.linalg.norm(pos_error,axis=0))
                if (len(states_V_target)==2):
                    V = self.get_value_function(states_V_target[0])
                    current_value_error = np.array([np.mean(np.abs(V - states_V_target[1]))])
                    test_error = np.concatenate((test_error, current_value_error))

                state_sequence = np.empty((state.shape[0], (max_episode_length+1)*update_every), dtype=np.float64, order='C')
                action_sequence = np.empty((action.shape[0], (max_episode_length+1)*update_every), dtype=np.float64, order='C')
                reward_sequence = np.empty(max_episode_length*(update_every+1), dtype=np.float64, order='C')
                num_steps = 0
                
            else:
                num_steps += 1
                reward_sequence[num_steps-1] = 0

            state_sequence[:,num_steps] = state[:,0]
            action_sequence[:,num_steps] = action[:,0]

            statistics.set_postfix(epi_length=num_steps_epi, \
                                   dict_size=self.D.shape[1], \
                                   cumm_cost=np.sum(reward_sequence), \
                                   v_err=current_value_error, \
                                   pos_err=current_pos_error)

        return test_error

    def get_value_function(self, states):
        
        V = np.zeros((states.shape[1],1), dtype=np.float64, order='C')
        for s in range(states.shape[1]):
            Q, _ = self.select_action_fromD(states[:,s][:,np.newaxis])
            V[s,0] = Q

        return V

    def test_policy(self, num_episodes, max_episode_length):
        """
        Returns - tuples of (start_state, cumm_disc_reward, final_error)
        """
        state = self.env.reset()
        start_states = np.empty((state.shape[0] + self.actions.shape[0], num_episodes))
        cumm_rewards = np.empty(num_episodes)
        final_err = np.empty((state.shape[0], num_episodes))

        for e in range(num_episodes):
            state = self.env.reset()
            action = self.policy(state, 0)
            start_states[:,e] = np.concatenate((state, action))[:,0]
            is_terminal = False
            num_steps = 0
            cumm_reward = 0
            while ((not is_terminal) and (num_steps<max_episode_length)):
                state, reward, is_terminal = self.env.step(action)
                action = self.policy(state, 0)
                cumm_reward += reward*self.gamma**(num_steps)
                num_steps +=1

            cumm_rewards[e] = cumm_reward
            final_err[:,e] = (self.env.goal - state)[:,0]

        return start_states, cumm_rewards, final_err