import numpy as np 
import GPy
from tqdm import trange
from ipdb import set_trace

class GPSARSA_monteCarlo:
    def __init__(self, env, u_limits, nu, sigma0, gamma, epsilon, kernel, sparseD_size, Q_mu=[], simulation_policy=[], explore_policy=[]):
        
        self.env = env
        self.actions = u_limits
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma0 = sigma0
        self.kernel = kernel
        self.sparseD_size = sparseD_size
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

    def k_(self,x):

        if (len(x.shape)==1):
            x = x[:,np.newaxis]
        assert len(x.shape)==2, "Check state dimensions"

        return self.kernel.kernel(self.D, np.repeat(x, self.D.shape[1], axis=1))

    def update(self, state_sequence, action_sequence, value_sequence):
        """
        Update GP after observing states (state_sequence), 
        actions (action_sequence) and monte-carlo returns (value_sequence)
        """

        kernel = GPy.kern.RBF(input_dim=3, lengthscale=self.kernel.sigma_l[:,0], variance=self.kernel.sigma_f, ARD=True)
        Z = np.concatenate((self.env.sample_states(self.sparseD_size), \
                              (self.actions[0,1] - self.actions[0,0])*np.random.rand(1,self.sparseD_size) + \
                               self.actions[0,0]\
                             ), axis=0).T

        model = GPy.models.SparseGPRegression(X=np.concatenate((state_sequence, action_sequence), axis=0).T, \
                                              Y=value_sequence[:,np.newaxis], \
                                              Z=Z, \
                                              kernel=kernel)
        model.optimize('lbfgsb', max_iters=1000000)

    def select_action_fromD(self, state, epsilon=0):
        """
        Select action epsilon-greedily
        Return action and corresponding Q value
        """

        num_actions_to_sample = 15
        actions = np.repeat(self.actions[:,0][:,np.newaxis], num_actions_to_sample, axis=1) +\
                  (np.random.rand(self.actions.shape[0], num_actions_to_sample) *\
                   np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], num_actions_to_sample, axis=1))
        action_explore = self.exploration_policy(state)
        explore = np.random.rand()

        if(explore<epsilon):
            Q = self.get_Q(state, action_explore)[0,0]
            action = action_explore

        else:
            states = np.repeat(state, num_actions_to_sample, axis=1)
            Q = self.get_Q(states, actions, mu_only=False)
            action_exploit = np.argmin(Q, axis=0)[0]
            Q = Q[action_exploit, 0]
            action = actions[:, action_exploit][:,np.newaxis]

        return Q, action

    def build_policy_monte_carlo(self, num_episodes, max_episode_length, update_every=1, states_V_target=()):
        """
        """

        statistics = trange(num_episodes)
        test_value_error = np.array([])
        test_pos_error = np.array([])
        current_value_error = 0
        current_pos_error = 0

        state = self.env.reset()
        action = self.policy(state, self.epsilon)
        num_steps = 0
        
        state_sequence = np.zeros((state.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
        state_sequence[:,num_steps] = state[:,0]
        action_sequence = np.zeros((action.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
        action_sequence[:,num_steps] = action[:,0]
        value_sequence = np.zeros((max_episode_length+1)*(update_every+1), dtype=np.float64, order='C')

        for e in statistics:
            is_terminal = False
            num_steps_epi = 0
            reward_sequence = np.zeros(max_episode_length, dtype=np.float64, order='C')

            while ((num_steps_epi < max_episode_length) and (not is_terminal)):
                num_steps+=1
                num_steps_epi+=1
                state, reward, is_terminal = self.env.step(action)
                action = self.policy(state, self.epsilon)

                state_sequence[:, num_steps] = state[:,0]
                action_sequence[:, num_steps] = action[:,0]
                reward_sequence[num_steps_epi-1] = reward

            reward_sequence = reward_sequence[0:num_steps_epi]
            reward_sequence = np.concatenate((reward_sequence, np.zeros(1)))
            discounted_reward_sequence = \
                    np.array([self.gamma**i for i in range(num_steps_epi+1)])*reward_sequence
            value_sequence[num_steps-num_steps_epi:num_steps+1] = self.get_Q(state, action) +\
                    np.array([np.sum(discounted_reward_sequence[i:]) for i in range(num_steps_epi+1)])

            state = self.env.reset()
            action = self.policy(state, self.epsilon)
            # set_trace()

            if ((e+1)%update_every==0):
                state_sequence = state_sequence[:, 0:(num_steps+1)]
                action_sequence = action_sequence[:, 0:(num_steps+1)]
                value_sequence = value_sequence[0:(num_steps+1)]
                self.update(state_sequence, action_sequence, value_sequence)
                _,_,pos_error = self.test_policy(num_episodes=10, max_episode_length=max_episode_length)
                current_pos_error = np.mean(np.linalg.norm(pos_error,axis=0))
                test_pos_error = np.concatenate((test_pos_error, np.array([current_pos_error])))

                if (len(states_V_target)==2):
                    V = self.get_value_function(states_V_target[0])
                    current_value_error = np.array([np.mean(np.abs(V - states_V_target[1]))])
                    test_value_error = np.concatenate((test_value_error, current_value_error))

                state_sequence = np.empty((state.shape[0], (max_episode_length+1)*update_every), dtype=np.float64, order='C')
                action_sequence = np.empty((action.shape[0], (max_episode_length+1)*update_every), dtype=np.float64, order='C')
                value_sequence = np.empty((max_episode_length+1)*(update_every+1), dtype=np.float64, order='C')
                num_steps = 0

            else:
                num_steps += 1

            state_sequence[:,num_steps] = state[:,0]
            action_sequence[:,num_steps] = action[:,0]

            statistics.set_postfix(epi_length=num_steps_epi, \
                                   dict_size=self.sparseD_size, \
                                   cumm_cost=np.sum(reward_sequence), \
                                   v_err=current_value_error, \
                                   pos_err=current_pos_error)

        return test_value_error, test_pos_error

    def get_Q(self, states, actions, mu_only=True):

        if (not hasattr(self, 'model')):
            Q_mus = self.Q_mu(states, actions)
            Q_vars = np.zeros(Q_mus.shape)

        else:
            Q_mus, Q_vars = self.model.predict(np.concatenate((states, actions), axis=0))

        if (mu_only):
            return Q_mus
        else:
            return Q_mus + np.random.randn(Q_vars.shape[0],1)*Q_vars          

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