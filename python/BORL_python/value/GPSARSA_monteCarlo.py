import numpy as np 
import GPy
# import climin
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
        self.Q_D_mu = 0
        self.Q_D_sigma = 1
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
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        
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

        self.D = np.concatenate((state_sequence[:,0][:,np.newaxis], \
                                 action_sequence[:,0][:,np.newaxis]))
        self.Q_D = np.array([[value_sequence[0]]])
        self.K_inv = 1/self.kernel.kernel(self.D, self.D)

        for i in trange(value_sequence.shape[0]-1, position=1):
            trajt_1 = np.concatenate((state_sequence[:,i][:,np.newaxis], action_sequence[:,i][:,np.newaxis]))
            trajt = np.concatenate((state_sequence[:,i+1][:,np.newaxis], action_sequence[:,i+1][:,np.newaxis]))
            k_t_1 = self.kernel.kernel(self.D, trajt_1)
            k_t = self.kernel.kernel(self.D, trajt)
            ktt = self.kernel.kernel(trajt, trajt)
            at = np.dot(self.K_inv, k_t)
            et = (ktt - np.dot(k_t.T, at))
            delk_t_1 = k_t_1 - self.gamma*k_t

            if (et - self.nu) > 10**(-4):
                self.D = np.concatenate((self.D, trajt), axis=1)
                self.Q_D = np.concatenate((self.Q_D, \
                                           np.array([[value_sequence[i+1]]])), axis=0)
                
                at_by_et = at/et
                self.K_inv = np.concatenate((self.K_inv + np.dot(at_by_et, at.T), -at_by_et), axis=1)
                self.K_inv = np.concatenate((self.K_inv, np.concatenate((-at_by_et.T, 1/et), axis=1)), axis=0)

        kernel = GPy.kern.RBF(input_dim=3, lengthscale=self.kernel.sigma_l[:,0], variance=self.kernel.sigma_f, ARD=True)
        Z = np.concatenate((self.env.sample_states(self.sparseD_size), \
                            (self.actions[0,1] - self.actions[0,0])*np.random.rand(1,self.sparseD_size) + \
                             self.actions[0,0]), axis=0).T

        # self.model = GPy.core.SVGP(X=self.D.T, \
        #                            Y=self.Q_D, \
        #                            Z=Z, \
        #                            kernel=kernel,
        #                            likelihood=GPy.likelihoods.Gaussian(),
        #                            batchsize=400)
        self.Q_D_mu = np.mean(self.Q_D)
        self.Q_D_sigma = np.std(self.Q_D)
        self.model = GPy.models.SparseGPRegression(X=self.D.T,
                                                   Y=(self.Q_D - self.Q_D_mu)/self.Q_D_sigma,
                                                   kernel=kernel,
                                                   num_inducing=1000)

        # optimizer = climin.Adadelta(self.model.optimizer_array, self.model.stochastic_grad, step_rate=0.2, momentum=0.9)
        # for info in optimizer:
        #     if (info['n_iter']>=1000):
        #         break
        # self.model = GPy.models.SparseGPRegression(X=np.concatenate((state_sequence, action_sequence), axis=0).T, \
        #                                            Y=value_sequence[:,np.newaxis], \
        #                                            Z=Z, \
        #                                            kernel=kernel)
        print("optimizing model")
        self.model.optimize(messages=True, ipython_notebook=False, max_iters=1000)

        self.policy = lambda s,e: self.select_action_fromD(s, mu_only=False, epsilon=e)[1]

    def select_action_fromD(self, state, mu_only=False, epsilon=0):
        """
        Select action epsilon-greedily
        Return action and corresponding Q value
        """

        explore = np.random.rand()

        if(explore<epsilon):
            actions = self.exploration_policy(state)
            Q_mus, Q_sigma = self.get_Q(state, actions)

        else:
            num_actions_to_sample = 15
            actions = np.repeat(self.actions[:,0][:,np.newaxis], num_actions_to_sample, axis=1) +\
                      (np.random.rand(self.actions.shape[0], num_actions_to_sample) *\
                       np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], num_actions_to_sample, axis=1))
            states = np.repeat(state, num_actions_to_sample, axis=1)
            Q_mus, Q_sigma = self.get_Q(states, actions)
        
        Q = Q_mus + mu_only*np.random.rand(Q_sigma.shape[0], Q_sigma.shape[1])*Q_sigma
        action = np.argmin(Q, axis=0)[0]
        Q = Q[action, 0]
        action = actions[:, action][:,np.newaxis]

        return Q, action

    def build_policy_monte_carlo(self, num_episodes, max_episode_length, update_every=1, update_length=0, states_V_target=()):
        """
        """

        if (update_length==0):
            update_length = max_episode_length

        statistics = trange(num_episodes)
        test_value_error = np.array([])
        test_value_var = np.array([])
        test_pos_error = np.array([])
        current_value_error = 0
        current_value_var = 0
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
            num_steps_epi_ = 0
            reward_sequence = np.empty(max_episode_length, dtype=np.float64, order='C')

            while ((num_steps_epi < max_episode_length) and (not is_terminal)):
                
                state, reward, is_terminal = self.env.step(action)
                action = self.policy(state, self.epsilon)
                num_steps_epi_+=1
                while(num_steps_epi<update_length):
                    num_steps+=1
                    num_steps_epi+=1
                    state_sequence[:, num_steps] = state[:,0]
                    action_sequence[:, num_steps] = action[:,0]
                    reward_sequence[num_steps_epi-1] = reward

            reward_sequence = reward_sequence[0:num_steps_epi]
            reward_sequence = np.concatenate((reward_sequence, np.zeros(1)))
            discounted_reward_sequence = \
                    np.array([self.gamma**i for i in range(num_steps_epi+1)])*reward_sequence
            value_sequence[num_steps-num_steps_epi:num_steps+1] = self.get_Q(state, action)[0][0,0] +\
                    np.array([np.sum(discounted_reward_sequence[i:]) for i in range(num_steps_epi+1)])
            # print('Final cost : ', self.get_Q(state, action)[0][0,0])
            state = self.env.reset()
            action = self.policy(state, self.epsilon)

            if ((e+1)%update_every==0):
                state_sequence = state_sequence[:, 0:(num_steps+1)]
                action_sequence = action_sequence[:, 0:(num_steps+1)]
                value_sequence = value_sequence[0:(num_steps+1)]
                self.update(state_sequence, action_sequence, value_sequence)
                _,_,pos_error = self.test_policy(num_episodes=10, max_episode_length=max_episode_length)
                current_pos_error = np.mean(np.linalg.norm(pos_error, axis=0))
                test_pos_error = np.concatenate((test_pos_error, np.array([current_pos_error])))

                if (len(states_V_target)==2):
                    V, policy = self.get_value_function(states_V_target[0])
                    current_value_error = np.array([np.mean(np.abs(V - states_V_target[1]))])
                    test_value_error = np.concatenate((test_value_error, current_value_error))
                    
                    _, V_sigma = self.get_Q(states_V_target[0], policy)
                    current_value_var = np.array([np.mean(np.abs(V_sigma))])
                    test_value_var = np.concatenate((test_value_var, current_value_var))

                state_sequence = np.empty((state.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
                action_sequence = np.empty((action.shape[0], (max_episode_length+1)*(update_every+1)), dtype=np.float64, order='C')
                value_sequence = np.empty((max_episode_length+1)*(update_every+1), dtype=np.float64, order='C')
                num_steps = 0

            else:
                num_steps += 1

            state_sequence[:,num_steps] = state[:,0]
            action_sequence[:,num_steps] = action[:,0]

            statistics.set_postfix(epi_length=num_steps_epi_, \
                                   dict_size=self.sparseD_size, \
                                   cumm_cost=np.sum(reward_sequence), \
                                   v_err=current_value_error, \
                                   v_var=current_value_var, \
                                   pos_err=current_pos_error)

        return test_value_error, test_value_var, test_pos_error

    def get_Q(self, states, actions):

        if (not hasattr(self,'model')):
            Q_mus = self.Q_D_mu + self.Q_D_sigma*self.Q_mu(states, actions)
            Q_sigma = np.zeros(Q_mus.shape)

        else:
            Q_mus, Q_vars = self.model.predict(np.concatenate((states, actions), axis=0).T)
            Q_mus = self.Q_D_mu + self.Q_D_sigma*Q_mus
            Q_sigma = self.Q_D_sigma*np.sqrt(Q_vars)

        return Q_mus, Q_sigma         

    def get_value_function(self, states):
        
        V = np.zeros((states.shape[1],1), dtype=np.float64, order='C')
        policy = np.zeros((self.actions.shape[0], states.shape[1]), dtype=np.float64, order='C')
        for s in range(states.shape[1]):
            Q, a = self.select_action_fromD(states[:,s][:,np.newaxis], mu_only=True)
            V[s,0] = Q
            policy[:,s] = a[:,0]

        return V, policy

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