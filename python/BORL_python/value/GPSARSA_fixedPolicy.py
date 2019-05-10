import numpy as np 
import GPy
from tqdm import trange
from ipdb import set_trace

class GPSARSA_fixedPolicy:
    def __init__(self, env, u_limits, nu, sigma0, gamma, epsilon, kernel, Q_mu=[], simulation_policy=[], explore_policy=[]):
        
        self.env = env
        self.actions = u_limits
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma0 = sigma0
        self.kernel = kernel
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

        return self.kernel.kernel(self.D, np.repeat(x, self.D.shape[1], axis=1))

    def update(self, state_sequence, action_sequence, reward_sequence):
        """
        Update GP after observing states (state_sequence), 
        actions (action_sequence) and rewards (reward_sequence)
        """

        traj = np.concatenate((state_sequence[:,0][:,np.newaxis], action_sequence[:,0][:,np.newaxis]))
        self.D = traj
        self.Q_D = self.Q_mu(state_sequence[:,0][:,np.newaxis], action_sequence[:,0][:,np.newaxis])
        self.K_inv = 1/self.kernel.kernel(traj, traj)
        self.A = np.array([[1]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[0]], dtype=np.float64, order='C')
        self.C_= np.array([[0]], dtype=np.float64, order='C')
        self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

        for i in range(reward_sequence.shape[0]):
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
                                           self.Q_mu(state_sequence[:,i+1][:,np.newaxis], \
                                                     action_sequence[:,i+1][:,np.newaxis])), axis=0)

                at_by_et = at/et
                self.K_inv = np.concatenate((self.K_inv + np.dot(at_by_et, at.T), -at_by_et), axis=1)
                self.K_inv = np.concatenate((self.K_inv, np.concatenate((-at_by_et.T, 1/et), axis=1)), axis=0)

                c_t = np.dot(self.C_, delk_t_1) - self.A

                delktt = np.dot(self.A.T, delk_t_1 - self.gamma*k_t) + (self.gamma**2)*ktt
                s_t = self.sigma0**2 + delktt - np.dot(delk_t_1.T, np.dot(self.C_, delk_t_1))

                diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - reward_sequence[i]
                self.alpha_ = np.concatenate((self.alpha_ + c_t/s_t*diff_r, self.gamma/s_t*diff_r), axis=0)

                gc_t_by_s_t = (self.gamma/s_t)*c_t
                self.C_ = np.concatenate((self.C_ + np.dot(c_t, c_t.T)/s_t, gc_t_by_s_t), axis=1) 
                self.C_ = np.concatenate((self.C_, np.concatenate((gc_t_by_s_t.T, self.gamma**2/s_t), axis=1)), axis=0)

                self.A = np.zeros((self.A.shape[0]+1, self.A.shape[1]), dtype=np.float64, order='C')
                self.A[-1, 0] = 1

            else:

                ct = np.dot(self.C_, delk_t_1) - (self.A - self.gamma*at)
                st = self.sigma0**2 - np.dot(ct.T, delk_t_1)

                diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - reward_sequence[i]
                self.alpha_ = self.alpha_ + ct/st*diff_r

                self.C_ = self.C_ + np.dot(ct, ct.T)/st

                self.A = at

            assert (not np.isnan(self.alpha_).any()), "Check alpha for NaN values"
        
        # Optimize hyper-parameters from data
        # print('Optimizing hyper-parameters')
        # kernel = GPy.kern.RBF(input_dim=self.D.shape[0], variance=self.kernel.sigma_f, lengthscale=self.kernel.sigma_l[:,0], ARD=True)
        # mean_function = GPy.core.Mapping(self.D.shape[0], 1)
        # mean_function.f = lambda x: self.Q_mu(x.T[0:-self.actions.shape[1],:], x.T[-self.actions.shape[1]:,:])
        # mean_function.update_gradients = lambda a,b: None
        # self.model = GPy.models.GPRegression(X=self.D.T, \
        #                                      Y=np.dot(np.linalg.inv(self.K_inv), self.alpha_),\
        #                                      kernel=kernel,\
        #                                      mean_function=mean_function)
        # self.model.optimize('lbfgsb')
        # self.kernel.sigma_f = self.model.flattened_parameters[0][0]
        # self.kernel.sigma_l = np.array([[self.model.flattened_parameters[1][0]], \
        #                                 [self.model.flattened_parameters[1][1]], \
        #                                 [self.model.flattened_parameters[1][2]]])

        # self.epsilon /= 2

        self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

        self.policy = lambda s,e: self.select_action_fromD(s,e)[1]

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

        if (self.D.shape[1]==0):
            Q = self.Q_mu(state, action_explore)[0,0]
            action = action_explore

        elif(explore<epsilon):
            traj = np.concatenate((state, action_explore), axis=0)
            k_ = self.kernel.kernel(self.D, traj)
            Q = self.Q_mu(state, action_explore)[0,0] +\
                np.dot(k_.T, self.diff_alpha_CQ_D)[0,0] +\
                np.random.randn()*(self.kernel.kernel(traj,traj) - np.dot(k_.T, np.dot(self.C_, k_)))[0,0]
            # Q = self.model.predict(traj.T)[0][0,0] + self.model.predict(traj.T)[1][0,0]*np.random.randn()
            action = action_explore

        else:
            Q = np.empty((num_actions_to_sample, 1), dtype=np.float64, order='C')
            for a in range(num_actions_to_sample):
                action = actions[:,a][:,np.newaxis]
                traj = np.concatenate((state, action), axis=0)
                k_ = self.kernel.kernel(self.D, traj)
                Q[a,0] = self.Q_mu(state, action)[0,0] + \
                         np.dot(k_.T, self.diff_alpha_CQ_D)[0,0] +\
                         np.random.randn()*(self.kernel.kernel(traj,traj) - np.dot(k_.T, np.dot(self.C_, k_)))[0,0]
                # Q[a,0] = self.model.predict(traj.T)[0][0,0] + self.model.predict(traj.T)[1][0,0]*np.random.randn()

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
                test_pos_error = np.concatenate((test_pos_error, np.array([current_pos_error])))

                if (len(states_V_target)==2):
                    V = self.get_value_function(states_V_target[0])
                    current_value_error = np.array([np.mean(np.abs(V - states_V_target[1]))])
                    test_value_error = np.concatenate((test_value_error, current_value_error))

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

        return test_value_error, test_pos_error

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