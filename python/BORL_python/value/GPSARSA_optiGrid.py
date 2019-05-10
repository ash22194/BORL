import numpy as np
import GPy
from tqdm import trange
from scipy.optimize import minimize
from ipdb import set_trace

class GPSARSA_optiGrid:
    def __init__(self, env, u_limits, nu, sigma0, gamma, epsilon, kernel, spraseD_size, sparsify_factor=2, Q_mu=[], policy_prior=[]):
        
        self.env = env
        self.actions = u_limits
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        self.spraseD_size = spraseD_size
        self.sparsify_factor = sparsify_factor
        if (not Q_mu):
            Q_mu = lambda s,a: np.zeros(s.shape[1])[:,np.newaxis]
        self.Q_mu = Q_mu
        if (not policy_prior):
            policy_prior = lambda s: np.repeat(self.actions[:,0][:,np.newaxis], s.shape[1], axis=1) + \
                                     np.repeat((self.actions[:,-1] - self.actions[:,0])[:,np.newaxis], s.shape[1], axis=1)* \
                                     np.random.rand(self.actions.shape[0], s.shape[1])
        self.policy_prior = policy_prior
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

        for i in range(reward_sequence.shape[0]):
            trajt_1 = np.concatenate((state_sequence[:,i][:,np.newaxis], action_sequence[:,i][:,np.newaxis]))
            trajt = np.concatenate((state_sequence[:,i+1][:,np.newaxis], action_sequence[:,i+1][:,np.newaxis]))
            k_t_1 = self.kernel(self.D, trajt_1)
            k_t = self.kernel(self.D, trajt)
            ktt = self.kernel(trajt, trajt)
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

        self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

    def optimize_dictionary_GPy(self):
        """
        Compute pseudo inputs that 'adequately' explain the observations (such that |pseudo_inputs| < |D|)
        """
        # First spraseD_size elements in the dictionary are the x_ from the previous optimization
        # Treat all the elements of the dictionary the same.. they are all "observations"
        # K_inv -> (K_vv)^(-1)

        K = np.linalg.inv(self.K_inv)
        self.Q = self.Q_D + np.dot(K, self.diff_alpha_CQ_D)
        self.K_diag = K.diagonal()
        Z = np.concatenate((self.env.sample_states(self.spraseD_size), \
                              (self.actions[0,1] - self.actions[0,0])*np.random.rand(1,self.spraseD_size) + \
                               self.actions[0,0]\
                             ), axis=0).T

        kernel = GPy.kern.RBF(input_dim=3, variance=13.6596**2, lengthscale=np.array([0.5977, 1.9957, 5.7314]), ARD=True)
        model = GPy.models.SparseGPRegression(X=self.D.T, Y=self.Q, Z=Z, kernel=kernel)
        model.optimize('lbfgs', max_iters=1000000)
        
        self.Q_mu = lambda s,a: model.predict(np.concatenate((s,a),axis=0).T)[0]
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.Q_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CQ_D = np.array([[]], dtype=np.float64, order='C')

    def optimize_dictionary(self):
        """
        Compute pseudo inputs that 'adequately' explain the observations (such that |pseudo_inputs| < |D|)
        """
        # First spraseD_size elements in the dictionary are the x_ from the previous optimization
        # Treat all the elements of the dictionary the same.. they are all "observations"
        # K_inv -> (K_vv)^(-1)

        K = np.linalg.inv(self.K_inv)
        self.Q = np.dot(K, self.alpha_)
        self.K_diag = K.diagonal()
        x_0 = np.concatenate((self.env.sample_states(self.spraseD_size), \
                              (self.actions[0,1] - self.actions[0,0])*np.random.rand(1,self.spraseD_size) + \
                               self.actions[0,0]\
                             ), axis=0)

        obj = lambda x_: self.compute_neg_log_probability_with_pseudo_inputs(x_)
        result = minimize(obj, x_0, options={'maxiter':1000000, 'disp':True}, method='L-BFGS-B', \
                            bounds=((self.env.x_limits[0]+0.001, self.env.x_limits[1]-0.001),)*self.spraseD_size + \
                                   ((self.env.x_dot_limits[0]+0.001, self.env.x_dot_limits[1]-0.001),)*self.spraseD_size + \
                                   ((self.actions[0,0]+0.001, self.actions[0,1]-0.001),)*self.spraseD_size)
        x_ = np.reshape(result.x, (self.D.shape[0], self.spraseD_size))
        K_m = self.kernel(x_, x_)
        K_m_inv = np.linalg.pinv(K_m)
        K_mn = self.kernel(x_, self.D)

        lambda_ = self.K_diag + self.sigma0**2 - np.dot(K_mn.T, np.dot(K_m_inv, K_mn)).diagonal()
        lambda_inv = np.diag(1/lambda_)
        Q_inv = np.linalg.pinv(K_m + np.dot(K_mn, np.dot(lambda_inv, K_mn.T)))
        alpha_ = np.dot(Q_inv,\
                        (-self.Q_mu(x_[0:self.D.shape[0]-self.actions.shape[0],:],\
                                    x_[self.D.shape[0]-self.actions.shape[0]:,:]\
                                    )\
                         + np.dot(K_mn, np.dot(lambda_inv,(self.Q - self.Q_D)))\
                        )\
                    )
        Q_mu_ = self.Q_mu
        self.Q_mu = lambda s,a: Q_mu_(s,a) + np.dot(self.kernel(np.concatenate((s,a), axis=0), x_), \
                                                    alpha_)
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.Q_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CQ_D = np.array([[]], dtype=np.float64, order='C')
        
        return x_

    def compute_neg_log_probability_with_pseudo_inputs(self, x_):

        """
        p(Y|X,X_) = N(mu(X) - K_nm(K_m_inv)(mu(X_)), K_nm(K_m_inv)(K_mn) + lambda_ + (sigma0**2)I)
        """
        if (len(x_.shape)==1):
            m = int(x_.shape[0]/self.D.shape[0])
            x_ = np.reshape(x_, (self.D.shape[0], m))
        elif (len(x_.shape)==2 and x_.shape[0]!=self.D.shape[0]):
            print('Check pseudo-inputs!')

        """
        Compute pseudo covariance matrices
        """
        K_m = self.kernel(x_, x_)
        K_m_inv = np.linalg.pinv(K_m)
        K_mn = self.kernel(x_, self.D)

        if (np.any(x_[0,:]>self.env.x_limits[1]) or np.any(x_[0,:]<self.env.x_limits[0]) \
            or np.any(x_[1,:]>self.env.x_dot_limits[1]) or np.any(x_[1,:]<self.env.x_dot_limits[0]) \
            or np.any(x_[2,:]>self.actions[0,1]) or np.any(x_[2,:]<self.actions[0,0])):
            set_trace()
        mean_posterior = self.Q_D - np.dot(K_mn.T, \
                                            np.dot(K_m_inv, \
                                                    self.Q_mu(x_[0:self.D.shape[0]-self.actions.shape[0],:],\
                                                              x_[self.D.shape[0]-self.actions.shape[0]:,:])))
        cov_posterior = np.dot(K_mn.T, np.dot(K_m_inv, K_mn))
#         set_trace()
        np.fill_diagonal(cov_posterior, self.K_diag + self.sigma0**2)
#         set_trace()
        # negLogProb = 0.5*(np.log(np.linalg.det(cov_posterior)) + np.dot((self.Q - mean_posterior).T,\
        #                                                              np.dot(np.linalg.pinv(cov_posterior), (self.Q - mean_posterior)))[0,0])
        negLogProb = np.dot((self.Q - mean_posterior).T,\
                            np.dot(np.linalg.pinv(cov_posterior), \
                                (self.Q - mean_posterior)))[0,0]
        return negLogProb

    def select_action(self, state, epsilon=0):
        """
        Select action epsilon-greedily
        Return action and corresponding Q value
        """

        num_actions_to_sample = 10
        actions = np.repeat(self.actions[:,0][:,np.newaxis], num_actions_to_sample, axis=1) +\
                  (np.random.rand(self.actions.shape[0], num_actions_to_sample) *\
                   np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], num_actions_to_sample, axis=1))
        action_explore = self.policy_prior(state)
        explore = np.random.rand()

        if (self.D.shape[1]==0):
            Q = self.Q_mu(state, action_explore)
            action = action_explore

        else:
            Q = np.empty((num_actions_to_sample, 1), dtype=np.float64, order='C')
            for a in range(num_actions_to_sample):
                action = actions[:,a][:,np.newaxis]
                traj = np.concatenate((state, action), axis=0)
                Q[a,0] = self.Q_mu(state, action) + np.dot(self.kernel(self.D, traj).T, self.diff_alpha_CQ_D)

            action_exploit = np.argmin(Q, axis=0)[0]
            Q_exploit = Q[action_exploit, 0]
            action_exploit = actions[:, action_exploit][:,np.newaxis]

            Q_explore = self.Q_mu(state, action_explore) +\
                np.dot(self.kernel(self.D,\
                                    np.concatenate((state, action_explore), axis=0)).T, \
                       self.diff_alpha_CQ_D)[0,0]

            action = (explore<epsilon)*action_explore + (explore>epsilon)*action_exploit
            Q = (explore<epsilon)*Q_explore + (explore>epsilon)*Q_exploit

        return Q, action

    def build_policy_monte_carlo(self, num_episodes, max_episode_length, test_every=np.inf, states_V_target=()):
        """
        """

        statistics = trange(num_episodes)
        test_error = np.array([])
        error_before_comp = 0
        error_after_comp = 0

        for e in statistics:
            is_terminal = False
            num_steps = 0
            state = self.env.reset()
            _, action = self.select_action(state, self.epsilon)

            state_sequence = np.empty((state.shape[0], max_episode_length+1), dtype=np.float64, order='C')
            state_sequence[:, 0] = state[:,0]
            action_sequence = np.empty((action.shape[0], max_episode_length+1), dtype=np.float64, order='C')
            action_sequence[:, 0] = action[:,0]
            reward_sequence = np.empty(max_episode_length, dtype=np.float64, order='C')
            
            while ((num_steps < max_episode_length) and (not is_terminal)):
                num_steps+=1
                state, reward, is_terminal = self.env.step(action)
                _, action = self.select_action(state, self.epsilon)

                state_sequence[:, num_steps] = state[:,0]
                action_sequence[:, num_steps] = action[:,0]
                reward_sequence[num_steps-1] = reward

            state_sequence = state_sequence[:, 0:(num_steps+1)]
            action_sequence = action_sequence[:, 0:(num_steps+1)]
            reward_sequence = reward_sequence[0:num_steps]

            if (self.D.shape[1]==0):

                traj = np.concatenate((state_sequence[:,0][:,np.newaxis], action_sequence[:,0][:,np.newaxis]))
                self.D = traj
                self.Q_D = self.Q_mu(state_sequence[:,0][:,np.newaxis], action_sequence[:,0][:,np.newaxis])
                self.K_inv = 1/self.kernel(traj, traj)
                self.A = np.array([[1]], dtype=np.float64, order='C')
                self.alpha_ = np.array([[0]], dtype=np.float64, order='C')
                self.C_= np.array([[0]], dtype=np.float64, order='C')
                self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

            self.update(state_sequence, action_sequence, reward_sequence)

            # if ((self.D.shape[1] >= self.spraseD_size*self.sparsify_factor) or ((e+1)%100==0)):
            if (self.D.shape[1] >= self.spraseD_size*self.sparsify_factor):    
            # if ((e+1)%100==0):

                if (len(states_V_target)>0):
                    V_before = self.get_value_function(states_V_target[0])
                # plt.subplot(2,1,1)
                # plt.imshow(np.reshape(V_before, (numPointsx, numPointsx_dot)).T, aspect='auto',\
                #             extent=(self.env.x_limits[0], self.env.x_limits[1], self.env.x_dot_limits[1], self.env.x_dot_limits[0]), origin='upper')
                # plt.colorbar()
                # plt.xlabel('theta')
                # plt.ylabel('theta-dot')
                # plt.title('Before')
                # plt.scatter(self.D[0,:], self.D[1,:], c='red')
                self.x_ = self.optimize_dictionary_GPy()
                if (len(states_V_target)>0):
                    V_after = self.get_value_function(states_V_target[0])
                    error_before_comp = np.mean(np.abs(V_before - states_V_target[1]))
                    error_after_comp = np.mean(np.abs(V_after - states_V_target[1]))
                    # print('Mean error before compression : %f'%np.mean(np.abs(V_before - states_V_target[1])))
                    # print('Mean error after compression : %f'%np.mean(np.abs(V_after - states_V_target[1])))

                # plt.subplot(2,1,2)
                # plt.imshow(np.reshape(V_after, (numPointsx, numPointsx_dot)).T, aspect='auto',\
                #             extent=(self.env.x_limits[0], self.env.x_limits[1], self.env.x_dot_limits[1], self.env.x_dot_limits[0]), origin='upper')
                # plt.colorbar()
                # plt.xlabel('theta')
                # plt.ylabel('theta-dot')
                # plt.title('After')
                # plt.scatter(x_[0,:], x_[1,:], c='red')
                # plt.show()

            statistics.set_postfix(epi_length=num_steps, dict_size=self.D.shape[1], cumm_cost=np.sum(reward_sequence),\
                                   err_bcomp=error_before_comp, err_acomp=error_after_comp)
            if (e%test_every==0 and len(states_V_target)==2):
                V = self.get_value_function(states_V_target[0])
                test_error = np.concatenate((test_error, np.array([np.mean(np.abs(V - states_V_target[1]))])))

    def get_value_function(self, states):
        
        V = np.zeros((states.shape[1],1), dtype=np.float64, order='C')
        for s in range(states.shape[1]):
            Q, _ = self.select_action(states[:,s][:,np.newaxis])
            V[s,0] = Q

        return V