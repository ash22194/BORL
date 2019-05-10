import numpy as np
import matplotlib.pyplot as plt
import GPy
from tqdm import trange
from scipy.optimize import minimize
from ipdb import set_trace
from copy import deepcopy

class GPTD_optiGrid:
    def __init__(self, env, nu, sigma0, gamma, kernel, spraseD_size, sparsify_factor=2, V_mu=[]):
        
        self.env = env
        self.nu = nu
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        self.spraseD_size = spraseD_size
        self.sparsify_factor = sparsify_factor
        if (not V_mu):
            V_mu = lambda s: np.zeros((s.shape[1], 1))
        self.V_mu = V_mu
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.V_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CV_D = np.array([[]], dtype=np.float64, order='C')

    def k_(self,x):

        if (len(x.shape)==1):
            x = x[:,np.newaxis]
        assert len(x.shape)==2, "Check state dimensions"

        return self.kernel(self.D, np.repeat(x, self.D.shape[1], axis=1))

    def update(self, state_sequence, reward_sequence):
        """
        Update GP after observing states (state_sequence) and rewards (reward_sequence)
        """

        for i in range(reward_sequence.shape[0]):

            trajt_1 = state_sequence[:,i][:,np.newaxis]
            trajt = state_sequence[:,i+1][:,np.newaxis]
            k_t_1 = self.kernel(self.D, trajt_1)
            k_t = self.kernel(self.D, trajt)
            ktt = self.kernel(trajt, trajt)
            at = np.dot(self.K_inv, k_t)
            et = (ktt - np.dot(k_t.T, at))
            delk_t_1 = k_t_1 - self.gamma*k_t

            if (et - self.nu) > 10**(-4):
                self.D = np.concatenate((self.D, trajt), axis=1)
                self.V_D = np.concatenate((self.V_D, self.V_mu(state_sequence[:,i+1][:,np.newaxis])), axis=0)

                at_by_et = at/et
                self.K_inv = np.concatenate((self.K_inv + np.dot(at, at.T)/et, -at_by_et), axis=1)
                self.K_inv = np.concatenate((self.K_inv, \
                                             np.concatenate((-at_by_et.T, 1/et), axis=1)), axis=0)

                c_t = np.dot(self.C_, delk_t_1) - self.A

                delktt = np.dot(self.A.T, delk_t_1 - self.gamma*k_t) + (self.gamma**2)*ktt
                s_t = self.sigma0**2 + delktt - np.dot(delk_t_1.T, np.dot(self.C_, delk_t_1))

                diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - reward_sequence[i]
                self.alpha_ = np.concatenate((self.alpha_ + c_t/s_t*diff_r, self.gamma/s_t*diff_r), axis=0)

                gc_t_by_s_t = (self.gamma/s_t)*c_t
                self.C_ = np.concatenate((self.C_ + np.dot(c_t, c_t.T)/s_t, gc_t_by_s_t), axis=1) 
                self.C_ = np.concatenate((self.C_, \
                                          np.concatenate((gc_t_by_s_t.T, self.gamma**2/s_t), axis=1)), axis=0)
                    
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

        self.diff_alpha_CV_D = self.alpha_ - np.dot(self.C_, self.V_D)

    def optimize_dictionary_GPy(self):

        K = np.linalg.inv(self.K_inv)
        self.V = self.V_D + np.dot(K, self.diff_alpha_CV_D)
        Z = self.env.sample_states(self.spraseD_size).T
        kernel = GPy.kern.RBF(input_dim=2, variance=7.6156**2, lengthscale=np.array([0.6345, 1.2656]), ARD=True)
        model = GPy.models.SparseGPRegression(X=self.D.T, Y=self.V, Z=Z, kernel=kernel)
        model.optimize('lbfgs', max_iters=1000000)
        
        self.V_mu = lambda x: model.predict(x.T)[0]
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.V_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CV_D = np.array([[]], dtype=np.float64, order='C')

    def optimize_dictionary(self):
        """
        Compute pseudo inputs that 'adequately' explain the observations (such that |pseudo_inputs| < |D|)
        """
        # First spraseD_size elements in the dictionary are the x_ from the previous optimization
        # Treat all the elements of the dictionary the same.. they are all "observations"
        # K_inv -> (K_vv)^(-1)

        K = np.linalg.inv(self.K_inv)
        self.V = self.V_D + np.dot(K, self.diff_alpha_CV_D)
        self.K_diag = K.diagonal()
        x_0 = self.env.sample_states(self.spraseD_size)

        obj = lambda x_: self.compute_neg_log_probability_with_pseudo_inputs(x_)
        result = minimize(obj, x_0, options={'maxiter':1000000, 'disp':True}, method='L-BFGS-B', \
                            bounds=((self.env.x_limits[0]+0.001, self.env.x_limits[1]-0.001),)*self.spraseD_size + \
                                   ((self.env.x_dot_limits[0]+0.001, self.env.x_dot_limits[1]-0.001),)*self.spraseD_size)
        x_ = np.reshape(result.x, (self.D.shape[0], self.spraseD_size))
        K_m = self.kernel(x_, x_)
        try:
            K_m_inv = np.linalg.pinv(K_m)
        except Exception as e:
            set_trace()
            raise e
        K_mn = self.kernel(x_, self.D)

        lambda_ = self.K_diag + self.sigma0**2 - np.dot(K_mn.T, np.dot(K_m_inv, K_mn)).diagonal()
        lambda_inv = np.diag(1/lambda_)
        Q_inv = np.linalg.pinv(K_m + np.dot(K_mn, np.dot(lambda_inv, K_mn.T)))
        alpha_ = np.dot(Q_inv,\
                        (-self.V_mu(x_) + np.dot(K_mn, \
                                                np.dot(lambda_inv,\
                                                    (self.V - self.V_D)))\
                        )\
                    )
        V_mu_ = self.V_mu
        self.V_mu = lambda x: V_mu_(x) + np.dot(self.kernel(x, x_), alpha_)
        self.D = np.array([[]], dtype=np.float64, order='C')
        self.A = np.array([[]], dtype=np.float64, order='C')
        self.V_D = np.array([[]], dtype=np.float64, order='C')
        self.K_inv = np.array([[]], dtype=np.float64, order='C')
        self.alpha_ = np.array([[]], dtype=np.float64, order='C')
        self.C_ = np.array([[]], dtype=np.float64, order='C')
        self.diff_alpha_CV_D = np.array([[]], dtype=np.float64, order='C')

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
            or np.any(x_[1,:]>self.env.x_dot_limits[1]) or np.any(x_[1,:]<self.env.x_dot_limits[0])):
            set_trace()
        mean_posterior = self.V_D - np.dot(K_mn.T, np.dot(K_m_inv, self.V_mu(x_)))
        cov_posterior = np.dot(K_mn.T, np.dot(K_m_inv, K_mn))
        np.fill_diagonal(cov_posterior, self.K_diag + self.sigma0**2)
        
        u,s,v = np.linalg.svd(cov_posterior)
        approx_zero_singular_values = s<10**(-3)
        s[approx_zero_singular_values] = 10**(-3)
        s_pinv = deepcopy(s)
        s_pinv[~approx_zero_singular_values] = 1/s[~approx_zero_singular_values]
        cov_posterior_pinv = np.dot(v.T, np.dot(np.diag(s_pinv), u.T))

        cov_posterior_det = np.prod(s)
        if (cov_posterior_det==np.inf):
            num_s = s.shape[0]
            s1 = s[0:int(num_s/2)]
            s2 = s[int(num_s/2):]
            if (num_s%2==0):
                cov_posterior_det = np.prod(s1*np.flip(s2))
            else:
                cov_posterior_det = np.prod(s1*np.flip(s2[0:int(num_s/2)]))*s2[-1]

        if (cov_posterior_det==0):
            cov_posterior_det = 10**(-10)

        negLogProb = 0.5*(np.log(cov_posterior_det) + np.dot((self.V - mean_posterior).T,\
                                                                     np.dot(cov_posterior_pinv, (self.V - mean_posterior)))[0,0])
        # negLogProb = np.dot((self.V - mean_posterior).T, \
        #                     np.dot(np.linalg.pinv(cov_posterior), \
        #                            (self.V - mean_posterior)))[0,0]
        return negLogProb

    def build_posterior(self, policy, num_episodes, max_episode_length, test_every=np.inf, states_V_target=()):
        """
        policy is a function that take state as input and returns an action
        """

        statistics = trange(num_episodes)
        test_error = np.array([])
        error_before_comp = 0
        error_after_comp = 0

        for e in statistics:
            is_terminal = False
            num_steps = 0
            state = self.env.reset()
            action = policy(state)
            
            state_sequence = np.empty((state.shape[0], max_episode_length+1), dtype=np.float64, order='C')
            state_sequence[:, 0] = state[:,0]
            reward_sequence = np.empty(max_episode_length, dtype=np.float64, order='C')
            
            while ((num_steps < max_episode_length) and (not is_terminal)):
                num_steps+=1
                state, reward, is_terminal = self.env.step(action)
                action = policy(state)

                state_sequence[:, num_steps] = state[:,0]
                reward_sequence[num_steps-1] = reward

            state_sequence = state_sequence[:, 0:(num_steps+1)]
            reward_sequence = reward_sequence[0:num_steps]

            if (self.D.shape[1]==0):

                traj = state_sequence[:,0][:,np.newaxis]
                self.D = traj
                self.V_D = self.V_mu(state_sequence[:,0][:,np.newaxis])
                self.K_inv = 1/self.kernel(traj, traj)
                self.A = np.array([[1]])
                self.H_ = np.array([[]], dtype=np.float64, order='C')
                self.Q_ = np.array([[]], dtype=np.float64, order='C') # Q = (HK_H' + sigma0^2I)
                self.alpha_ = np.array([[0]])
                self.C_= np.array([[0]])
                self.diff_alpha_CV_D = self.alpha_ - np.dot(self.C_, self.V_D)

            self.update(state_sequence, reward_sequence)
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
                self.x_ = self.optimize_dictionary()
                # self.optimize_dictionary_GPy()
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

            statistics.set_postfix(epi_length=num_steps, dict_size=self.D.shape[1], cumm_cost=np.sum(reward_sequence), \
                                   err_bcomp=error_before_comp, err_acomp=error_after_comp)
            if (e%test_every==0 and len(states_V_target)==2):
                V = self.get_value_function(states_V_target[0])
                test_error = np.concatenate((test_error, np.array([np.mean(np.abs(V - states_V_target[1]))])))

        return test_error

    def get_value_function(self, states):

        if (self.D.shape[1]==0):
            return self.V_mu(states) 

        else:
            return self.V_mu(states) + np.dot(self.kernel(self.D, states).T, self.diff_alpha_CV_D)
            
        