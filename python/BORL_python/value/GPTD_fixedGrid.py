from tqdm import trange
import numpy as np

class GPTD_fixedGrid:
    def __init__(self, env, sigma0, gamma, kernel, D, V_mu=[]):
        
        self.env = env
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        if (not V_mu):
            V_mu = lambda s: np.zeros(s.shape[1])
        self.V_mu = V_mu
        self.V_D = self.V_mu(D)
        self.D = D
        # self.D = np.concatenate((self.D, self.V_D.T), axis=0) # Use V_mu in computing distances!
        self.A = np.zeros((self.D.shape[1],1), dtype=np.float64, order='C')
        self.A[-1,0] = 1
        K = self.kernel(self.D, self.D)
        self.K_inv = np.linalg.inv(K)
        self.alpha_ = np.zeros((self.D.shape[1],1), dtype=np.float64, order='C')
        self.C_ = np.zeros((self.D.shape[1],self.D.shape[1]), dtype=np.float64, order='C')
        self.diff_alpha_CV_D = np.empty((self.D.shape[1],1), dtype=np.float64, order='C')
        
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

            trajt_1 = state_sequence[:,i][:,np.newaxis]                     # No use of V_mu in computing distances!
            trajt = state_sequence[:,i+1][:,np.newaxis]
            # trajt_1 = np.concatenate((trajt_1, self.V_mu(trajt_1)), axis=0) # Use V_mu as well
            # trajt = np.concatenate((trajt, self.V_mu(trajt)), axis=0)
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

        self.diff_alpha_CV_D = self.alpha_ - np.dot(self.C_, self.V_D)

    def build_posterior(self, policy, num_episodes, max_episode_length):
        """
        policy is a function that take state as input and returns an action
        """

        statistics = trange(num_episodes)

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
                self.alpha_ = np.array([[0]])
                self.C_= np.array([[0]])
                self.diff_alpha_CV_D = self.alpha_ - np.dot(self.C_, self.V_D)

            self.update(state_sequence, reward_sequence)
            statistics.set_postfix(epi_length=num_steps, dict_size=self.D.shape[1], cumm_cost=np.sum(reward_sequence))

    def get_value_function(self, states):

        if (self.D.shape[1]==0):
            return self.V_mu(states) 

        else:
            return self.V_mu(states) + np.dot(self.kernel(self.D, states).T, self.diff_alpha_CV_D)