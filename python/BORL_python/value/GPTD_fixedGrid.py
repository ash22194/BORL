from tqdm import trange
import numpy as np

class GPTD_fixedGrid:
    def __init__(self, env, nu, sigma0, gamma, kernel, D, V_mu=[]):
        
        self.env = env
        self.nu = nu
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        if (not V_mu):
            V_mu = lambda s: np.zeros((s.shape[1],1))
        self.V_mu = V_mu
        self.D = D
        self.A = np.zeros((self.D.shape[1],1))
        self.A[-1,0] = 1
        self.V_D = self.V_mu(self.D)
        K = np.zeros((self.D.shape[1], self.D.shape[1]))
        for i in range(self.D.shape[1]):
            K[:,i] = self.k_(self.D[:,i])[:,0]
        self.K_inv = np.linalg.inv(K)
        self.alpha_ = np.zeros((self.D.shape[1],1))
        self.C_ = np.zeros((self.D.shape[1],self.D.shape[1]))

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
            k_t_1 = self.k_(trajt_1)
            k_t = self.k_(trajt)
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
            
            state_sequence = np.zeros((state.shape[0], max_episode_length+1))
            state_sequence[:, 0] = state[:,0]
            reward_sequence = np.zeros(max_episode_length)
            
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
                self.V_D = self.V_mu(state_sequence[:,0])
                self.K_inv = 1/self.kernel(traj, traj)
                self.A = np.array([[1]])
                self.alpha_ = np.array([[0]])
                self.C_= np.array([[0]])

            self.update(state_sequence, reward_sequence)
            statistics.set_postfix(epi_length=num_steps, dict_size=self.D.shape[1], cumm_cost=np.sum(reward_sequence))

    def get_value_function(self, states):

        V = np.zeros(states.shape[1])
        for s in range(states.shape[1]):
            state = states[:,s][:,np.newaxis]
            traj = state
            V[s] = self.V_mu(state) + np.dot(self.k_(traj).T, self.diff_alpha_CV_D)

        return V