from tqdm import trange
import numpy as np
from ipdb import set_trace

class GPTD:
    def __init__(self, env, nu, sigma0, gamma, kernel, V_mu=[]):
        
        self.env = env
        self.nu = nu
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        if (not V_mu):
            V_mu = lambda s,a: 0    
        self.V_mu = V_mu
        self.D = np.array([[]])
        self.A = np.array([[]])
        self.V_D = np.array([[]])
        self.K_inv = np.array([[]])
        self.alpha_ = np.array([[]])
        self.C_ = np.array([[]])

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
            et = (ktt - np.dot(k_t.T, at))
            delk_t_1 = k_t_1 - self.gamma*k_t

            if (et - self.nu) > 10**(-4):
                self.D = np.concatenate((self.D, trajt), axis=1)
                self.V_D = np.concatenate((self.V_D, self.V_mu(state_sequence[:,i+1])), axis=0)

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

                self.A = np.zeros((self.A.shape[0]+1, self.A.shape[1]))
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
            if ((state[0,0] > self.env.x_limits[1] or state[0,0] < self.env.x_limits[0]) or \
                (state[1,0] > self.env.x_dot_limits[1] or state[1,0] < self.env.x_dot_limits[0])):
                set_trace()
            V[s] = self.V_mu(state) + np.dot(self.k_(traj).T, self.diff_alpha_CV_D)

        return V