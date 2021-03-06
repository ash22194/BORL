from tqdm import trange
import numpy as np 
from ipdb import set_trace
class GPSARSA_rewardBased:
    def __init__(self, env, u_limits, nu, sigma0, gamma, epsilon, kernel, Q_mu=[]):
        
        self.env = env
        self.actions = u_limits
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        if (not Q_mu):
            Q_mu = lambda s,a: np.zeros(s.shape[1]) 
        self.Q_mu = Q_mu
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

    def update(self, state_sequence, action_sequence, Q_sequence, reward_sequence):
        """
        Update GP after observing states (state_sequence), 
        actions (action_sequence) and rewards (reward_sequence)
        """

        for i in range(reward_sequence.shape[0]):
            trajt_1 = np.concatenate((state_sequence[:,i], action_sequence[:,i]))[:,np.newaxis]
            trajt = np.concatenate((state_sequence[:,i+1], action_sequence[:,i+1]))[:,np.newaxis]
            qt_1 = Q_sequence[:,i]
            qt = Q_sequence[:,i+1]
            rt_1 = reward_sequence[i]

            k_t_1 = self.k_(trajt_1)
            k_t = self.k_(trajt)
            ktt = self.kernel(trajt, trajt)
            at = np.dot(self.K_inv, k_t)
            et = (ktt - np.dot(k_t.T, at))
            delk_t_1 = k_t_1 - self.gamma*k_t

            if abs((qt_1 - self.gamma*qt) - rt_1) > 0.1*abs(rt_1):
                self.D = np.concatenate((self.D, trajt), axis=1)
                self.Q_D = np.concatenate((self.Q_D, self.Q_mu(state_sequence[:,i+1], action_sequence[:,i+1])[:,np.newaxis]), axis=0)

                at_by_et = at/et
                self.K_inv = np.concatenate((self.K_inv + np.dot(at, at.T)/et, -at_by_et), axis=1)
                self.K_inv = np.concatenate((self.K_inv, np.concatenate((-at_by_et.T, 1/et), axis=1)), axis=0)

                c_t = np.dot(self.C_, delk_t_1) - self.A

                delktt = np.dot(self.A.T, delk_t_1 - self.gamma*k_t) + (self.gamma**2)*ktt
                s_t = self.sigma0**2 + delktt - np.dot(delk_t_1.T, np.dot(self.C_, delk_t_1))

                diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - rt_1
                self.alpha_ = np.concatenate((self.alpha_ + c_t/s_t*diff_r, self.gamma/s_t*diff_r), axis=0)

                gc_t_by_s_t = (self.gamma/s_t)*c_t
                self.C_ = np.concatenate((self.C_ + np.dot(c_t, c_t.T)/s_t, gc_t_by_s_t), axis=1) 
                self.C_ = np.concatenate((self.C_, np.concatenate((gc_t_by_s_t.T, self.gamma**2/s_t), axis=1)), axis=0)

                self.A = np.zeros((self.A.shape[0]+1, self.A.shape[1]), dtype=np.float64, order='C')
                self.A[-1, 0] = 1

            else:

                ct = np.dot(self.C_, delk_t_1) - (self.A - self.gamma*at)
                st = self.sigma0**2 - np.dot(ct.T, delk_t_1)

                diff_r = np.dot(delk_t_1.T, self.alpha_)[0,0] - rt_1
                self.alpha_ = self.alpha_ + ct/st*diff_r

                self.C_ = self.C_ + np.dot(ct, ct.T)/st

                self.A = at

            assert (not np.isnan(self.alpha_).any()), "Check alpha for NaN values"

        self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

    def select_action(self, state, epsilon=0):
        """
        Select action epsilon-greedily
        Return action and corresponding Q value
        """

        num_actions_to_sample = 10
        actions = np.repeat(self.actions[:,0][:,np.newaxis], num_actions_to_sample, axis=1) +\
                  (np.random.rand(self.actions.shape[0], num_actions_to_sample) *\
                   np.repeat((self.actions[:,1] - self.actions[:,0])[:,np.newaxis], num_actions_to_sample, axis=1))
        action_explore = self.actions[:,0] +\
                  np.random.rand(self.actions.shape[0])*(self.actions[:,1] - self.actions[:,0])
        action_explore = action_explore[:, np.newaxis]            
        explore = np.random.rand()

        if (self.D.shape[1]==0):
            Q = self.Q_mu(state, action_explore)
            action = action_explore

        else:
            Q = np.zeros((num_actions_to_sample, 1), dtype=np.float64, order='C')
            for a in range(num_actions_to_sample):
                action = actions[:,a][:,np.newaxis]
                traj = np.concatenate((state, action), axis=0)
                Q[a,0] = self.Q_mu(state, action) + np.dot(self.k_(traj).T, self.diff_alpha_CQ_D)

            action_exploit = np.argmin(Q, axis=0)
            Q_exploit = Q[action_exploit, 0]
            action_exploit = actions[:, action_exploit][:,np.newaxis]

            Q_explore = self.Q_mu(state, action_explore) +\
                np.dot(self.k_(np.concatenate((state, action_explore), axis=0)).T, self.diff_alpha_CQ_D)[0,0]

            action = (explore<epsilon)*action_explore + (explore>epsilon)*action_exploit
            Q = (explore<epsilon)*Q_explore + (explore>epsilon)*Q_exploit

        return Q, action


    def build_policy_monte_carlo(self, num_episodes, max_episode_length):
        """
        """

        statistics = trange(num_episodes)

        for e in statistics:
            is_terminal = False
            num_steps = 0
            state = self.env.reset()
            Q, action = self.select_action(state, self.epsilon)

            state_sequence = np.zeros((state.shape[0], max_episode_length+1), dtype=np.float64, order='C')
            state_sequence[:, 0] = state[:,0]
            action_sequence = np.zeros((action.shape[0], max_episode_length+1), dtype=np.float64, order='C')
            action_sequence[:, 0] = action[:,0]
            Q_sequence = np.zeros((1, max_episode_length+1), dtype=np.float64, order='C')
            Q_sequence[:, 0] = Q[:,0]
            reward_sequence = np.zeros(max_episode_length, dtype=np.float64, order='C')
            
            while ((num_steps < max_episode_length) and (not is_terminal)):
                num_steps+=1
                state, reward, is_terminal = self.env.step(action)
                Q, action = self.select_action(state, self.epsilon)

                state_sequence[:, num_steps] = state[:,0]
                action_sequence[:, num_steps] = action[:,0]
                Q_sequence[:, num_steps] = Q[:,0]
                reward_sequence[num_steps-1] = reward

            state_sequence = state_sequence[:, 0:(num_steps+1)]
            action_sequence = action_sequence[:, 0:(num_steps+1)]
            Q_sequence = Q_sequence[:, 0:(num_steps+1)]
            reward_sequence = reward_sequence[0:num_steps]

            if (self.D.shape[1]==0):

                traj = np.concatenate((state_sequence[:,0], action_sequence[:,0]))[:,np.newaxis]
                self.D = traj
                self.Q_D = self.Q_mu(state_sequence[:,0], action_sequence[:,0])[:,np.newaxis]
                self.K_inv = 1/self.kernel(traj, traj)
                self.A = np.array([[1]], dtype=np.float64, order='C')
                self.alpha_ = np.array([[0]], dtype=np.float64, order='C')
                self.C_= np.array([[0]], dtype=np.float64, order='C')
                self.diff_alpha_CQ_D = self.alpha_ - np.dot(self.C_, self.Q_D)

            self.update(state_sequence, action_sequence, Q_sequence, reward_sequence)
            statistics.set_postfix(epi_length=num_steps, dict_size=self.D.shape[1], cumm_reward=np.sum(reward_sequence))

    def get_value_function(self, states):
        
        V = np.zeros((states.shape[1],1), dtype=np.float64, order='C')
        for s in range(states.shape[1]):
            Q, _ = self.select_action(states[:,s][:,np.newaxis])
            V[s,0] = Q

        return V

    def get_q_value(self, state, action):

        return self.Q_mu(state, action) + \
                    np.dot(self.k_(np.concatenate((state, action), axis=0)).T, self.diff_alpha_CQ_D)