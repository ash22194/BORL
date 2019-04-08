import numpy as np
import gym

class QLearningDiscrete:
    def __init__(self, env, gamma, epsilon, alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        assert type(env.action_space)==gym.spaces.discrete.Discrete, 'Only for discrete action spaces'
        assert type(env.observation_space)==gym.spaces.discrete.Discrete, 'Only for discrete state spaces'
        self.env = env
        self.test_env = env
        self.q = dict()
        
    def update(self,s,a,r,s_):
        if (self.q.get(s_)==None):
            self.q[s_] = dict()
            for a_ in range(self.env.action_space.n):
                self.q[s_][a_] = 0
        q_next = [self.q[s_][a_] for a_ in range(self.env.action_space.n)]
        self.q[s][a] += self.alpha*(r + self.gamma*np.max(q_next) - self.q[s][a]) 
    
    def train(self, num_episodes, max_length_episode, test_every, debug):
        for e in range(num_episodes):
            s = self.env.reset()
            is_terminal = False
            num_steps = 0
            while((not is_terminal) and (num_steps<max_length_episode)):
                num_steps+=1
                a = self.select_epsilon_greedy_action(s)
                s_, r, is_terminal, debug_info = self.env.step(a)
                self.update(s,a,r,s_)
                s = s_
            if ((e%test_every==0) and (debug)):
                R_avg, total_success = self.test(10,max_length_episode)
                print('Episode %d, Average Reward %f, Success : %d'%(e,R_avg,total_success))
                
    def test(self, num_episodes, max_length_episode, show=False):
        total_R = 0
        total_success = 0
        for e in range(num_episodes):
            s = self.test_env.reset()
            is_terminal = False
            num_steps = 0
            R = 0
            while((not is_terminal) and (num_steps<max_length_episode)):
                num_steps+=1
                a = self.select_greedy_action(s)
                s_, r, is_terminal, debug_info = self.test_env.step(a)
                R+=r
                s = s_
                if (show):
                    self.test_env.render()
                
                if ('goal' in debug_info):
                    total_success+=1
            total_R+=R
        return float(total_R)/num_episodes, total_success
    
    def select_greedy_action(self,s):
        if (self.q.get(s)==None):
            return self.env.action_space.sample()
        q = [self.q[s][a] for a in range(self.env.action_space.n)]
        q_max = np.max(q)
        q_max_indices = [index for index in range(self.env.action_space.n) if q[index]==q_max]
        return np.random.choice(q_max_indices)
    
    def select_epsilon_greedy_action(self,s):
        if (self.q.get(s)==None):
            self.q[s] = dict()
            for a in range(self.env.action_space.n):
                self.q[s][a] = 0
        if (np.random.rand()<self.epsilon):
            return self.test_env.action_space.sample()
        q = [self.q[s][a] for a in range(self.env.action_space.n)]
        q_max = np.max(q)
        q_max_indices = [index for index in range(self.env.action_space.n) if q[index]==q_max]
        return np.random.choice(q_max_indices)
    
    def get_policy(self, states):
        policy = dict()
        for s in states:
            policy[s] = self.select_greedy_action(s)
        return policy
    
    def get_value_function(self, states):
        V = dict()
        for s in states:
            if (self.q.get(s)==None):
                V[s] = 0
            else:
                q = [self.q[s][a] for a in range(self.env.action_space.n)]
                V[s] = np.max(q)
        return V
