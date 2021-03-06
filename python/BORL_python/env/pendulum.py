import numpy as np
from ipdb import set_trace

class pendulum:
    def __init__(self, m, l, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R):
        self.m = m
        self.l = l
        self.b = b
        self.g = g
        self.dt = dt
        assert type(goal)==type(np.array([])) and goal.shape[0]==2 and goal.shape[1]==1, "Check goal!"
        self.goal = goal
        self.x_limits = x_limits
        self.dx = dx
        self.dx_dot = dx_dot
        self.x_dot_limits = x_dot_limits
        self.Q = Q
        self.R = R
        _ = self.reset()

    def reset(self):
        self.x = np.array([\
                  [self.x_limits[0] + (self.x_limits[1] - self.x_limits[0])*np.random.rand()],\
                  [self.x_dot_limits[0] + (self.x_dot_limits[1] - self.x_dot_limits[0])*np.random.rand()]\
                  ])

        return self.x

    def sample_states(self, n):
        """
        Returns n randomly sampled states
        """
        return np.concatenate((self.x_limits[0] + np.random.rand(1, n)*(self.x_limits[1] - self.x_limits[0]),\
                    self.x_dot_limits[0] + np.random.rand(1, n)*(self.x_dot_limits[1] - self.x_dot_limits[0])), axis=0)

    def step(self, u):

        x = np.zeros((2,1))
        x[0,0] = self.x[0,0] + self.x[1,0]*self.dt
        x[1,0] = self.x[1,0] + (u \
                      - self.m*self.g*self.l*np.sin(self.x[0,0])\
                      - self.b*self.x[1,0])/(self.m*self.l**2)*self.dt

        if (x[0,0] > self.x_limits[1]):
            x[0,0] = x[0,0] - self.x_limits[1]
        elif (x[0,0] < self.x_limits[0]):
            x[0,0] = x[0,0] + self.x_limits[1]

        if (x[1,0] > self.x_dot_limits[1]):
            x[1,0] = self.x_dot_limits[1]
        elif (x[1,0] < self.x_dot_limits[0]):
            x[1,0] = self.x_dot_limits[0]
        self.x = x

        r = (self.R*u**2 + np.dot((x - self.goal).T, np.dot(self.Q, x - self.goal)))*self.dt
        
        is_terminal = False
        if ((np.abs(self.goal[0,0] - x[0,0]) < self.dx) and (np.abs(self.goal[1,0] - x[1,0]) < self.dx_dot)):
            is_terminal = True

        return x, r, is_terminal

    def dynamics(self, states, actions):

        assert states.shape[0]==2, "states.shape[0] != 2"
        assert states.shape[1]==actions.shape[1], "states.shape[1] != actions.shape[1]"
        
        states_ = np.zeros(states.shape)
        states_[0,:] = states[0,:] + states[1,:]*self.dt
        states_[1,:] = states[1,:] + (actions - self.m*self.g*self.l*np.sin(states[0,:])\
                           - self.b*states[1,:])/(self.m*self.l**2)*self.dt

        states_[0, states_[0,:] > self.x_limits[1]] -= self.x_limits[1]
        states_[0, states_[0,:] < self.x_limits[0]] += self.x_limits[1]
        states_[1, states_[1,:] > self.x_dot_limits[1]] = self.x_dot_limits[1]
        states_[1, states_[1,:] < self.x_dot_limits[0]] = self.x_dot_limits[0]

        diff = states_ - np.repeat(self.goal, states_.shape[1], axis=1)
        rewards = self.dt*(np.power(actions, 2)*self.R + \
                           np.power(diff[0,:],2)*self.Q[0,0] + \
                           np.power(diff[1,:],2)*self.Q[1,1] + \
                           (self.Q[0,1] + self.Q[1,0])*diff[0,:]*diff[1,:])[0,:]

        are_terminal = np.logical_and(abs(states_[0,:] - self.goal[0,0]) < self.dx, \
                                      abs(states_[1,:] - self.goal[1,0]) < self.dx_dot)
        
        return states_, rewards, are_terminal

    def dynamics_continuous(self, states, policy):

        assert states.shape[0]==2, "states.shape[0] != 2"
        if (len(states.shape)==1):
            states = states[:,np.newaxis]
        
        states[0, states[0,:] > self.x_limits[1]] -= self.x_limits[1]
        states[0, states[0,:] < self.x_limits[0]] += self.x_limits[1]
        states[1, states[1,:] > (self.x_dot_limits[1]-0.001)] = self.x_dot_limits[1]
        states[1, states[1,:] < (self.x_dot_limits[0]+0.001)] = self.x_dot_limits[0]

        actions = policy(states.T)[np.newaxis,:]
        assert states.shape[1]==actions.shape[1], "states.shape[1] != actions.shape[1]"

        states_ = np.zeros(states.shape)
        states_[0,:] = states[1,:]
        states_[1,:] = (actions - self.m*self.g*self.l*np.sin(states[0,:])\
                           - self.b*states[1,:])/(self.m*self.l**2)
        
        return states_
        


