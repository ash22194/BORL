import numpy as np
from tqdm import trange
from scipy.interpolate import interpn

def ValueIterationSwingUp(env, gamma, x_grid, x_dot_grid, u_grid, num_iterations, update_policy_every_ith_iteration=1, tol=10**(-4)):

	numPointsx = (x_grid[-1] - x_grid[0])/(x_grid.shape[0]-1)					# Two dimensional state space
	numPointsx_dot = (x_dot_grid[-1] - x_dot_grid[0])/(x_dot_grid.shape[0]-1)
    states = np.mgrid[x_grid[0]:x_grid[-1]:numPointsx,\
                      x_dot_grid[0]:x_dot_grid[-1]:numPointsx_dot]
    states = np.concatenate((np.reshape(states[0,:,:], (1,states.shape[1]*states.shape[2])),\
                    np.reshape(states[1,:,:], (1,states.shape[1]*states.shape[2]))), axis=0)
    
    # Initialize value function and policy
    V = np.zeros(numPointsx, numPointsx_dot)
    policy = u_grid[0] + (u_grid[-1] - u_grid[0])*np.random.rand(1, states.shape[1]) # One-dimensional actions
    
    for i in trange(num_iterations):
        states_, rewards, are_terminal = env.dynamics(states, policy)
        V_ = np.reshape(interpn((x_grid, x_dot_grid), V, states_.T), (x_grid.shape[0], x_dot_grid.shape[0]))
        V = np.reshape(rewards, (x_grid.shape[0], x_dot_grid.shape[0])) + gamma*V_

        if ((i+1)%update_policy_every_ith_iteration==0):
        	# Update policy
        	Q = np.zeros((u_grid.shape[0], states.shape[1]))
        	for a in range(u_grid.shape[0]):
        		action = np.ones(states.shape[1])*u_grid[a]
        		states_, rewards, are_terminal = env.dynamics(states, action)
        		V_ = np.reshape(interpn((x_grid, x_dot_grid), V, states_.T), (x_grid.shape[0], x_dot_grid.shape[0]))
        		Q[a,:] = np.reshape(rewards, (x_grid.shape[0], x_dot_grid.shape[0])) + gamma*V_

        	policy = u_grid[np.argmin(Q,axis=0)] # minimize the 'cost'
