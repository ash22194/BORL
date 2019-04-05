import numpy as np
from tqdm import trange

def ValueIterationSwingUp(env, gamma, x_grid, x_dot_grid, u_limits, num_iterations, tol=10**(-4)):

    states = np.mgrid[x_grid, x_dot_grid]
    states = np.concatenate((np.reshape(states[0,:,:], (1,states.shape[1]*states.shape[2]))\
                    np.reshape(states[1,:,:], (1,states.shape[1]*states.shape[2]))), axis=1)
    valueFunction = np.zeros(states.shape[1],1)

    policy = u_limits[0] + (u_limits[1] - u_limits[0])*np.random.rand(1, states.shape[1])
    for i in trange(num_iterations):
        states_, rewards, are_terminal = env.dynamics(states, policy)


