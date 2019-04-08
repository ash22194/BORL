import numpy as np

def buildQfromV(V_interp, env, gamma, state_grid, action_grid):
    """
    V_interp    - Function handle that returns V values for input states
    env         - object of the environment class
    gamma       - discount factor
    state_grid  - tuple with grid-points in each dimension
    action_grid - ax(num_actions) array of actions
    """
    
    Q = np.zeros((action_grid.shape[1], state_grid.shape[1]))
    for a in range(action_grid.shape[1]):
        action = np.dot(action_grid[:,a][:,np.newaxis], np.ones((1, state_grid.shape[1])))
        states_next, rewards, are_terminal = env.dynamics(state_grid, action)
        Q[a,:] = rewards + gamma*V_interp(states_next)

    return Q

