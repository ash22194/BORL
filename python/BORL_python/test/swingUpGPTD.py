import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.interpolate import interpn
from BORL_python.env.pendulum import pendulum
from BORL_python.value.ValueIteration import ValueIterationSwingUp
from BORL_python.value.GPTD import GPTD
from BORL_python.utils.kernels import SqExpArd

def main():

    """
    Initialize environments
    """
    m = 1
    mass_factor = 1.6
    l = 1
    length_factor = 1.6
    b = 0.15
    g = 9.81
    dt = 0.005
    goal = np.array([[np.pi],[0]])
    x_limits = np.array([0, 6.2832])
    numPointsx = 51
    dx = (x_limits[-1] - x_limits[0])/(numPointsx - 1)
    x_dot_limits = np.array([-6.5, 6.5])
    numPointsx_dot = 81
    dx_dot = (x_dot_limits[-1] - x_dot_limits[0])/(numPointsx_dot - 1)
    Q = np.array([[40, 0], [0, 0.02]])
    R = 0.02
    environment = pendulum(m, l, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)
    environment_target = pendulum(m*mass_factor, l*length_factor, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)

    """
    Learn an initial policy and value function
    """
    gamma = 0.99
    x_grid = np.linspace(x_limits[0], x_limits[1], numPointsx)
    x_dot_grid = np.linspace(x_dot_limits[0], x_dot_limits[1], numPointsx_dot)
    u_limits = np.array([-20,20])
    numPointsu = 31
    u_grid = np.linspace(u_limits[0], u_limits[1], numPointsu)
    num_iterations = 400

    policy_target, V_target = ValueIterationSwingUp(environment_target, gamma, x_grid, x_dot_grid, u_grid, num_iterations)
    policy_start, V_start = ValueIterationSwingUp(environment, gamma, x_grid, x_dot_grid, u_grid, num_iterations)
    # policy_start = np.zeros((numPointsx, numPointsx_dot))
    # policy_target = np.zeros((numPointsx, numPointsx_dot))
    # V_start = np.zeros((numPointsx, numPointsx_dot))
    # V_target = np.zeros((numPointsx, numPointsx_dot))

    V_target = np.reshape(V_target, (numPointsx, numPointsx_dot))
    V_start = np.reshape(V_start, (numPointsx, numPointsx_dot))
    policy_target = np.reshape(policy_target, (numPointsx, numPointsx_dot))
    policy_start = np.reshape(policy_start, (numPointsx, numPointsx_dot))

    """
    GPTD
    """
    sigma0 = 0.2
    sigmaf = 7.6156
    sigmal = np.array([[0.6345],[1.2656]])
    kernel = SqExpArd(sigmal, sigmaf)
    p_target = lambda s: interpn((x_grid, x_dot_grid), policy_target, s.T)[0]
    V_mu = lambda s: interpn((x_grid, x_dot_grid), V_start, s.T)[0]

    nu = (sigmaf**2)*(np.exp(-1)-0.36)
    max_episode_length = 1000
    num_episodes = 1000
    states = np.mgrid[x_grid[0]:(x_grid[-1]+dx):dx, x_dot_grid[0]:(x_dot_grid[-1] + dx_dot):dx_dot]
    states = np.concatenate((np.reshape(states[0,:,:], (1,states.shape[1]*states.shape[2])),\
                    np.reshape(states[1,:,:], (1,states.shape[1]*states.shape[2]))), axis=0)

    gptd = GPTD(environment_target, nu, sigma0, gamma, kernel, V_mu)
    gptd.build_posterior(p_target, num_episodes, max_episode_length)
    V_gptd = gptd.get_value_function(states)
    V_gptd = np.reshape(V_gptd, (numPointsx, numPointsx_dot))
    
    """
    Results
    """
    plt.subplot(2,1,1)
    plt.imshow(V_target.T, aspect='auto',\
        extent=(x_limits[0], x_limits[1], x_dot_limits[1], x_dot_limits[0]), origin='upper')
    plt.ylabel('theta-dot')
    plt.xlabel('theta')
    plt.title('Target')
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(np.reshape(V_gptd, (numPointsx, numPointsx_dot)).T, aspect='auto',\
        extent=(x_limits[0], x_limits[1], x_dot_limits[1], x_dot_limits[0]), origin='upper')
    plt.ylabel('theta-dot')
    plt.xlabel('theta')
    plt.title('Final')
    plt.colorbar()
    plt.savefig('GPTD.png')
    plt.show()

if __name__=='__main__':
    main()