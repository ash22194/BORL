import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import ode
from BORL_python.env.pendulum import pendulum
from BORL_python.value.ValueIteration import ValueIterationSwingUp
from BORL_python.value.GPTD import GPTD
from BORL_python.utils.kernels import SqExpArd

def main():

    """
    Initialize environments
    """
    m = 1
    mass_factor = 2.1
    l = 1
    length_factor = 1.1
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
    R = 0.2
    test_policies = False
    environment = pendulum(m, l, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)
    environment_target = pendulum(m*mass_factor, l*length_factor, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)

    """
    Learn an initial policy and value function
    """
    gamma = 0.99
    x_grid = np.linspace(x_limits[0], x_limits[1], numPointsx)
    x_dot_grid = np.linspace(x_dot_limits[0], x_dot_limits[1], numPointsx_dot)
    u_limits = np.array([-15,15])
    numPointsu = 121
    u_grid = np.linspace(u_limits[0], u_limits[1], numPointsu)
    num_iterations = 600

    code_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = '../data/GPTD'
    data_dir = os.path.join(code_dir, data_dir)

    print('Value Iteration for target domain')
    target_file = 'data_m_%.2f_l_%.2f.pkl'%(m*mass_factor, l*length_factor)
    fileFound = False
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (file.endswith('.pkl') and file==target_file):
                fileFound = True
                print('Relevant pre-computed data found!')
                data = pkl.load(open(os.path.join(data_dir, target_file), 'rb'))
                policy_target = data[0]
                V_target = data[1]
    if (not fileFound):
        policy_target, V_target = ValueIterationSwingUp(environment_target, gamma, x_grid, x_dot_grid, u_grid, num_iterations)
        pkl.dump((policy_target, V_target), open(os.path.join(data_dir, target_file), 'wb'))

    print('Value Iteration in simulation')
    start_file = 'data_m_%.2f_l_%.2f.pkl'%(m, l)
    fileFound = False
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (file.endswith('.pkl') and file==start_file):
                fileFound = True
                print('Relevant pre-computed data found!')
                data = pkl.load(open(os.path.join(data_dir, start_file), 'rb'))
                policy_start = data[0]
                V_start = data[1]
    if (not fileFound):
        policy_start, V_start = ValueIterationSwingUp(environment, gamma, x_grid, x_dot_grid, u_grid, num_iterations)
        pkl.dump((policy_start, V_start), open(os.path.join(data_dir, start_file), 'wb'))

    # policy_start = np.zeros((numPointsx, numPointsx_dot))
    # policy_target = np.zeros((numPointsx, numPointsx_dot))
    # V_start = np.zeros((numPointsx, numPointsx_dot))
    # V_target = np.zeros((numPointsx, numPointsx_dot))

    V_target = np.reshape(V_target, (numPointsx, numPointsx_dot))
    V_start = np.reshape(V_start, (numPointsx, numPointsx_dot))
    policy_target = np.reshape(policy_target, (numPointsx, numPointsx_dot))
    policy_start = np.reshape(policy_start, (numPointsx, numPointsx_dot))

    """
    Test learned policies
    """
    if (test_policies):
        
        policy_start_ = RegularGridInterpolator((x_grid, x_dot_grid), policy_start)
        dyn_start = lambda t,s: environment_target.dynamics_continuous(s, policy_start_)
        int_start = ode(dyn_start).set_integrator('vode', method='bdf', with_jacobian=False)
        int_start.set_initial_value(np.array([[0], [0]]), 0)
        t_final = 10
        trajectory_start = np.empty((2, int(t_final/dt)))
        num_steps = 0
        while int_start.successful() and int_start.t<t_final:
            int_start.integrate(int_start.t+dt)
            trajectory_start[:, num_steps] = int_start.y[:,0]
            num_steps+=1

        trajectory_start = trajectory_start[:,0:num_steps]
        plt.plot(trajectory_start[0,:], trajectory_start[1,:])
        plt.scatter(np.pi, 0, c='red', marker='o')
        plt.xlabel('theta')
        plt.ylabel('theta-dot')
        plt.title('Bootstrapped Policy')
        plt.show()

        policy_target_ = RegularGridInterpolator((x_grid, x_dot_grid), policy_target)
        dyn_target = lambda t,s: environment_target.dynamics_continuous(s, policy_target_)
        int_target = ode(dyn_target).set_integrator('vode', method='bdf', with_jacobian=False)
        int_target.set_initial_value(np.array([[0], [0]]), 0)
        trajectory_target = np.empty((2, int(t_final/dt)))
        num_steps = 0
        while int_target.successful() and int_target.t<t_final:
            int_target.integrate(int_target.t+dt)
            trajectory_target[:, num_steps] = int_target.y[:,0]
            num_steps+=1

        trajectory_target = trajectory_target[:,0:num_steps]
        plt.plot(trajectory_target[0,:], trajectory_target[1,:])
        plt.scatter(np.pi, 0, c='red', marker='o')
        plt.xlabel('theta')
        plt.ylabel('theta-dot')
        plt.title('Target Policy')
        plt.show()

    """
    GPTD
    """
    sigma0 = 0.2
    sigmaf = 7.6156
    sigmal = np.array([[0.6345],[1.2656]], dtype=np.float64)
    kernel = SqExpArd(sigmal, sigmaf)
    p_target = lambda s: RegularGridInterpolator((x_grid, x_dot_grid), policy_target)(s.T)
    V_mu = lambda s: RegularGridInterpolator((x_grid, x_dot_grid), V_start)(s.T)

    nu = (np.exp(-1)-0.3)
    max_episode_length = 1000
    num_episodes = 600
    states = np.mgrid[x_grid[0]:(x_grid[-1]+dx):dx, x_dot_grid[0]:(x_dot_grid[-1] + dx_dot):dx_dot]
    states = np.concatenate((np.reshape(states[0,:,:], (1,states.shape[1]*states.shape[2])),\
                    np.reshape(states[1,:,:], (1,states.shape[1]*states.shape[2]))), axis=0)

    gptd = GPTD(environment_target, nu, sigma0, gamma, kernel, V_mu)
    print('GPTD.. ')
    gptd.build_posterior(p_target, num_episodes, max_episode_length)
    V_gptd = gptd.get_value_function(states)
    V_gptd = np.reshape(V_gptd, (numPointsx, numPointsx_dot))
    print('Initial mean error:%f'%np.mean(np.abs(V_target - V_start)))
    print('Final mean error:%f'%np.mean(np.abs(V_target - V_gptd)))
    set_trace()
    
    """
    Results
    """
    plt.subplot(3,1,1)
    plt.imshow(np.abs(V_target - V_start).T, aspect='auto',\
        extent=(x_limits[0], x_limits[1], x_dot_limits[1], x_dot_limits[0]), origin='upper')
    plt.ylabel('theta-dot')
    plt.xlabel('theta')
    plt.title('Initial Diff')
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.imshow(np.abs(V_target - V_gptd).T, aspect='auto',\
        extent=(x_limits[0], x_limits[1], x_dot_limits[1], x_dot_limits[0]), origin='upper')
    plt.ylabel('theta-dot')
    plt.xlabel('theta')
    plt.title('Final Diff')
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.scatter(gptd.D[0,:], gptd.D[1,:], marker='o', c='red')
    plt.xlim(x_limits[0], x_limits[1])
    plt.xlabel('theta')
    plt.ylim(x_dot_limits[0], x_dot_limits[1])
    plt.ylabel('theta-dot')
    plt.title('Dictionary Points')
    
    set_trace()
    resultDirName = 'GPTD_run'
    run = -1
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if (d.startswith(resultDirName)):
                extension = d.split(resultDirName)[-1]
                if (extension.isdigit() and int(extension)>=run):
                    run = int(extension)
    run += 1
    saveDirectory = os.path.join(data_dir, resultDirName + str(run))
    os.mkdir(saveDirectory)
    dl.dump_session(filename=os.path.join(saveDirectory, 'session_%d'%num_episodes))
    plt.savefig(os.path.join(saveDirectory,'V_Diff.png'))
    plt.show()
    set_trace()

if __name__=='__main__':
    main()