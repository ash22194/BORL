import numpy as np
import matplotlib.pyplot as plt
from BORL_python.env.pendulum import pendulum
from BORL_python.value.ValueIteration import ValueIterationSwingUp

def main():

    m = 1
    l = 1
    b = 0.15
    g = 9.81
    dt = 0.005
    goal = np.array([[np.pi],[0]])
    x_limits = np.array([0, 2*np.pi])
    numPointsx = 51
    dx = (x_limits[-1] - x_limits[0])/(numPointsx - 1)
    x_dot_limits = np.array([-6.5, 6.5])
    numPointsx_dot = 81
    dx_dot = (x_dot_limits[-1] - x_dot_limits[0])/(numPointsx_dot - 1)
    Q = np.array([[40, 0], [0, 0.02]])
    R = 0.02
    environment = pendulum(m, l, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)

    gamma = 0.99
    x_grid = np.linspace(x_limits[0], x_limits[1], numPointsx)
    x_dot_grid = np.linspace(x_dot_limits[0], x_dot_limits[1], numPointsx_dot)
    u_limits = np.array([-15,15])
    numPointsu = 31
    u_grid = np.linspace(u_limits[0], u_limits[1], numPointsu)
    num_iterations = 400
    policy, V = ValueIterationSwingUp(environment, gamma, x_grid, x_dot_grid, u_grid, num_iterations)

    plt.imshow(np.reshape(V, (numPointsx, numPointsx_dot)).T, aspect='auto',\
        extent=(x_limits[0], x_limits[1], x_dot_limits[1], x_dot_limits[0]), origin='upper')
    plt.ylabel('theta-dot')
    plt.xlabel('theta')
    plt.title('Value Function')
    plt.colorbar()
    plt.show()

if __name__=='__main__':
    main()