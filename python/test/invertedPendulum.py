import numpy as np
from env.pendulum import pendulum
from value.ValueIteration import ValueIterationSwingUp

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

	env = pendulum(m, l, b, g, dt, goal, x_limits, dx, x_dot_limits, dx_dot, Q, R)






if __name__=='__main__':
	main()