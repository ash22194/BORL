classdef pendulum
	properties
	dt
	m
	x_limits
	num_points_x
	x_dot_limits
	num_points_x_dot
	Q
	R
	x
	goal
	end
	
	methods 
		function p = pendulum(m, dt, x_limits, num_points_x, x_dot_limits, num_points_x_dot, Q, R, goal)
			p.dt = dt;
			p.m = m;
			p.x_limits = x_limits;
			p.x_dot_limits = x_dot_limits;
			p.num_points_x = num_points_x;
			p.num_points_x_dot = num_points_x_dot;
			p.Q = Q;
			p.R = R;
			p.goal = goal;
		end

		function [p, r, is_goal] = dynamics(p, dt, u)
			x = zeros(2,1);
			x(1) = p.x(1) + p.x(2)*dt;
			x(2) = p.x(2) + (u - sin(p.x(1)) - p.x(2))*dt;
			p.x = x;
			r = p.R*u^2 + 
		end

		function step(u)

		end
	end
end