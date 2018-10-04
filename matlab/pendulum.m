classdef pendulum < handle
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

		function [x, r, is_goal] = step(p, u)
			x = zeros(2,1);
            u = u/p.m;
			x(1) = p.x(1) + p.x(2)*p.dt;
			x(2) = p.x(2) + (u - sin(p.x(1)) - p.x(2))*p.dt;
			% Wrap around
			if x(1) > p.x_limits(2)
				x(1) = x(1) - p.x_limits(2);
			end
			if x(1) < p.x_limits(1)
				x(1) = x(1) + p.x_limits(2);
			end
			if x(2) > p.x_dot_limits(2)
				x(2) = p.x_dot_limits(2);
			end
			if x(2) < p.x_dot_limits(1)
				x(2) = p.x_dot_limits(1);
			end
			p.x = x;
			r = (p.R*u^2 + (p.x - p.goal)'*p.Q*(p.x - p.goal))*p.dt;

			% Check if goal is reached
			is_goal = false;
			dx = (p.x_limits(2)-p.x_limits(1))/(p.num_points_x - 1); 
    		dx_dot = (p.x_dot_limits(2)-p.x_dot_limits(1))/(p.num_points_x_dot - 1);
    		if (abs(p.goal(1) - p.x(1))<dx && abs(p.goal(2)-p.x(2))<dx_dot)
    			is_goal = true;
    		end
		end

		function x = reset(p)
			x = zeros(2,1);
			x(1) = p.x_limits(1) + (p.x_limits(2) - p.x_limits(1))*rand();
			x(2) = p.x_dot_limits(1) + (p.x_dot_limits(2) - p.x_dot_limits(1))*rand();
			p.x = x;
		end
	end
end