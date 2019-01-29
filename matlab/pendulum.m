classdef pendulum < handle
	properties
	dt
	m
    l
    b
    g
	x_limits
	num_points_x
	x_dot_limits
	num_points_x_dot
    num_states
    num_actions
	Q
	R
	x
	goal
	end
	
	methods 
		function p = pendulum(m, l, b, g, dt, x_limits, num_points_x, x_dot_limits, num_points_x_dot, num_points_u, Q, R, goal)
			p.dt = dt;
			p.m = m;
            p.l = l;
            p.b = b;
            p.g = g;
			p.x_limits = x_limits;
			p.x_dot_limits = x_dot_limits;
			p.num_points_x = num_points_x;
			p.num_points_x_dot = num_points_x_dot;
            p.num_states = num_points_x*num_points_x_dot;
            p.num_actions = num_points_u;
			p.Q = Q;
			p.R = R;
			p.goal = goal;
		end

		function [x, r, is_goal] = step(p, u)
			x = zeros(2,1);
            %  u = u/p.m;
			x(1) = p.x(1) + p.x(2)*p.dt;
			x(2) = p.x(2) + (u - p.m*p.g*p.l*sin(p.x(1)) - p.b*p.x(2))/(p.m*p.l^2)*p.dt;
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
        
        function [s_, is_goal] = dynamics(p, s, a)
            s_ = zeros(size(s));
            s_(:,1) = s(:,1) + s(:,2)*p.dt;
            s_(:,2) = s(:,2) + (a - p.m*p.l*p.g*sin(s(:,1)) - p.b*s(:,2))/(p.m*p.l^2)*p.dt;
            
            s_(s_(:,1) > p.x_limits(2),1) = s_(s_(:,1) > p.x_limits(2),1) - p.x_limits(2);
			s_(s_(:,1) < p.x_limits(1),1) = s_(s_(:,1) < p.x_limits(1),1) + p.x_limits(2);
			s_(s_(:,2) > p.x_dot_limits(2),2) = p.x_dot_limits(2);
			s_(s_(:,2) < p.x_dot_limits(1),2) = p.x_dot_limits(1);
            
            dx = (p.x_limits(2)-p.x_limits(1))/(p.num_points_x - 1); 
    		dx_dot = (p.x_dot_limits(2)-p.x_dot_limits(1))/(p.num_points_x_dot - 1);
            is_goal = (abs(s(:,1)-p.goal(1)) < 0.5*dx) & (abs(s(:,2)-p.goal(2)) < 0.5*dx_dot);
            s_(is_goal,:) = s(is_goal,:);
        end
        
        function r = cost(p, s_, a)
            r = (p.R*a.^2 + sum(((s_ - repmat(p.goal',size(s_,1),1))*p.Q).*(s_ - repmat(p.goal',size(s_,1),1)),2))*p.dt;
        end
	end
end