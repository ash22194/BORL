classdef pendulum_discrete < handle
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
    state_grid
    num_states
    state_visits
    action_grid
    num_actions
	Q
	R
	x
	goal
	end
	
	methods 
		function p = pendulum_discrete(m, l, b, g, dt, x_limits, num_points_x, x_dot_limits, num_points_x_dot, u_limits, num_points_u, Q, R, goal)
			p.dt = dt;
			p.m = m;
            p.l = l;
            p.b = b;
            p.g = g;
			
            p.x_limits = x_limits;
			p.x_dot_limits = x_dot_limits;
			p.num_points_x = num_points_x;
			p.num_points_x_dot = num_points_x_dot;
            dx = (x_limits(2)-x_limits(1))/(num_points_x - 1); 
            dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(num_points_x_dot - 1);
            [grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
            p.state_grid = [reshape(grid_x, num_points_x_dot*num_points_x, 1), reshape(grid_x_dot, num_points_x_dot*num_points_x, 1)]';
            p.num_states = num_points_x*num_points_x_dot;
            p.state_visits = zeros(p.num_states,1);
            
            du = (u_limits(2)-u_limits(1))/(num_points_u - 1);
            p.action_grid = u_limits(1):du:u_limits(2);
            p.num_actions = num_points_u;
			
            p.Q = Q;
			p.R = R;
			p.goal = goal;
		end

		function [x, r, is_goal] = step(p, u)
            u = p.action_grid(u);
			x = zeros(2,1);
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
            diff = p.state_grid - repmat(p.x,1,p.num_states);
            [~,x] = min(sum(diff.^2));
            p.x = p.state_grid(:,x);
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
            unvisited = find(p.state_visits==0);
            if (~isempty(unvisited))
                x = randperm(length(unvisited),1);
                x = unvisited(x);
            else
                x = randperm(p.num_states,1);
            end
            p.x = p.state_grid(:,x);
        end
        
        function is_terminal = set(p, s)
            s_ = zeros(2,1);
            s_(1) = s(1);
            s_(2) = s(2);
            p.x = s_;
            diff = p.state_grid - repmat(p.x,1,p.num_states);
            [~,i] = min(sum(diff.^2));
            p.x = p.state_grid(:,i);
            
            is_terminal = false;
            dx = (p.x_limits(2)-p.x_limits(1))/(p.num_points_x - 1); 
    		dx_dot = (p.x_dot_limits(2)-p.x_dot_limits(1))/(p.num_points_x_dot - 1);
    		if (abs(p.goal(1) - p.x(1))<dx && abs(p.goal(2)-p.x(2))<dx_dot)
    			is_terminal = true;
    		end
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