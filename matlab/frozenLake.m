classdef frozenLake < handle
	properties
    map
    map_x
    map_y
    num_states
    num_actions
    is_slippery
    starts
    holes
	goal
    state_count
    s
	end
	
	methods 
		function f = frozenLake(map, num_actions, is_slippery)
			f.map = map;
            f.map_x = size(map,2);
            f.map_y = size(map,1);
            f.num_states = f.map_x*f.map_y;
            f.num_actions = num_actions;
            f.is_slippery = is_slippery;
            f.starts = find(map=='S');
            f.goal = find(map=='G');
            f.holes = find(map=='H');
            f.state_count = zeros(f.num_states,1);
            f.reset();
		end

		function [s_, r_, is_goal] = step(f, u)
			% 1,2,3,4 - L,D,R,U
            s_ = f.s;
            if (f.is_slippery)
               r = rand();
               if (u==1)
                   if (r>0.9 && r<0.95)
                       u = 4;
                   elseif (r>0.95)
                       u = 2;
                   end
               elseif (u==2)
                   if (r>0.9 && r<0.95)
                       u = 1;
                   elseif (r>0.95)
                       u = 3;
                   end
               elseif (u==3)
                   if (r>0.9 && r<0.95)
                       u = 4;
                   elseif (r>0.95)
                       u = 2;
                   end
               elseif (u==4)
                   if (r>0.9 && r<0.95)
                       u = 1;
                   elseif (r>0.95)
                       u = 3;
                   end
               end
            end
            if (u==1)
                s_(1) = max(f.s(1) - 1,1);
            elseif (u==2)
                s_(2) = min(f.s(2) + 1,f.map_y);    
            elseif (u==3)
                s_(1) = min(f.s(1) + 1,f.map_x);
            elseif (u==4)
                s_(2) = max(f.s(2) - 1,1);
            end
			index_ = (s_(1)-1)*f.map_y + s_(2);
            index = (f.s(1)-1)*f.map_y + f.s(2);
            is_goal = false;
%             r_= -0.02;
            r_ = 0;
            if (~isempty(find(f.holes==index,1)) || ~isempty(find(f.goal==index,1)))
                s_ = index;
                is_goal = true;
            else
                if (~isempty(find(f.holes==index_,1)))
                    r_ = -1;
                    is_goal = true;
                elseif (~isempty(find(f.goal==index_,1)))
                    r_ = 1;
                    is_goal = true;
                end
                f.s = s_;
                s_ = index_;
            end
            f.state_count(s_,1) = f.state_count(s_,1) + 1; 
		end

		function s = reset(f)
            s = zeros(2,1);
            unvisited = find(f.state_count(f.starts,:) == 0);
			if (isempty(unvisited))
                index = randperm(length(f.starts),1);
            else
                index = randperm(length(unvisited),1);
                index = unvisited(index);
            end
            index = f.starts(index);
            s(1) = ceil(index/f.map_y);
            s(2) = index - (s(1)-1)*f.map_y;
            f.s = s;
            s = index;
            f.state_count(s,1) = f.state_count(s,1) + 1; 
        end
        
        function is_terminal = set(f,s)
            is_terminal = false;
            if (size(s,1)==1 && size(s,2)==1)
                s_ = zeros(2,1);
                s_(1) = ceil(s/f.map_y);
                s_(2) = s - (s_(1)-1)*f.map_y;
                f.s = s_;
                if (any(f.holes==s) || any(f.goal==s))
                    is_terminal = true;
                end
            elseif ((size(s,1)==2 && size(s,2)==1) || (size(s,1)==1 && size(s,2)==2))
                f.s = s;
                index = (s(1) - 1)*f.map_y + s(2);
                if (any(f.holes==index) || any(f.goal==index))
                    is_terminal = true;
                end
            else
                disp('Check state input');
            end
        end
	end
end