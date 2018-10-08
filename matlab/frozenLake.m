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
			index = (s_(1)-1)*f.map_y + s_(2);
            is_goal = false;
            r_= 0;
            if (~isempty(find(f.holes==index,1)))
                r_ = -1;
                is_goal = true;
            elseif (~isempty(find(f.goal==index,1)))
                r_ = 1;
                is_goal = true;
            end
            s_ = index;
		end

		function s = reset(f)
            s = zeros(2,1);
			index = randperm(length(f.starts),1);
            index = f.starts(index);
            s(1) = ceil(index/f.map_y);
            s(2) = index - (s(1)-1)*f.map_y;
            f.s = s;
            s = index;
		end
	end
end