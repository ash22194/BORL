classdef QLearningDiscrete < handle
    properties
        epsilon
        alpha
        gamma_
        env
        test_env
        q
    end
    methods
        function obj = QLearningDiscrete(env, gamma_, epsilon, alpha)
            obj.epsilon = epsilon;
            obj.alpha = alpha;
            obj.gamma_ = gamma_;
            obj.env = env;
            obj.test_env = env;
            obj.q = zeros(env.num_states,env.num_actions);
        end
        
        function update(obj,s,a,r,s_)
            q_next = obj.q(s_,:)';
            obj.q(s,a) = obj.q(s,a) + obj.alpha*(r + obj.gamma_*max(q_next) - obj.q(s,a));
        end
    
        function train(obj, num_episodes, max_length_episode, test_every, debug)
            for e = 1:1:num_episodes
                s = obj.env.reset();
                is_terminal = false;
                num_steps = 0;
                while((~is_terminal) && (num_steps<max_length_episode))
                    num_steps = num_steps + 1;
                    a = obj.select_epsilon_greedy_action(s);
                    [s_, r, is_terminal] = obj.env.step(a);
                    obj.update(s,a,r,s_);
                    s = s_;
                    if ((mod(e,test_every)==0) && (debug))
                        R_avg = obj.test(10,max_length_episode);
                        disp(strcat('Episode ',int2str(e),', Average Reward ', string(R_avg)));
                    end
                end
            end
        end
                
        function R_avg = test(obj, num_episodes, max_length_episode)
            total_R = 0;
            for e = 1:1:num_episodes
                s = obj.test_env.reset();
                is_terminal = false;
                num_steps = 0;
                R = 0;
                while((~is_terminal) && (num_steps<max_length_episode))
                    num_steps = num_steps + 1;
                    a = obj.select_greedy_action(s);
                    [s_, r, is_terminal] = obj.test_env.step(a);
                    R = R + r;
                    s = s_;
                end
                total_R = total_R + R;
            R_avg = float(total_R)/num_episodes;
            end
        end
    
        function a = select_greedy_action(obj,s)
            q_max = max(obj.q(s,:));
            q_max_indices = find(obj.q(s,:)==q_max);
            a = q_max_indices(randperm(length(q_max_indices),1));
        end
    
        function a = select_epsilon_greedy_action(obj,s)
            if (rand()<obj.epsilon)
                a = randperm(obj.env.num_actions,1);
            else
                a = obj.select_greedy_action(s);
            end
        end
    
        function policy = get_policy(obj, states)
            policy = zeros(obj.env.num_states,1);
            for s =1:1:length(states)
                policy(s) = obj.select_greedy_action(s);
            end
        end
    
        function V = get_value_function(obj, states)
            V = zeros(obj.env.num_states,1);
            for s =1:1:length(states)
                V(s) = max(obj.q(s,:));
            end
        end
    end
    
end