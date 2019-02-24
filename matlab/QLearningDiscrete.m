classdef QLearningDiscrete < handle
    properties
        epsilon
        alpha
        gamma_
        env
        test_env
        is_reward
        q
    end
    methods
        function obj = QLearningDiscrete(env, test_env, gamma_, epsilon, alpha, is_reward)
            obj.epsilon = epsilon;
            obj.alpha = alpha;
            obj.gamma_ = gamma_;
            obj.env = env;
            obj.test_env = test_env;
            obj.is_reward = is_reward;
            obj.q = zeros(env.num_states,env.num_actions);
        end
        
        function update(obj,s,a,r,s_)
            if (obj.is_reward)
                obj.q(s,a) = obj.q(s,a) + obj.alpha*(r + obj.gamma_*max(obj.q(s_,:)) - obj.q(s,a));
            else
                obj.q(s,a) = obj.q(s,a) + obj.alpha*(r + obj.gamma_*min(obj.q(s_,:)) - obj.q(s,a));
            end
        end
    
        function train(obj, num_episodes, max_length_episode, test_every, debug_)
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
                end
                if ((mod(e,test_every)==0) && (debug_))
                    [R_avg, num_terms] = obj.test(10,max_length_episode);
                    disp(strcat('Episode ',int2str(e),', Average Reward :', string(R_avg), ', Terminations :', num2str(num_terms)));
                end
            end
        end
        
        function train_fixedstarts(obj, start_states, num_episodes, max_length_episode, test_every, debug_)
            for e = 1:1:num_episodes
                s = start_states(:,(mod(e,size(start_states,2))+1));
                obj.env.set(s);
                s = (mod(e,size(start_states,2))+1);
                is_terminal = false;
                num_steps = 0;
                while((~is_terminal) && (num_steps<max_length_episode))
                    num_steps = num_steps + 1;
                    a = obj.select_epsilon_greedy_action(s);
                    [s_, r, is_terminal] = obj.env.step(a);
                    obj.update(s,a,r,s_);
                    s = s_;
                end
                if ((mod(e,test_every)==0) && (debug_))
                    [R_avg, num_terms] = obj.test(10,max_length_episode);
                    disp(strcat('Episode ',int2str(e),', Average Reward :', string(R_avg), ', Terminations :', num2str(num_terms)));
                end
            end
        end
                
        function [R_avg, num_terms] = test(obj, num_episodes, max_length_episode)
            total_R = 0;
            num_terms = 0;
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
                if (is_terminal) 
                    num_terms = num_terms + 1;
                end
                total_R = total_R + R;
            end
            R_avg = total_R/num_episodes;
        end
    
        function a = select_greedy_action(obj,s)
            if (obj.is_reward)
                q_max = max(obj.q(s,:));
            else
                q_max = min(obj.q(s,:));
            end
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
    
        function policy = get_policy(obj)
            policy = zeros(obj.env.num_states,1);
            for s =1:1:obj.env.num_states
                policy(s) = obj.select_greedy_action(s);
            end
        end
    
        function V = get_value_function(obj)
            V = zeros(obj.env.num_states,1);
            if (obj.is_reward)
                for s =1:1:obj.env.num_states
                    V(s) = max(obj.q(s,:));
                end
            else
                for s =1:1:obj.env.num_states
                    V(s) = min(obj.q(s,:));
                    if (V(s)>150)
                        disp('Hold!');
                    end
                end
            end
        end
    end
    
end