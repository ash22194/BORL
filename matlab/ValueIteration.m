function [V, policy] = ValueIteration(env, states, actions, alpha, gamma_, num_iter)
    V = zeros(size(states,2),1);                        % stateID -> V
    policy = randi(size(actions,2), size(states,2), 1)'; % stateID -> actionID
    for iter=1:1:num_iter
        % Value update loop
        for s = 1:1:size(states,2)
            env.set(states(:,s));
            [s_, r_, is_goal] = env.step(actions(:,policy(s)));
            stateIDFound = false;
            for i=1:1:size(states,2)
                if(states(:,i) == s_)
                    stateIDFound = true;
                    break;
                end
            end
            if (~stateIDFound)
                disp('Check state transition');
                return;
            end
            if (is_goal)
                V(i) = 0;
            end
            V(s) = V(s) + alpha*(r_ + gamma_*V(i) - V(s));
        end
        
        % Update policy
        for s=1:1:size(states,2)
            V_ = zeros(size(actions,2),1);
            for a = 1:1:size(actions,2)
                env.set(states(:,s));
                [s_, r_,~] = env.step(actions(:,a));
                stateIDFound = false;
                for i=1:1:size(states,2)
                    if(states(:,i) == s_)
                        stateIDFound = true;
                        break;
                    end
                end
                if (~stateIDFound)
                    disp('Check state transition');
                    return;
                end
                V_(a) = r_ + gamma_*V(i);
            end
            V_max = max(V_);
            a_max = find(V_==V_max);
            policy(s) = a_max(randperm(length(a_max),1));
        end
    end
end