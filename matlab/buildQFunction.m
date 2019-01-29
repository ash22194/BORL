function Q = buildQFunction (states, actions, dynamics, reward, V, gamma_)
    
    Q = zeros(size(states,1), size(actions,1));
    for a = 1:1:size(actions,1)
        [states_, is_goal] = dynamics(states, repmat(actions(a,:), size(states,1), 1));
        r = reward(states_, repmat(actions(a,:), size(states,1), 1));
        r(is_goal) = 0;
        Q(:,a) = r + gamma_*V(states_);
    end
    
end