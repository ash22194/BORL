function [policy, V] = ValueIterationSwingUp(m, l, b, g, numPointsx_, numPointsx_dot_, numPointsu_, x_limits_, x_dot_limits_, u_limits_, Q_, R_, goal_, start_, gamma_, gtol_, dt_, maxiter_, max_policy_iter_, visualize, test_policy)

    if (visualize)
        clear;
        clc;
        close all;
    end

    %% Parameters
    dt = dt_;
    numPointsx = numPointsx_;
    numPointsx_dot = numPointsx_dot_;
    numPointsu = numPointsu_;
    x_limits = x_limits_;  % Don't change limits for x as the values are wrapped around when an 
                           % action takes the system beyond 2pi or below 0 and any other
                           % limts physically don't make sense under this implementation
                           % If you change numPointsx ensure that goal x is a
                           % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

    x_dot_limits = x_dot_limits_; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                                  % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
    Q = Q_;
    u_limits = u_limits_;
    R = R_;
    dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
    dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
    du = (u_limits(2) - u_limits(1))/(numPointsu - 1);
    gtol = gtol_;
    goal = goal_; 
    start = start_;
    gamma = gamma_;
    maxiter = maxiter_;
    max_policy_iter = max_policy_iter_;

    %% Initialize
    [grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
    goal_grid = [0,0];
    goal_grid(1) = find(grid_x(:,1)==goal(1));
    goal_grid(2) = find(grid_x_dot(1,:)==goal(2));
    possibleActions = u_limits(1):du:u_limits(2);
    policy = possibleActions(randi(length(possibleActions),size(grid_x,1),size(grid_x_dot,2)));
    
    %% Policy Iteration
    % Since the grid is fixed, the position cost (x-xd)'Q(x-xd) for all states doesn't
    % change, the only part of instantaneous cost that changes is u'Ru where u
    % depends on policy
    % positionCost = zeros(size(grid_x_dot,1),size(grid_x,2));
    % for i = 1:1:length(grid_x(1,:))
    %    for j = 1:1:length(grid_x_dot(:,1))
    %       relP = [grid_x(j,i),grid_x_dot(j,i)] - goal;
    %       positionCost(j,i) = (relP)*Q*relP';
    %    end
    % end
        
    iter = 0;
    G_ = zeros(size(grid_x,1),size(grid_x_dot,2));
        
    % Compute value function for current policy
    while (iter < maxiter)
        iter = iter + 1;
        policy_iter = 0;
        G = 10*ones(size(grid_x,1),size(grid_x_dot,2));
        while ((max(max(abs(G_ - G))) > gtol) && (policy_iter < max_policy_iter))
            % Iterate to estimate value function
            % Euler integration
            G = G_;
            policy_iter = policy_iter + 1;
            grid_x_ = grid_x + dt*grid_x_dot;
            grid_x_dot_ = grid_x_dot + dt*(policy - b*grid_x_dot - m*g*l*sin(grid_x))/(m*l^2);
            % Check for wrap around in x
            greaterThan = grid_x_ > x_limits(2);
            grid_x_ = grid_x_ - x_limits(2)*greaterThan;
            lessThan = grid_x_ < x_limits(1);
            grid_x_ = grid_x_ + x_limits(2)*lessThan;

            greaterThan = grid_x_dot_ > x_dot_limits(2);
            grid_x_dot_(greaterThan) = x_dot_limits(2);
            lessThan = grid_x_dot_ < x_dot_limits(1);
            grid_x_dot_(lessThan) = x_dot_limits(1);

            % If x_dot is out of bounds assign Gnext i.e. G(x+udt) = G(x) similar to grid
            % world i.e. if the agent takes an action that takes it out of the
            % grid, it lands up in the same state it took the action from
            Gnext = interpn(grid_x,grid_x_dot,G,grid_x_,grid_x_dot_);
            % outOfBounds = isnan(Gnext);
            % x_dot_outOfBounds = (grid_x_dot_ > x_dot_limits(2)) | (grid_x_dot_ < x_dot_limits(1));
            % if (prod(prod(x_dot_outOfBounds == outOfBounds))~=1)
            %    fprintf('Interp messed up!\n');
            % end
            % Gnext(outOfBounds) = G(outOfBounds);

            % Update value function
            Gnext(goal_grid(1),goal_grid(2)) = 0;
            pos_cost = (grid_x_ - goal(1));
            pos_dot_cost = (grid_x_dot_ - goal(2));
            positionCost = Q(1,1)*pos_cost.^2 + (Q(2,1) + Q(1,2))*(pos_cost.*pos_dot_cost) + Q(2,2)*pos_dot_cost.^2;
            G_ = (positionCost + R*policy.^2)*dt + gamma*Gnext;
            G_(goal_grid(1),goal_grid(2)) = 0;
        end
        % Uncomment to visualize the value function as it updates
        % imagesc([x_limits(1),x_limits(2)],[x_dot_limits(1),x_dot_limits(2)],G_);
        % colorbar;
        % axis equal;
        % axis([x_limits(1),x_limits(2),x_dot_limits(1),x_dot_limits(2)]);
        % pause(0.1)

        % Update policy
        % Compute Q(x,u) values g(x,u) + G(x+udt) for different actions when value
        % function G has converged and greedily update the policy i.e. choose
        % action u* at state x such that u* = argmin Q(x,u)
        possibleG = zeros([size(grid_x,1),size(grid_x_dot,2),length(possibleActions)]);
        for i = 1:1:length(possibleActions)
            policy_ = possibleActions(i)*ones(size(grid_x,1),size(grid_x_dot,2));
            grid_x_ = grid_x + dt*grid_x_dot;
            grid_x_dot_ = grid_x_dot + dt*(policy_ - b*grid_x_dot - m*g*l*sin(grid_x))/(m*l^2);

            % Check for wrap around in x
            greaterThan = grid_x_ > x_limits(2);
            grid_x_ = grid_x_ - x_limits(2)*greaterThan;
            lessThan = grid_x_ < x_limits(1);
            grid_x_ = grid_x_ + x_limits(2)*lessThan;
            
            greaterThan = grid_x_dot_ > x_dot_limits(2);
            grid_x_dot_(greaterThan) = x_dot_limits(2);
            lessThan = grid_x_dot_ < x_dot_limits(1);
            grid_x_dot_(lessThan) = x_dot_limits(1);
            
            % If x_dot is out of bounds assign Gnext i.e. G(x+udt) = G(x) similar to grid
            % world i.e. if the agent takes an action that takes it out of the
            % grid, it lands up in the same state it took action from
            % x_dot_outOfBounds = (grid_x_dot_ > x_dot_limits(2)) | (grid_x_dot_ < x_dot_limits(1));
            Gnext_ = interpn(grid_x,grid_x_dot,G_,grid_x_,grid_x_dot_);
            % outOfBounds = isnan(Gnext_);
            % if (prod(prod(x_dot_outOfBounds == outOfBounds))~=1)
            %   fprintf('Interp messed up!\n');
            % end
            % Gnext_(outOfBounds) = G_(outOfBounds);

            Gnext_(goal_grid(1),goal_grid(2)) = 0;
            pos_cost = (grid_x_ - goal(1));
            pos_dot_cost = (grid_x_dot_ - goal(2));
            positionCost = Q(1,1)*pos_cost.^2 + (Q(2,1) + Q(1,2))*(pos_cost.*pos_dot_cost) + Q(2,2)*pos_dot_cost.^2;
            possibleG(:,:,i) = (positionCost + R*policy_.^2)*dt + gamma*Gnext_;
            possibleG(goal_grid(1),goal_grid(2),i) = 0;
        end
        [~,I] = min(possibleG,[],3);
        policyNew = possibleActions(I);
        policyNew(goal_grid(1),goal_grid(2)) = 0;
        if (prod(prod(policy == policyNew))==1)
%             fprintf('Policy converged!\n')
%             break;
        end
        policy = policyNew;
        if (visualize)
            imagesc([x_limits(1),x_limits(2)],[x_dot_limits(1),x_dot_limits(2)],policy);
            colorbar;
            axis equal;
            axis([x_limits(1),x_limits(2),x_dot_limits(1),x_dot_limits(2)]);
            xlabel('theta'); ylabel('theta-dot');
            pause(0.1);
        end
    end

    %% Return the value function
    V = G;
    
    %% Test policy
    if (test_policy)
%         start = start';
        tspan = [0,200];
        opts = odeset('RelTol',1e-8,'AbsTol',1e-8);
        [t,y] = ode45(@(t,y) gridBasedSwingUp(t,y,m,l,b,g,grid_x,grid_x_dot,policy,x_limits,x_dot_limits),tspan,start,opts);
        figure;
        plot(y(:,1),y(:,2)); xlabel('theta'); ylabel('theta-dot');
        hold on;
        scatter(start(1),start(2),20,[0,1,0],'filled');
        text(start(1),start(2),'Start','horizontal','left','vertical','bottom');
        scatter(goal(1),goal(2),20,[1,0,0],'filled');
        text(goal(1),goal(2),'Goal','horizontal','left','vertical','bottom');
        title('VI Policy');
        hold off;
    end
end