classdef GPSARSA_episodic < handle

    properties (SetAccess='private')
        env
        Q_bootstrapped;
        actions
        discra % discrete actions?
        epsilon
        nu
        gamma_
        is_reward
        sigma_s
        sigma_a
        sigma_q
        sigma0
        sigmaf
        D
        Q_D
        A
        H_
        K_inv
        alpha_
        C_
        c_
        d
        s
    end
    
    methods
        function gpsarsa = GPSARSA_episodic(env, Q_bootstrapped, actions, discra, epsilon, nu, sigma_s, sigma_a, sigma_q, sigma0, sigmaf, gamma_, is_reward)
            gpsarsa.env = env;
            gpsarsa.Q_bootstrapped = Q_bootstrapped;
            gpsarsa.actions = actions;
            gpsarsa.discra = discra;
            gpsarsa.epsilon = epsilon;
            gpsarsa.nu = nu; % TODO: Compute nu based on kernel and sigma(s)
            gpsarsa.sigma0 = sigma0;
            gpsarsa.sigmaf = sigmaf;
            gpsarsa.sigma_s = sigma_s;
            gpsarsa.sigma_a = sigma_a;
            gpsarsa.sigma_q = sigma_q;
            gpsarsa.gamma_ = gamma_;
            gpsarsa.is_reward = is_reward;
            
            gpsarsa.D = [];
            gpsarsa.Q_D = [];
            gpsarsa.A = [];
            gpsarsa.H_ = [];
            gpsarsa.K_inv = [];
            gpsarsa.alpha_ = [];
            gpsarsa.C_ = [];
        end
        
        function k = kernel_q(gpsarsa,x,y)
            % x,y are nx1 inputs that represent (states, actions)
            k = gpsarsa.sigmaf^2*exp(-sum(((x-y)./repmat([gpsarsa.sigma_q],1,size(x,2))).^2,1)/2);
        end
        
        function k = kernel(gpsarsa,x,y)
            % x,y are nx1 inputs that represent (states, actions)
%             k = gpsarsa.sigmaf^2*exp(-sum(((x-y)./repmat([gpsarsa.sigma_s; gpsarsa.sigma_a],1,size(x,2))).^2,1)/2);
            k = gpsarsa.sigmaf^2*exp(-sum(((x-y)./repmat([gpsarsa.sigma_s; gpsarsa.sigma_a; gpsarsa.sigma_q],1,size(x,2))).^2,1)/2);
        end
        
        function k_ = kernel_vector_q(gpsarsa,x)
            if (size(x,1)==1)
                x = x';
            end
            if (size(x,1)~=size(gpsarsa.D,1))
                disp('Check dimension');
                return;
            end
            k_ = sparse(gpsarsa.kernel_q(gpsarsa.D,repmat(x,1,size(gpsarsa.D,2)))');
        end
        
        function k_ = kernel_vector(gpsarsa,x)
            if (size(x,1)==1)
                x = x';
            end
            if (size(x,1)~=size(gpsarsa.D,1))
                disp('Check dimension');
                return;
            end
            k_ = sparse(gpsarsa.kernel(gpsarsa.D,repmat(x,1,size(gpsarsa.D,2)))');
        end
        
        function update(gpsarsa, state_sequence, action_sequence, reward_sequence, gamma_sequence)
            
            episode_length = size(reward_sequence,1);
            for j=1:1:episode_length
                
                gamma_ = gamma_sequence(j,1);
                statet_1 = state_sequence(j,:)';
                actiont_1 = action_sequence(j,:)';
                statet = state_sequence(j+1,:)';
                actiont = action_sequence(j+1,:);
%                 rewardt_1 = gamma_.^(linspace(0, episode_length-j, episode_length-j+1))*reward_sequence(j:end,1);
%                 rewardt = (rt_1 - reward_sequence(j,1))/(gamma_ + eps);
                r = reward_sequence(j,1);
%                 qt_1 = gpsarsa.Q_bootstrapped(statet_1, actiont_1);
                qt = gpsarsa.Q_bootstrapped(statet, actiont);
%                 trajt_1 = qt_1;
%                 trajt_1 = [statet_1; actiont_1; qt_1]; 
                trajt_1 = [statet_1; actiont_1];
%                 trajt = qt;
%                 trajt = [statet; actiont; qt]; 
                trajt = [statet; actiont];
                
                k_t_1 = gpsarsa.kernel_vector(trajt_1);
                k_t = gpsarsa.kernel_vector(trajt);
                ktt = gpsarsa.kernel(trajt, trajt);
                at = gpsarsa.K_inv*k_t;
                et = ktt - k_t'*at;

                delk_t_1 = k_t_1 - gamma_*k_t;

                if ((et - gpsarsa.nu) > 10^(-4))
                    gpsarsa.D(:,size(gpsarsa.D,2)+1) = trajt;
                    gpsarsa.Q_D(size(gpsarsa.Q_D,1)+1,:) = qt;
                    gpsarsa.K_inv = [gpsarsa.K_inv+1/et*(at*at'), -at/et; -at'/et, 1/et];

                    % Dimension issues
                    c_t = gpsarsa.C_*delk_t_1 - gpsarsa.A;
                    %

                    delktt = gpsarsa.A'*(delk_t_1 - gamma_*k_t) + gamma_^2*ktt;
                    s_t = gpsarsa.sigma0^2 + delktt - delk_t_1'*gpsarsa.C_*delk_t_1;

                    gpsarsa.alpha_ = [gpsarsa.alpha_ + c_t/s_t*(delk_t_1'*gpsarsa.alpha_-r); gamma_/s_t*(delk_t_1'*gpsarsa.alpha_-r)];

                    gpsarsa.C_ = [gpsarsa.C_ + 1/s_t*(c_t*c_t'), gamma_/s_t*c_t; gamma_/s_t*c_t', gamma_^2/s_t];

                    gpsarsa.A = zeros(size(at,1)+1,1);
                    gpsarsa.A(end,1) = 1;

                else

                    h_t = gpsarsa.A - gamma_*at;
                    ct = gpsarsa.C_*delk_t_1 - h_t;
                    st = gpsarsa.sigma0^2 - ct'*delk_t_1;

                    gpsarsa.alpha_ = gpsarsa.alpha_ + ct/st*(delk_t_1'*gpsarsa.alpha_ - r);

                    gpsarsa.C_ = gpsarsa.C_ + 1/st*(ct*ct');

                    gpsarsa.A = at;

                end

                if (any(isnan(gpsarsa.alpha_)))
                    disp('Check alpha!');
                end
            end
        end
        
        
        function [q, action] = select_action(gpsarsa, state, epsilon)
            num_actions_to_sample = 20;
            actions_sampled = repmat(gpsarsa.actions(:,1),1,num_actions_to_sample) +...
                (gpsarsa.actions(:,2) - gpsarsa.actions(:,1))*rand(size(gpsarsa.actions,1), num_actions_to_sample);
            explore = rand();
            aexplore = randperm(num_actions_to_sample, 1);
            if (size(gpsarsa.D,1)>0)
                Q = zeros(num_actions_to_sample,1);
                diff = (gpsarsa.alpha_ - gpsarsa.C_*gpsarsa.Q_D);
                for a=1:num_actions_to_sample
                    action = actions_sampled(:,a);
                    qb = gpsarsa.Q_bootstrapped(state, action);
                    traj = [state; action; qb];
%                     traj = [state; action];
                    Q(a) = qb + gpsarsa.kernel_vector(traj)'*diff;
%                     Q(a) = qb + gpsarsa.kernel_vector_q(qb)'*diff;
                end
                
                [qmax, amax] = max(Q);
                [qmin, amin] = min(Q);
                q = (explore<epsilon)*Q(aexplore) + (explore>epsilon)*(gpsarsa.is_reward*qmax + (~gpsarsa.is_reward)*qmin);
                a = (explore<epsilon)*aexplore + (explore>epsilon)*(gpsarsa.is_reward*amax + (~gpsarsa.is_reward)*amin);
                
                action = actions_sampled(:,a);
            else
                action = actions_sampled(:,aexplore);
                q = 0;
            end           
        end
        
        function build_policy_monte_carlo(gpsarsa, num_episodes, max_episode_length, epsilon_end, debug_)
            state = gpsarsa.env.reset();            
            [~,action] = gpsarsa.select_action(state, gpsarsa.epsilon);
            
            for e=1:1:num_episodes
                is_terminal = false;
                num_steps = 0;
                epsi = gpsarsa.epsilon + (epsilon_end - gpsarsa.epsilon)/num_episodes*e;
                
                state_sequence = zeros(max_episode_length+1, size(state,1));
                state_sequence(1,:) = state';
                action_sequence = zeros(max_episode_length+1, size(action,1));
                action_sequence(1,:) = action';
                
                reward_sequence = zeros(max_episode_length, 1);
                gamma_sequence = zeros(max_episode_length, 1);
                
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    [state_, r, is_terminal] = gpsarsa.env.step(action);
                    [~, action_] = gpsarsa.select_action(state_, epsi);
                    
                    state_sequence(num_steps+1,:) = state_';
                    action_sequence(num_steps+1,:) = action_';
                    reward_sequence(num_steps,1) = r;
                    gamma_sequence(num_steps,1) = gpsarsa.gamma_;
                    action = action_;
                end
                state_ = gpsarsa.env.reset();
                [~, action_] = gpsarsa.select_action(state_, epsi);
                
                state_sequence = state_sequence(1:num_steps+1,:);
                state_sequence = [state_sequence; state_'];
                action_sequence = action_sequence(1:num_steps+1,:);
                action_sequence = [action_sequence; action_'];
                reward_sequence = reward_sequence(1:num_steps,1);
                reward_sequence = [reward_sequence; 0];
                gamma_sequence = gamma_sequence(1:num_steps,1);
                gamma_sequence = [gamma_sequence; 0];
                state = state_;
                action = action_;
                
                if (isempty(gpsarsa.D))
%                     rt = gptd.gamma_.^(linspace(0,size(reward_sequence,1)-1,size(reward_sequence,1)))*reward_sequence;
%                     traj = [state_sequence(1,:)';rt];
                    traj = [state_sequence(1,:)'; action_sequence(1,:)'];
%                     traj = rt;
%                     traj = [state_sequence(1,:)'; action_sequence(1,:)'; gpsarsa.Q_bootstrapped(state_sequence(1,:)', action_sequence(1,:)')];
%                     traj = [gpsarsa.Q_bootstrapped(state_sequence(1,:)', action_sequence(1,:)')];
                    gpsarsa.D = [traj];
                    gpsarsa.Q_D = [gpsarsa.Q_bootstrapped(state_sequence(1,:)', action_sequence(1,:)')];
                    gpsarsa.K_inv = 1/(gpsarsa.kernel(traj,traj));
                    gpsarsa.A = 1;
                    gpsarsa.alpha_ = 0;
                    gpsarsa.C_ = 0;
                end
                
                gpsarsa.update(state_sequence, action_sequence, reward_sequence, gamma_sequence);
                
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gpsarsa.D,2))));
                end
            end
        end
        
        function build_policy_monte_carlo_fixed_starts(gpsarsa, start_states, num_episodes, max_episode_length, epsilon_end, debug_)
            state = start_states(:,1);
            [~,a] = gpsarsa.select_action(state, gpsarsa.epsilon);
            is_terminal = gpsarsa.env.set(state);
            
            if (isempty(gpsarsa.D))
                gpsarsa.D = [state; a];
                gpsarsa.K_inv = 1/(gpsarsa.kernel_s(state, state)*gpsarsa.kernel_a(a, a));
                gpsarsa.A = 1;
                gpsarsa.alpha_ = 0;
                gpsarsa.C_ = 0;
                % Original GPSARSA
%                 gpsarsa.c_ = 0;
%                 gpsarsa.d = 0;
%                 gpsarsa.s = inf;
            end
            
            for e=1:1:num_episodes
                num_steps = 0;
                epsi = gpsarsa.epsilon + (epsilon_end - gpsarsa.epsilon)/num_episodes*e;
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    [state_, r, is_terminal] = gpsarsa.env.step(a);
                    [~, a_] = gpsarsa.select_action(state_, epsi);
                    gpsarsa.update(state_, a_, state, a, r, gpsarsa.gamma_);
                    state = state_;
                    a = a_;
                end
                state_ = start_states(:,1+mod(e,size(start_states,2)));
                [~, a_] = gpsarsa.select_action(state_, epsi);
                is_terminal = gpsarsa.env.set(state_);
                gpsarsa.update(state_, a_, state, a, 0, 0);
                state = state_;
                a = a_;
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gpsarsa.D,2))));
                end
            end
        end
        
        function [V, policy] = get_value_function(gpsarsa, states)
            V = zeros(size(states,2),1);
            policy = zeros(size(states,2),1);
            for i=1:1:size(states,2)
                state = states(:,i);
                [V(i),policy(i)] = gpsarsa.select_action(state, 0);
            end
        end
        
        function test_policy(gpsarsa, start, rand_start)
            
            tspan = [0,20];
            opts = odeset('RelTol',1e-8,'AbsTol',1e-8);
            m = gpsarsa.env.m;
            l = gpsarsa.env.l;
            b = gpsarsa.env.b;
            g = gpsarsa.env.g;
            x_limits = gpsarsa.env.x_limits;
            x_dot_limits = gpsarsa.env.x_dot_limits;
            if (rand_start)
                num_trials = 10;
                y_runs = cell(num_trials, 1);
                for i=1:1:num_trials
                    i
                    start = gpsarsa.env.reset();
                    [t,y] = ode45(@(t,y) GPBasedSwingUp(t,y,m,l,b,g,gpsarsa,0,x_limits,x_dot_limits),tspan,start,opts);
                    y_runs{i,1} = y;
                end
            else
                num_trials = 1;
                [t,y] = ode45(@(t,y) GPBasedSwingUp(t,y,m,l,b,g,gpsarsa,0,x_limits,x_dot_limits),tspan,start,opts);
                y_runs = cell(num_trials, 1);
                y_runs{1,1} = y;
            end
            
            figure;
            for i=1:1:num_trials
                y = y_runs{i,1};
                plot(y(:,1),y(:,2)); xlabel('theta'); ylabel('theta-dot');
                hold on;
                scatter(y(1,1),y(1,2),20,[0,1,0],'filled');
            end
            scatter(goal(1),goal(2),20,[1,0,0],'filled');
            text(goal(1),goal(2),'Goal','horizontal','left','vertical','bottom');
            title('GPSARSA');
            hold off;
           
        end
        
    end
end