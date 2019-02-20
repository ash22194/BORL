classdef GPSARSA_fast < handle

    properties (SetAccess='private')
        env
        actions
        discra % discrete actions?
        epsilon
        nu
        gamma_
        is_reward
        sigma_s
        sigma_a
        sigma0
        D
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
        function gpsarsa = GPSARSA_fast(env, actions, discra, epsilon, nu, sigma_s, sigma_a, sigma0, gamma_, is_reward)
            gpsarsa.env = env;
            gpsarsa.actions = actions;
            gpsarsa.discra = discra;
            gpsarsa.epsilon = epsilon;
            gpsarsa.nu = nu; % TODO: Compute nu based on kernel and sigma(s)
            gpsarsa.sigma0 = sigma0;
            gpsarsa.sigma_s = sigma_s;
            gpsarsa.sigma_a = sigma_a;
            gpsarsa.gamma_ = gamma_;
            gpsarsa.is_reward = is_reward;
            
            gpsarsa.D = [];
            gpsarsa.A = [];
            gpsarsa.H_ = [];
            gpsarsa.K_inv = [];
            gpsarsa.alpha_ = [];
            gpsarsa.C_ = [];
        end
        
        function k = kernel_s(gpsarsa, x, y)
            % x,y have nx1 columns that represent states
            if (size(x,1)~=size(y,1))
                fprintf('Check input dimensions!');
                k = 0;
                return;
            elseif (size(x,2)~=size(y,2)) 
                fprintf('Number of states in both inputs must be same!');
                k = 0;
                return;
            end
            k = exp(-sum(((x-y)./repmat(gpsarsa.sigma_s,1,size(x,2))).^2,1)/2)';
        end
        
        function k = kernel_a(gpsarsa, x, y)
            % x,y have nx1 columns that represent actions
            if (size(x,1)~=size(y,1))
                fprintf('Check input dimensions!');
                k = 0;
                return;
            elseif (size(x,2)~=size(y,2)) 
                fprintf('Number of actions in both inputs must be same!');
                k = 0;
                return;
            end
            k = exp(-sum(((x-y)./repmat(gpsarsa.sigma_a,1,size(x,2))).^2,1)/2)';
        end
        
        function k_ = kernel_vector(gpsarsa, x, a)
            if (size(x,1)==1)
                x = x';
            end
            if (((size(x,1) + size(a,1))~=size(gpsarsa.D,1) && size(gpsarsa.D,1)>0) || size(x,2)~=1)
                disp('Check dimension');
                return;
            end
            k_ = sparse(gpsarsa.kernel_s(gpsarsa.D(1:size(x,1),:),repmat(x,1,size(gpsarsa.D,2))).*...
                        gpsarsa.kernel_a(gpsarsa.D(size(x,1)+1:end,:),repmat(a,1,size(gpsarsa.D,2))));
        end
        
        function update(gpsarsa, xt, actiont, xt_1, actiont_1, r, gamma_)
            if (size(xt,1)==1)
                xt=xt';
            end
            if (size(xt_1,1)==1)
                xt_1=xt_1';
            end

            k_t_1 = gpsarsa.kernel_vector(xt_1, actiont_1);
            k_t = gpsarsa.kernel_vector(xt, actiont);
            ktt = gpsarsa.kernel_s(xt,xt)*gpsarsa.kernel_a(actiont,actiont);
            at = gpsarsa.K_inv*k_t;
            et = ktt - k_t'*at;

            delk_t_1 = k_t_1 - gamma_*k_t;

            if ((et - gpsarsa.nu) > 10^(-4))
                gpsarsa.D(:,size(gpsarsa.D,2)+1) = [xt; actiont];
                
                % Online GPTD
                    
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

                % Original GPSARSA
%                 gpsarsa.K_inv = [gpsarsa.K_inv+1.0/et*(at*at'), -at/et; -at'/et, 1.0/et];
%                 
%                 h_t = [gpsarsa.A; -gamma_];
% 
%                 c_t = gamma_*gpsarsa.sigma0^2/gpsarsa.s*[gpsarsa.c_; 0] + h_t - [gpsarsa.C_*delk_t_1; 0];
% 
%                 delktt = gpsarsa.A'*(delk_t_1 - gamma_*k_t) + gamma_^2*ktt;
%
%                 gpsarsa.d = gamma_*gpsarsa.sigma0^2/gpsarsa.s*gpsarsa.d + r - delk_t_1'*gpsarsa.alpha_;
%
%                 gpsarsa.s = (1+gamma_^2)*gpsarsa.sigma0^2 + delktt - delk_t_1'*gpsarsa.C_*delk_t_1 + ...
%                      2*gamma_*gpsarsa.sigma0^2/gpsarsa.s*gpsarsa.c_'*delk_t_1 - (gamma_*gpsarsa.sigma0^2)^2/gpsarsa.s;
%  
%                 gpsarsa.alpha_ = [gpsarsa.alpha_; 0];
% 
%                 gpsarsa.C_ = [gpsarsa.C_ zeros(size(gpsarsa.C_,1),1); zeros(size(gpsarsa.C_,1),1)', 0];
%
%                 gpsarsa.A = zeros(size(at,1)+1,1);
%                 gpsarsa.A(end,1) = 1;

            else
                
                h_t = gpsarsa.A - gamma_*at;
                ct = gpsarsa.C_*delk_t_1 - h_t;
                st = gpsarsa.sigma0^2 - ct'*delk_t_1;

                gpsarsa.alpha_ = gpsarsa.alpha_ + ct/st*(delk_t_1'*gpsarsa.alpha_ - r);

                gpsarsa.C_ = gpsarsa.C_ + 1/st*(ct*ct');
                
                gpsarsa.A = at;
                
                % Original GPSARSA
%                 h_t = (gpsarsa.A - gamma_*at);
% 
%                 c_t = gamma_*gpsarsa.sigma0^2/gpsarsa.s*gpsarsa.c_ + h_t - gpsarsa.C_*delk_t_1;
%
%                 dt = gpsarsa.gamma_*gpsarsa.sigma0^2/gpsarsa.s*gpsarsa.d + r - delk_t_1'*gpsarsa.alpha_;
% %                 dt = dt_1;
%
%                 gpsarsa.s = (1+gamma_^2)*gpsarsa.sigma0^2 + delk_t_1'*(c_t + gamma_*gpsarsa.sigma0^2/gpsarsa.s*gpsarsa.c_) - (gamma_*gpsarsa.sigma0^2)^2/gpsarsa.s;
%
%                 gpsarsa.A = at;
%
%                 gpsarsa.c_ = ct;
            end
            
            % Original GPSARSA
%             if (abs(st) < 10^(-4))
%                 disp('Check st');
%             end
%             gpsarsa.alpha_ = gpsarsa.alpha_ + c_t/st*dt;
% 
%             gpsarsa.C_ = gpsarsa.C_ + 1/st*(c_t*c_t');
            
            if (any(isnan(gpsarsa.alpha_)))
                disp('Check alpha!');
            end
        end
        
        
        function [q, a] = select_action(gpsarsa, s, epsilon)
            if (~gpsarsa.discra)
                num_actions_to_sample = 20;
                actions_sampled = repmat(gpsarsa.actions(:,1),1,num_actions_to_sample) +...
                    (gpsarsa.actions(:,2) - gpsarsa.actions(:,1))*rand(size(gpsarsa.actions,1), num_actions_to_sample);
            else
                num_actions_to_sample = size(gpsarsa.actions, 2);
                actions_sampled = gpsarsa.actions;
            end
            
            explore = rand();
            
            if (size(gpsarsa.D,1)>0 && explore>=epsilon)
                Q = zeros(num_actions_to_sample,1);
                for a=1:num_actions_to_sample
                    Q(a) = gpsarsa.alpha_'*gpsarsa.kernel_vector(s, actions_sampled(:,a));
                end
                if (gpsarsa.is_reward)
                    [q,a] = max(Q);
                else
                    [q,a] = min(Q);
                end
                a = actions_sampled(:,a);
            else
                a = actions_sampled(:,randperm(size(actions_sampled,2),1));
                if (size(gpsarsa.D,1)>0)
                    q = gpsarsa.alpha_'*gpsarsa.kernel_vector(s,a);
                else
                    q = 0;
                end
            end
            
        end
        
        
        function build_policy_monte_carlo(gpsarsa, num_episodes, max_episode_length, epsilon_end, debug_)
            state = gpsarsa.env.reset();
            [~,a] = gpsarsa.select_action(state, gpsarsa.epsilon);
            
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
                is_terminal = false;
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
                state_ = gpsarsa.env.reset();
                [~, a_] = gpsarsa.select_action(state_, epsi);
                gpsarsa.update(state_, a_, state, a, 0, 0);
                state = state_;
                a = a_;
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
        
        function V = get_value_function(gpsarsa, states)
            V = zeros(size(states,2),1);
            for i=1:1:size(states,2)
                state = states(:,i);
                [V(i),~] = gpsarsa.select_action(state, 0);
            end
        end
        
        function build_posterior_vi(gptd, states, policy, num_iterations, debug_)
            
            for iter = 1:1:num_iterations
                if (debug_)
                    disp(strcat('Iteration : ',int2str(iter)));
                end
                is_terminal = false;
                for s=1:1:size(states,2)
                    s1 = states(:,s);
                    if (is_terminal)
                        gptd.update(s1,s2,0,0);
                    end
                    if (gptd.env.set(s1))
                        continue;
                    end
                    [s2, r, is_terminal] = gptd.env.step(policy(s1));
                    gptd.update(s2,s1,r,gptd.gamma_);
                end 
            end
            
        end
        
    end
end