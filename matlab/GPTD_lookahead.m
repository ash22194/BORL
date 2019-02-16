classdef GPTD_lookahead < handle

    properties (SetAccess='private')
        env
        env_sim
        nu
        gamma_
        sigma0
        sigmak
        kernel_steps
        D
        A
        H_
        Q_
        K_
        K_inv
        alpha_
        C_
    end
    
    methods
        function gptd = GPTD_lookahead(env, env_sim, nu, sigma0, sigmak, gamma_)
            gptd.env = env;
            gptd.env_sim = env_sim;
            gptd.nu = nu;
            gptd.sigma0 = sigma0;
            gptd.sigmak = sigmak;
            gptd.gamma_ = gamma_;
            gptd.kernel_steps = 5;
            gptd.D = [];
            gptd.A = [];
            gptd.H_ = [];
            gptd.Q_ = [];
            gptd.K_ = [];
            gptd.K_inv = [];
            gptd.alpha_ = [];
            gptd.C_ = [];
        end
        
        function k = kernel(gptd,x,y)
            % x,y are nx1 inputs that represent states
            k = exp(-sum((x-y).^2,1)/(2*gptd.sigmak^2));
        end
        
        function k_ = kernel_vector(gptd,x)
            if (size(x,1)==1)
                x = x';
            end
            if (size(x,1)~=size(gptd.D,1))
                disp('Check dimension');
                return;
            end
            k_ = sparse(gptd.kernel(gptd.D,repmat(x,1,size(gptd.D,2)))');
        end
        
        function update(gptd, xt, xt_1, r, policy, gamma_)
            if (size(xt,1)==1)
                xt=xt';
            end
            if (size(xt_1,1)==1)
                xt_1=xt_1';
            end
            
            trajt = zeros(size(xt,1)*gptd.kernel_steps,1);
            trajt(1:size(xt,1),:) = xt;
            x_ = xt;
            gptd.env_sim.set(x_);
            for i=1:1:gptd.kernel_steps
                [x_, ~, ~] = gptd.env_sim.step(policy(x_));
                trajt(i*size(xt,1)+1:(i + 1)*size(xt,1),:) = x_; 
            end
            
            trajt_1 = zeros(size(xt_1,1)*gptd.kernel_steps,1);
            trajt_1(1:size(xt_1,1),:) = xt_1;
            x_ = xt_1;
            gptd.env_sim.set(x_);
            for i=1:1:gptd.kernel_steps
                [x_, ~, ~] = gptd.env_sim.step(policy(x_));
                trajt_1(i*size(xt_1,1)+1:(i + 1)*size(xt_1,1),:) = x_;
            end

            if isempty(gptd.D)
                gptd.D(:,1) = trajt_1;
                gptd.D(:,2) = trajt;
                K_t = sparse(zeros(2,2));
                K_t(:,1) = gptd.kernel_vector(trajt_1);
                K_t(:,2) = gptd.kernel_vector(trajt);
                gptd.K_inv = inv(K_t);
                gptd.A = [0;1];
                H_t = sparse([1,-gamma_]);
                Q_t = sparse([1/(H_t*K_t*H_t' + gptd.sigma0^2)]);
                gptd.alpha_ = H_t'*Q_t*r;
                gptd.C_ = H_t'*Q_t*H_t;
            else
                
                k_t_1 = gptd.kernel_vector(trajt_1);
                k_t = gptd.kernel_vector(trajt);
                ktt = gptd.kernel(trajt,trajt);
                at = gptd.K_inv*k_t;
                et = ktt - k_t'*at;
                
                delk_t_1 = k_t_1 - gamma_*k_t;
                
                if ((et - gptd.nu) > 10^(-4))
                    gptd.D(:,size(gptd.D,2)+1) = trajt;
                    
                    gptd.K_inv = [gptd.K_inv + 1/et*(at*at'), -at/et; -at'/et, 1/et];
                    
                    % Dimension issues
                    c_t = gptd.C_*delk_t_1 - gptd.A;
                    %
                    
                    delktt = gptd.A'*(delk_t_1 - gamma_*k_t) + gamma_^2*ktt;
                    s_t = gptd.sigma0^2 + delktt - delk_t_1'*gptd.C_*delk_t_1;

                    gptd.alpha_ = [gptd.alpha_ + c_t/s_t*(delk_t_1'*gptd.alpha_-r); gamma_/s_t*(delk_t_1'*gptd.alpha_-r)];
                    
                    gptd.C_ = [gptd.C_ + 1/s_t*(c_t*c_t'), gamma_/s_t*c_t; gamma_/s_t*c_t', gamma_^2/s_t];
                    
                    gptd.A = zeros(size(at,1)+1,1);
                    gptd.A(end,1) = 1;
                    
                else
                    h_t = gptd.A - gamma_*at;
                    ct = gptd.C_*delk_t_1 - h_t;
                    st = gptd.sigma0^2 - ct'*delk_t_1;

                    gptd.alpha_ = gptd.alpha_ + ct/st*(delk_t_1'*gptd.alpha_ - r);

                    gptd.C_ = gptd.C_ + 1/st*(ct*ct');
                    
                    gptd.A = at;
                end
            end
            if (any(isnan(gptd.alpha_)))
                disp('Check alpha!');
            end
        end

        function build_posterior_monte_carlo(gptd, policy, num_episodes, max_episode_length, debug_)
            s = gptd.env.reset();
            for e=1:1:num_episodes
                is_terminal = false;
                num_steps = 0;
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    a = policy(s);
                    [s_, r, is_terminal] = gptd.env.step(a);
                    gptd.update(s_, s, r, policy, gptd.gamma_);
                    s = s_;
                end
                s_ = gptd.env.reset();
                if (size(gptd.D,2)>gptd.env.num_states)
                    disp('Check dictionary size');
                end
                gptd.update(s_, s, 0, policy, 0);
                s = s_;
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
            end
        end
        
        function build_posterior_monte_carlo_fixed_starts(gptd, policy, start_states, num_episodes, max_episode_length, debug_)
            s = start_states(:,1);
            is_terminal = gptd.env.set(s);
            for e=1:1:num_episodes
                num_steps = 0;
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    a = policy(s);
                    [s_, r, is_terminal] = gptd.env.step(a);
                    gptd.update(s_, s, r, policy, gptd.gamma_);
                    s = s_;
                end
                s_ = start_states(:,1+mod(e,size(start_states,2)));
                is_terminal = gptd.env.set(s_);
                if (size(gptd.D,2)>gptd.env.num_states)
                    disp('Check dictionary size');
                end
                gptd.update(s_, s, 0, policy, 0);
                s = s_;
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
            end
        end
                
        function V = get_value_function(gptd, states)
            V = zeros(size(states,2),1);
            for i=1:1:size(states,2)
                s = states(:,i);
                V(i) = gptd.kernel_vector(s)'*gptd.alpha_;
            end
        end
        
        function visualize(gptd, grid_x, grid_x_dot) % Assuming a 2D state-space... Make it general?
            states = [reshape(grid_x,size(grid_x,1)*size(grid_x_dot,2),1),...
                      reshape(grid_x_dot,size(grid_x,1)*size(grid_x_dot,2),1)]';
            V = gptd.get_value_function(states);
            figure;
            x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
            y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
            imagesc(x,y,reshape(V,size(grid_x,1),size(grid_x_dot,2))');
            xlabel('theta'); ylabel('theta-dot');
            title('GPTD Value function');
            colorbar;
        end
    end
end