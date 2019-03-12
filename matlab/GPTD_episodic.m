classdef GPTD_episodic < handle

    properties (SetAccess='private')
        env
        env_sim
        V_bootstrapped
        V_D
        nu
        gamma_
        sigma0
        sigmak
        kernel_steps
        D
        A
        K_inv
        alpha_
        C_
    end
    
    methods
        function gptd = GPTD_episodic(env, env_sim, V_bootstrapped, nu, sigma0, sigmak, gamma_)
            gptd.env = env;
            gptd.env_sim = env_sim;
            gptd.V_bootstrapped = V_bootstrapped;
            gptd.nu = nu;
            gptd.sigma0 = sigma0;
            gptd.sigmak = sigmak;
            gptd.gamma_ = gamma_;
            gptd.kernel_steps = 100;
            gptd.V_D = [];
            gptd.D = [];
            gptd.A = [];
            gptd.K_inv = [];
            gptd.alpha_ = [];
            gptd.C_ = [];
        end
        
        function k = kernel(gptd,x,y)
            % x,y are nx1 inputs that represent states
%             k = exp(-sum((x-y).^2,1)/(2*gptd.sigmak^2));
            k = exp(-sum(((x-y)./repmat(gptd.sigmak,1,size(x,2))).^2,1)/2);
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
        
        function update(gptd, state_sequence, reward_sequence, gamma_sequence, policy)
            
            episode_length = size(reward_sequence,1);
            for j=1:1:episode_length
                gamma_ = gamma_sequence(j,1);
                
                xt_1 = state_sequence(j,:)';
                rt_1 = gamma_.^(linspace(0, episode_length-j, episode_length-j+1))*reward_sequence(j:end,1);
                
                xt = state_sequence(j+1,:)';
                rt = (rt_1 - reward_sequence(j,1))/(gamma_ + eps);
                
                r = reward_sequence(j,1);                

%                 rt = 0;
%                 x_ = xt;
%                 gptd.env_sim.set(x_);
%                 for i=1:1:(gptd.kernel_steps-1)
%                     [x_, r_, ~] = gptd.env_sim.step(policy(x_));
%                     rt = rt + gptd.gamma_^(i-1)*r_;
%                 end
                trajt = [xt;rt];
%                 trajt = rt;

%                 rt_1 = 0;
%                 x_ = xt_1;
%                 gptd.env_sim.set(x_);
%                 for i=1:1:(gptd.kernel_steps-1)
%                     [x_, r_, ~] = gptd.env_sim.step(policy(x_));
%                     rt_1 = rt_1 + gptd.gamma_^(i-1)*r_;
%                 end
                trajt_1 = [xt_1;rt_1];
%                 trajt_1 = rt_1;

                k_t_1 = gptd.kernel_vector(trajt_1);
                k_t = gptd.kernel_vector(trajt);
                ktt = gptd.kernel(trajt,trajt);
                at = gptd.K_inv*k_t;
                et = ktt - k_t'*at;

                delk_t_1 = k_t_1 - gamma_*k_t;

                if ((et - gptd.nu) > 10^(-4))
                    gptd.D(:,size(gptd.D,2)+1) = trajt;
                    gptd.V_D(size(gptd.V_D,1)+1,:) = gptd.V_bootstrapped(xt);

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

                if (any(isnan(gptd.alpha_)))
                    disp('Check alpha!');
                end
            end
        end

        function build_posterior_monte_carlo(gptd, policy_target, policy_bootstrapped, num_episodes, max_episode_length, debug_)
            s = gptd.env.reset();
            for e = 1:1:num_episodes
                is_terminal = false;
                num_steps = 0;
                state_sequence = zeros(max_episode_length+1, size(s,1));
                state_sequence(1,:) = s';
                reward_sequence = zeros(max_episode_length, 1);
                gamma_sequence = zeros(max_episode_length, 1);
                
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    a = policy_target(s);
                    [s_, r, is_terminal] = gptd.env.step(a);
                    state_sequence(num_steps+1,:) = s_';
                    reward_sequence(num_steps,1) = r;
                    gamma_sequence(num_steps,1) = gptd.gamma_;
                    s = s_;
                end
                s_ = gptd.env.reset();

                state_sequence = state_sequence(1:num_steps+1,:);
                state_sequence = [state_sequence; s_'];
                reward_sequence = reward_sequence(1:num_steps,1);
                reward_sequence = [reward_sequence; 0];
                gamma_sequence = gamma_sequence(1:num_steps,1);
                gamma_sequence = [gamma_sequence; 0];
                
                if (isempty(gptd.D))
                    rt = gptd.gamma_.^(linspace(0,size(reward_sequence,1)-1,size(reward_sequence,1)))*reward_sequence;
                    traj = [state_sequence(1,:)';rt];
%                     traj = rt;
                    gptd.D = [traj];
                    gptd.V_D = [gptd.V_bootstrapped(state_sequence(1,:)')];
                    gptd.K_inv = 1/(gptd.kernel(traj,traj));
                    gptd.A = 1;
                    gptd.alpha_ = 0;
                    gptd.C_ = 0;
                end
                
                gptd.update(state_sequence, reward_sequence, gamma_sequence, policy_bootstrapped);
                s = s_; 
                
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
            end
        end
        
        function build_posterior_monte_carlo_fixed_starts(gptd, policy_target, policy_bootstrapped, start_states, num_episodes, max_episode_length, debug_)
            s = start_states(:,1);
            is_terminal = gptd.env.set(s);
            for e = 1:1:num_episodes
                num_steps = 0;
                state_sequence = zeros(max_episode_length+1, size(s,1));
                state_sequence(1,:) = s';
                reward_sequence = zeros(max_episode_length, 1);
                gamma_sequence = zeros(max_episode_length, 1);
                
                while ((~is_terminal) && (num_steps < max_episode_length))
                    num_steps = num_steps + 1;
                    a = policy_target(s);
                    [s_, r, is_terminal] = gptd.env.step(a);
                    state_sequence(num_steps+1,:) = s_';
                    reward_sequence(num_steps,1) = r;
                    gamma_sequence(num_steps,1) = gptd.gamma_;
                    s = s_;
                end
                
                s_ = start_states(:,1+mod(e,size(start_states,2)));
                is_terminal = gptd.env.set(s_);
                
                state_sequence = state_sequence(1:num_steps+1,:);
                state_sequence = [state_sequence; s_'];
                reward_sequence = reward_sequence(1:num_steps,1);
                reward_sequence = [reward_sequence; 0];
                gamma_sequence = gamma_sequence(1:num_steps,1);
                gamma_sequence = [gamma_sequence; 0];
                
                if (isempty(gptd.D))
                    rt = gptd.gamma_.^(linspace(0,size(reward_sequence,1)-1,size(reward_sequence,1)))*reward_sequence;
                    traj = [state_sequence(1,:)';rt];
%                     traj = rt;
                    gptd.D = [traj];
                    gptd.V_D = [gptd.V_bootstrapped(state_sequence(1,:)')];
                    gptd.K_inv = 1/(gptd.kernel(traj,traj));
                    gptd.A = 1;
                    gptd.alpha_ = 0;
                    gptd.C_ = 0;
                end
                
                gptd.update(state_sequence, reward_sequence, gamma_sequence, policy_bootstrapped);
                s = s_; 
                
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
            end
        end
                
        function V = get_value_function(gptd, states, policy)
            
            V = zeros(size(states,2),1);
            for i=1:1:size(states,2)
                s = states(:,i);
                gptd.env_sim.set(s);
                r_ = 0;
%                 traj = zeros(size(s,1)*gptd.kernel_steps,1);
%                 traj(1:size(s,1),:) = s;
                for j=1:1:(gptd.kernel_steps-1)
                    [s, r, ~] = gptd.env_sim.step(policy(s));
                    r_ = r_ + gptd.gamma_^(j-1)*r;
%                     traj(j*size(s,1)+1:(j+1)*size(s,1),:) = s;
                end
                traj = [states(:,i);r_];
%                 traj = [r_];
                
                V(i,1) = gptd.V_bootstrapped(states(:,i)) + gptd.kernel_vector(traj)'*(gptd.alpha_ - gptd.C_*gptd.V_D);
            end
        end

        function visualize(gptd, grid_x, grid_x_dot) % Assuming a 2D state-space... Make it general?
            states = [reshape(grid_x, size(grid_x,1)*size(grid_x_dot,2),1),...
                      reshape(grid_x_dot, size(grid_x,1)*size(grid_x_dot,2),1)]';
            V = gptd.get_value_function(states);
            figure;
            x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
            y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
            imagesc(x,y,reshape(V,size(grid_x,1),size(grid_x_dot,2))');
            xlabel('theta'); ylabel('theta-dot');
            title('GPTD-Bootstrapped Value function');
            colorbar;
        end
    end
end