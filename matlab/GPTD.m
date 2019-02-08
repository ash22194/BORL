classdef GPTD < handle

    properties (SetAccess='private')
        env
        nu
        gamma_
        sigma0
        sigmak
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
        function gptd = GPTD(env, nu, sigma0, sigmak, gamma_)
            gptd.env = env;
            gptd.nu = nu;
            gptd.sigma0 = sigma0;
            gptd.sigmak = sigmak;
            gptd.gamma_ = gamma_;
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
            k_ = (gptd.kernel(gptd.D,repmat(x,1,size(gptd.D,2)))');
        end
        
        function update(gptd, xt, xt_1, r, gamma_)
            if (size(xt,1)==1)
                xt=xt';
            end
            if (size(xt_1,1)==1)
                xt_1=xt_1';
            end
            if isempty(gptd.D)
                gptd.D(:,1) = xt_1;
                %gptd.D(:,2) = xt;
                K_t = gptd.kernel(xt_1,xt_1); ;
                K_t_inv = inv(K_t);
                at =  1;
                h_t = 1;
                Q_t =  ([1/(h_t*K_t*h_t' + gptd.sigma0^2)]);
                alpha_t = [0]; %H_t'*Q_t*r;
                C_t = 0; %H_t'*Q_t*H_t;
                
                gptd.K_ = K_t;
                gptd.K_inv = K_t_inv;
                gptd.alpha_ = alpha_t;
                gptd.C_ = C_t;
                gptd.A = at;
                gptd.H_ = h_t;
            end
            K_t_1 = gptd.K_;
            K_t_1_inv = gptd.K_inv;
            alpha_t_1 = gptd.alpha_;
            C_t_1 = gptd.C_;
            At_1 = gptd.A;
            h_t_1 = gptd.H_;

            k_t_1 = gptd.kernel_vector(xt_1);
            k_t = gptd.kernel_vector(xt);
            ktt = gptd.kernel(xt,xt);
            at = K_t_1_inv*k_t;
            et = ktt - k_t'*at;

            delk_t = k_t_1 - gamma_*k_t;

            if ((abs(et) - gptd.nu) > 10^(-5))
                gptd.D(:,size(gptd.D,2)+1) = xt;
                % Dimension issues
                %c_t = H_t_1'*gt - at_1;
                %
                %delktt = at_1'*(delk_t_1 - gamma_*k_t) + gamma_^2*ktt;
                %s_t = gptd.sigma0^2 + delktt - delk_t_1'*C_t_1*delk_t_1;

                K_t = [K_t_1,k_t;k_t',ktt]; 

                K_t_inv = [et*K_t_1_inv+at*at' -at ; -at' 1]/et;
                at = [zeros(size(At_1, 1),1); 1];
                h_t = [At_1; -gamma_];
                delk_tt = At_1'*(k_t_1 - 2*gamma_*k_t) + gamma_^2*ktt;
                ct = h_t - [C_t_1*delk_t; 0];
                dt = r - delk_t'*alpha_t_1;
                st = delk_tt - delk_t'*C_t_1*delk_t;
                alpha_t_1 = [alpha_t_1; 0];
                C_t_1 = [C_t_1 zeros(size(C_t_1,1),1) ; zeros(size(C_t_1,2),1)', 0];
                gptd.K_ = K_t;
                gptd.K_inv = K_t_inv;
                gptd.A = at;

            else
%                 if (~any(gptd.D==xt))
%                     disp(strcat('Why should it be excluded? ',int2str(xt),', error ',string(et)));
%                 end
                h_t = At_1 - gamma_*at;
                delk_tt = h_t'*delk_t;
                dt = r - delk_t'*alpha_t_1;
                ct = h_t - C_t_1*delk_t;
                st = delk_t'*ct;
            end
            alpha_t = alpha_t_1 + ct*dt/st;
            C_t = C_t_1 + ct*ct'/st;

            if (any(isnan(alpha_t)))
                disp('Check alpha!');
            end

            gptd.alpha_ = alpha_t;
            gptd.C_ = C_t;
            gptd.H_ = h_t;
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
                    gptd.update(s_, s, r, gptd.gamma_);
                    s = s_;
% Display grid world
%                     disp(strcat('Current : ',int2str(s)));
%                     V = gptd.get_value_function(1:1:gptd.env.num_states);
%                     V = reshape(V,4,4);
%                     imagesc(V);
%                     colorbar;
%                     close;

% Display for swing-up
                    if(mod(e,10) == 0 && num_steps==1)
                        dx = (gptd.env.x_limits(2)-gptd.env.x_limits(1))/(gptd.env.num_points_x - 1); 
                        dx_dot = (gptd.env.x_dot_limits(2)-gptd.env.x_dot_limits(1))/(gptd.env.num_points_x_dot - 1);
                        [grid_x,grid_x_dot] = meshgrid(gptd.env.x_limits(1):dx:gptd.env.x_limits(2),gptd.env.x_dot_limits(1):dx_dot:gptd.env.x_dot_limits(2));
                        V = gptd.get_value_function([reshape(grid_x, size(grid_x,1)*size(grid_x,2), 1), reshape(grid_x_dot, size(grid_x,1)*size(grid_x,2), 1)]');
    %                     if(max(V)>50)
    %                         keyboard
    %                     end
                        V = reshape(V, 51, 51);
                        x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
                        y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
                        figure, title(['Iteration ', num2str(e)])
                        imagesc(x,y,V);
                        xlabel('theta'); ylabel('theta-dot');
                        colorbar;
                        hold on;
                        scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
                        pause(0.1)
                    end
                end
                s_ = gptd.env.reset();
                if (size(gptd.D,2)>gptd.env.num_states)
                    disp('Check dictionary size');
                end
                %gptd.update(s_, s, 0, 0);
                s = s_;
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
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
        
        function V = get_value_function(gptd, states)
            V = zeros(size(states,2),1);
            for i=1:1:size(states,2)
                s = states(:,i);
                V(i) = gptd.kernel_vector(s)'*gptd.alpha_;
            end
        end
        
        function visualize(gptd, grid_x, grid_x_dot) % Assuming a 2D state-space... Make it general?
            states = zeros(2,size(grid_x,2)*size(grid_x_dot,1));
            for i = 1:1:size(grid_x,2)
                for j = 1:1:size(grid_x_dot,1)
                    states(:,(i-1)*size(grid_x_dot,1) + j) = [grid_x(j,i);grid_x_dot(j,i)];
                end
            end
            V = gptd.get_value_function(states);
            figure;
            x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
            y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
            imagesc(x,y,reshape(V,size(grid_x_dot,1),size(grid_x,2)));
            xlabel('theta'); ylabel('theta-dot');
            title('GPTD Value function');
            colorbar;
        end
    end
end