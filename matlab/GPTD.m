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
            k = exp(-sum((x-y).^2)/(2*gptd.sigmak^2));
        end
        
        function k_ = kernel_vector(gptd,x)
            if (size(x,1)==1)
                x = x';
            end
            if (size(x,1)~=size(gptd.D,1))
                disp('Check dimension');
                return;
            end
            k_ = gptd.kernel(gptd.D,repmat(x,1,size(gptd.D,2)))';
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
                gptd.D(:,2) = xt;
                K_t = zeros(2,2);
                K_t(:,1) = gptd.kernel_vector(xt_1);
                K_t(:,2) = gptd.kernel_vector(xt);
                K_t_inv = inv(K_t);
                At = eye(2);
                H_t = [1,-gamma_];
                Q_t = [1/(H_t*K_t*H_t' + gptd.sigma0^2)];
                alpha_t = H_t'*Q_t*r;
                C_t = H_t'*Q_t*H_t;
            else
                K_t_1 = gptd.K_;
                K_t_1_inv = gptd.K_inv;
                alpha_t_1 = gptd.alpha_;
                C_t_1 = gptd.C_;
                Q_t_1 = gptd.Q_;
                At_1 = gptd.A;
                H_t_1 = gptd.H_;
                
                k_t_1 = gptd.kernel_vector(xt_1);
                k_t = gptd.kernel_vector(xt);
                ktt = gptd.kernel(xt,xt);
                at_1 = K_t_1_inv*k_t_1;
                at = K_t_1_inv*k_t;
                et = ktt - k_t'*at;
                
                delk_t_1 = k_t_1 - gamma_*k_t;
                gt = Q_t_1*H_t_1*delk_t_1;
                
                 if (et > gptd.nu)
                    gptd.D(:,size(gptd.D,2)+1) = xt;
                    % Dimension issues
                    c_t = H_t_1'*gt - at_1;
                    %
                    delktt = at_1'*(delk_t_1 - gamma_*k_t) + gamma_^2*ktt;
                    s_t = gptd.sigma0^2 + delktt - delk_t_1'*C_t_1*delk_t_1;

                    K_t = zeros(size(K_t_1,1)+1,size(K_t_1,2)+1);
                    K_t(1:size(K_t_1,1),1:size(K_t_1,2)) = K_t_1;
                    K_t(size(K_t_1,1)+1,size(K_t_1,2)+1) = ktt;
                    K_t(size(K_t_1,1)+1,1:size(K_t_1,2)) = k_t';
                    K_t(1:size(K_t_1,1),size(K_t_1,2)+1) = k_t;
                    K_tc = cond(K_t);
                    if (cond(K_tc) > 50)
                        disp(strcat('Condition number of ',int2str(K_tc)));
                    end

                    K_t_inv = zeros(size(K_t_1_inv,1)+1,size(K_t_1_inv,2)+1);
                    K_t_inv(1:size(K_t_1_inv,1),1:size(K_t_1_inv,2)) = et*K_t_1_inv + at*at';
                    K_t_inv(size(K_t_1_inv,1)+1,size(K_t_1_inv,2)+1) = 1;
                    K_t_inv(size(K_t_1_inv,1)+1,1:size(K_t_1_inv,2)) = -at';
                    K_t_inv(1:size(K_t_1_inv,1),size(K_t_1_inv,2)+1) = -at;
                    K_t_inv = 1/et*K_t_inv;

                    alpha_t = zeros(size(alpha_t_1,1)+1,size(alpha_t_1,2));
                    alpha_t(1:size(alpha_t_1,1),:) = alpha_t_1 + c_t/s_t*(delk_t_1'*alpha_t_1-r);
                    alpha_t(size(alpha_t_1,1)+1,:) = gamma_/s_t*(delk_t_1'*alpha_t_1-r);

                    C_t = zeros(size(C_t_1,1)+1,size(C_t_1,2)+1);
                    C_t(1:size(C_t_1,1),1:size(C_t_1,2)) = C_t_1 + 1/s_t*(c_t*c_t');
                    C_t(size(C_t_1,1)+1,size(C_t_1,2)+1) = gamma_^2/s_t;
                    C_t(size(C_t_1,1)+1,1:size(C_t_1,2)) = gamma_/s_t*c_t';
                    C_t(1:size(C_t_1,1),size(C_t_1,2)+1) = gamma_/s_t*c_t;

                    Q_t = zeros(size(Q_t_1,1)+1,size(Q_t_1,2)+1);
                    Q_t(1:size(Q_t_1,1),1:size(Q_t_1,2)) = s_t*Q_t_1 + (gt*gt');
                    Q_t(size(Q_t_1,1)+1,size(Q_t_1,2)+1) = 1;
                    Q_t(size(Q_t_1,1)+1,1:size(Q_t_1,2)) = -gt';
                    Q_t(1:size(Q_t_1,1),size(Q_t_1,2)+1) = -gt;
                    Q_t = Q_t/s_t;

                    At = zeros(size(At_1,1)+1,size(At_1,2)+1);
                    At(1:size(At_1,1),1:size(At_1,2)) = At_1;
                    At(size(At_1,1)+1,size(At_1,2)+1) = 1;

                    H_t = zeros(size(H_t_1,1)+1,size(H_t_1,2)+1);
                    H_t(1:size(H_t_1,1),1:size(H_t_1,2)) = H_t_1;
                    H_t(size(H_t_1,1)+1,1:size(H_t_1,2)) = at_1';
                    H_t(size(H_t_1,1)+1,size(H_t_1,2)+1) = -gamma_;
                    
                else
                    h_t = at_1 - gamma_*at;
                    ct = H_t_1'*gt - h_t;
                    st = gptd.sigma0^2 - ct'*delk_t_1;

                    K_t = K_t_1;
                    K_t_inv = K_t_1_inv;

                    alpha_t = alpha_t_1 + ct/st*(delk_t_1'*alpha_t_1 - r);

                    C_t = C_t_1 + 1/st*(ct*ct');

                    Q_t = zeros(size(Q_t_1,1)+1,size(Q_t_1,2)+1);
                    Q_t(1:size(Q_t_1,1),1:size(Q_t_1,2)) = st*Q_t_1 + (gt*gt');
                    Q_t(size(Q_t_1,1)+1,size(Q_t_1,2)+1) = 1;
                    Q_t(size(Q_t_1,1)+1,1:size(Q_t_1,2)) = -gt';
                    Q_t(1:size(Q_t_1,1),size(Q_t_1,2)+1) = -gt;
                    Q_t = Q_t/st;

                    At = zeros(size(At_1,1)+1,size(At_1,2));
                    At(1:size(At_1,1),:) = At_1;
                    At(size(At_1,1)+1,:) = at';

                    H_t = zeros(size(H_t_1,1)+1,size(H_t_1,2));
                    H_t(1:size(H_t_1,1),:) = H_t_1;
                    H_t(size(H_t_1,1)+1,:) = h_t';
                end
            end
            
            gptd.K_ = K_t;
            gptd.K_inv = K_t_inv;
            gptd.alpha_ = alpha_t;
            gptd.C_ = C_t;
            gptd.Q_ = Q_t;
            gptd.A = At;
            gptd.H_ = H_t;
        end

        function build_posterior(gptd, policy, num_episodes, max_episode_length, debug_)
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
                    dx = (gptd.env.x_limits(2)-gptd.env.x_limits(1))/(gptd.env.num_points_x - 1); 
                    dx_dot = (gptd.env.x_dot_limits(2)-gptd.env.x_dot_limits(1))/(gptd.env.num_points_x_dot - 1);
                    [grid_x,grid_x_dot] = meshgrid(gptd.env.x_limits(1):dx:gptd.env.x_limits(2),gptd.env.x_dot_limits(1):dx_dot:gptd.env.x_dot_limits(2));
                    V = gptd.get_value_function(grid_x, grid_x_dot);
                    x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
                    y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
                    imagesc(x,y,V);
                    xlabel('theta'); ylabel('theta-dot');
                    colorbar;
                    hold on;
                    scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
                    close;
                end
                s_ = gptd.env.reset();
                gptd.update(s_, s, 0, 0);
                if (debug_)
                    disp(strcat('Episode : ',int2str(e),', Dictionary size : ',int2str(size(gptd.D,2))));
                end
            end
        end

        function V = get_value_function(gptd, grid_x, grid_x_dot) % Assuming a 2D state-space... Make it general?
            if (size(grid_x,1)~=size(grid_x_dot,1) || size(grid_x,2)~=size(grid_x_dot,2))
                disp('Check grid');
            end
            V = zeros(size(grid_x));
            for i=1:1:size(grid_x,1)
                for j=1:1:size(grid_x,2)
                    V(i,j) = gptd.kernel_vector([grid_x(i,j);grid_x_dot(i,j)])'*gptd.alpha_;
                end
            end
        end
        
        function visualize(gptd, grid_x, grid_x_dot) % Assuming a 2D state-space... Make it general?
            V = gptd.get_value_function(grid_x,grid_x_dot);
            figure;
            x = [gptd.env.x_limits(1), gptd.env.x_limits(2)];
            y = [gptd.env.x_dot_limits(1), gptd.env.x_dot_limits(2)];
            imagesc(x,y,V);
            xlabel('theta'); ylabel('theta-dot');
            title('GPTD Value function');
            colorbar;
        end
    
    end

end