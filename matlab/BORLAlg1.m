classdef BORLAlg1 < handle
    properties
        env
        actions               % [min, max]
        gamma_                % discount factor
        sigma_ls              % Sx1 (dimension of state space)
        sigma_la              % Ax1 (dimension of action space)
        sigma_fsa
        sigma_nsa
        muQ_bootstrapped      % bootstrapped mean function for Q values, takes in the state (Sx1) and action (Ax1)
        sigmaQ_bootstrapped   % bootstrapped sigma function for Q values, takes in the state (Sx1) and action (Ax1)
        yQ_mismatch           % data points for the mismatch GP - y (Q)
        xQ_mismatch           % data points for the mismatch GP - x (s,a)
        KQ_mismatch
        KQ_mismatch_inv
        KQ_mismatch_inv_times_yQ
        % policy              % No explicit policy for now... Policy derived from Q function
    end
    
    methods
        function borl = BORLAlg1(env, actions, gamma_, sigma_s, sigma_a, muQ_bootstrapped, sigmaQ_bootstrapped)
            borl.env = env;
            borl.gamma_ = gamma_;
            borl.actions = actions;
            borl.sigma_ls = sigma_s;
            borl.sigma_la = sigma_a;
            borl.sigma_nsa = 10^-4;
            borl.sigma_fsa = 1;
            borl.muQ_bootstrapped = muQ_bootstrapped;
            borl.sigmaQ_bootstrapped = sigmaQ_bootstrapped;
        end
        
        function k = kernel_mismatch(borl, x1, x2)
            sigma_sa = [borl.sigma_ls; borl.sigma_la]';
            sigma_sa = ones(size(x1,1),1)*sigma_sa;
            k = borl.sigma_fsa*exp(-sum(((x1-x2).^2)./(2*sigma_sa.^2),2)) + borl.sigma_nsa*prod(x1==x2,2);
        end
        
        function train(borl, maxiter, horizon, debug_)
            iter = 0;
            while(iter<maxiter)
                if (debug_)
                    iter = iter+1
                else
                    iter = iter+1;
                end
                t = 0;
                state = borl.env.reset();
                is_terimnal = false;
                state_visits = cell(horizon,3);
                while(~is_terminal && t<horizon)
                    t = t+1;
                    a = borl.select_action(state);
                    [next_state, reward, is_terminal] = borl.env.step(a);
                    state_visits{t,1} = state;
                    state_visits{t,2} = action;
                    state_visits{t,3} = reward;
                    state = next_state;
                end
                borl.update_mismatch_GP(state_visits, t);
            end
        end
        
        function update_mismatch_GP(borl, state_visits, horizon) 
            for t=1:1:horizon
                R = 0;
                for i=t:1:horizon
                    R = R + borl.gamma_^(i-t)*state_visits{i,3}; % Monte-Carlo Q value estimates
                end
                 
                % sample Q from the bootstrapped
                muQ = borl.muQ_bootstrapped(state_visits{t,1}', state_visits{t,2}');
                sigmaQ = borl.sigmaQ_bootstrapped(state_visits{t,1}', state_visits{t,2}');

                if (abs(R - muQ) > sigmaQ)
                    % Add point in the mismatch GP
                    x = [state_visits{t,1}', state_visits{t,2}'];
                    y = R - muQ;
                    
                    if (isfield(borl,'xQ_mismatch') && size(borl.xQ_mismatch,1)>1)
                        k = borl.kernel_mismatch(repmat(x,size(borl.xQ_mismatch,1),1),borl.xQ_mismatch);
                        xx = borl.kernel_mismatch(x,x);

                        b = -borl.KQ_mismatch_inv*k;
                        c = 1/(xx + k'*b);
                        a = borl.KQ_mismatch_inv + c*(b*b');
                        b = b*c;

                        borl.KQ_mismatch_inv = [a, b; b', c];
                        borl.KQ_mismatch = [borl.KQ_mismatch, k; k', xx];
                        borl.xQ_mismatch = [borl.xQ_mismatch; x];
                        borl.yQ_mismatch = [borl.yQ_mismatch; y];
                        
                    elseif (isfield(borl,'xQ_mismatch') && size(borl.xQ_mismatch,1)==1)
                        borl.xQ_mismatch = [borl.xQ_mismatch; x];
                        borl.yQ_mismatch = [borl.yQ_mismatch; y];
                        borl.KQ_mismatch = zeros(2,2);
                        borl.KQ_mismatch(:,1) = borl.kernel_mismatch(repmat(borl.xQ_mismatch(1,:),2,1),borl.xQ_mismatch);
                        borl.KQ_mismatch(:,2) = borl.kernel_mismatch(repmat(borl.xQ_mismatch(2,:),2,1),borl.xQ_mismatch);
                        borl.KQ_mismatch_inv = pinv(borl.KQ_mismatch);
                        
                    else
                        borl.xQ_mismatch = x;
                        borl.yQ_mismatch = y;
                    end
                end
            end 
            borl.KQ_mismatch_inv_times_yQ = borl.KQ_mismatch_inv*borl.yQ_mismatch;
        end
        
        function a = select_action(borl, state)
           % Use bootstrapped + mismatch GP to pick an action
           num_a_to_sample = 10;
           actions_sampled = repmat(borl.actions(:,1),1,num_a_to_sample) ...
               + rand(size(borl.actions,1), num_a_to_sample).*repmat((borl.actions(:,2) - borl.actions(:,1)),1,num_a_to_sample);
           states_sampled = repmat(state,1,num_a_to_sample);
           
           x = [states_sampled', actions_sampled'];
           
           % Sample mismatch for each action
           Q = zeros(num_a_to_sample,1);
           for i=1:1:num_a_to_sample
               if (isfield(borl,'xQ_mismatch'))
                   k_mismatch = borl.kernel_mismatch(repmat(x(i,:),size(borl.xQ_mismatch,1),1), borl.xQ_mismatch);
                   Q(i,1) = muQ(states_sampled(:,i)', actions_sampled(:,i)') + k_mismatch'*borl.KQ_mismatch_inv_times_yQ ...
                        + ((borl.kernel_mismatch(x(i,:),x(i,:)) - k_mismatch'*borl.KQ_mismatch_inv*k_mismatch) + ...
                            borl.sigmaQ_bootstrapped(states_sampled(:,i)', actions_sampled(:,i)'))*randn();
               else
                   Q(i,1) = muQ(states_sampled(:,i)', actions_sampled(:,i)') + ...
                            borl.sigmaQ_bootstrapped(states_sampled(:,i)', actions_sampled(:,i)')*randn();   
               end
           end
           [~,minA] = min(Q);
           a = actions_sampled(:,minA);
        end
    end
end