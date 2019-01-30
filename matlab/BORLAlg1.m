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
            borl.sigma_nsa = 10^-2;
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
                iter = iter+1;
                t = 0;
                state = borl.env.reset();
                is_terminal = false;
                state_visits = cell(horizon+1,3);
                while(~is_terminal && t<horizon)
                    t = t+1;
                    [~,a] = borl.select_action(state);
                    [next_state, reward, is_terminal] = borl.env.step(a);
                    state_visits{t,1} = state;
                    state_visits{t,2} = a;
                    state_visits{t,3} = reward;
                    state = next_state;
                end
                state_visits{t+1,1} = state;
                state_visits{t+1,2} = 0;
                state_visits{t+1,3} = 0;
                num_points_added = borl.update_mismatch_GP(state_visits, t);
                if (debug_)
                    disp(strcat('Episode :',num2str(iter),', Mismatch points added :',num2str(num_points_added),' Total points :',num2str(size(borl.xQ_mismatch,1))));
                end
            end
        end
        
        function num_points_added = update_mismatch_GP(borl, state_visits, horizon) 
            num_points_added = 0;
            [QLast, ~] = borl.select_action(state_visits{horizon+1,1});
            for t=1:1:horizon
                R = 0;
                for i=t:1:horizon
                    R = R + borl.gamma_^(i-t)*state_visits{i,3}; % Monte-Carlo Q value estimates
                end
                R = R + QLast;
                
                % sample Q from the bootstrapped
                muQ = borl.muQ_bootstrapped(state_visits{t,1}, state_visits{t,2});
                sigmaQ = borl.sigmaQ_bootstrapped(state_visits{t,1}, state_visits{t,2});
               
                if (abs(R - muQ) > 500*sigmaQ)
                    % Add point in the mismatch GP
                    num_points_added = num_points_added + 1;
                    x = [state_visits{t,1}', state_visits{t,2}'];
                    y = R - muQ;
                    
                    if (size(borl.xQ_mismatch,1)>1)
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
                        
                    elseif (size(borl.xQ_mismatch,1)==1)
                        borl.xQ_mismatch = [borl.xQ_mismatch; x];
                        borl.yQ_mismatch = [borl.yQ_mismatch; y];
                        borl.KQ_mismatch = zeros(2,2);
                        borl.KQ_mismatch(:,1) = borl.kernel_mismatch(repmat(borl.xQ_mismatch(1,:),2,1),borl.xQ_mismatch);
                        borl.KQ_mismatch(:,2) = borl.kernel_mismatch(repmat(borl.xQ_mismatch(2,:),2,1),borl.xQ_mismatch);
                        borl.KQ_mismatch_inv = pinv(borl.KQ_mismatch);
                        
                    elseif (size(borl.xQ_mismatch,1)==0)
                        borl.xQ_mismatch = x;
                        borl.yQ_mismatch = y;
                    end
                end
            end 
            borl.KQ_mismatch_inv_times_yQ = borl.KQ_mismatch_inv*borl.yQ_mismatch;
        end
         
        function [q, a] = select_action(borl, state)
           % Use bootstrapped + mismatch GP to pick an action
           num_a_to_sample = 20;
           actions_sampled = repmat(borl.actions(:,1),1,num_a_to_sample) ...
               + rand(size(borl.actions,1), num_a_to_sample).*repmat((borl.actions(:,2) - borl.actions(:,1)),1,num_a_to_sample);
           
           % Sample Q values for each action
           Q = zeros(num_a_to_sample,1);
           
           for i=1:1:num_a_to_sample
               Q(i,1) = borl.getQ(state, actions_sampled(:,i),0);
           end
           [minQ,minA] = min(Q);
           a = actions_sampled(:,minA);
           q = minQ;
        end
        
        function Q = getQ(borl, state, action, justMu)
           muQ = borl.muQ_bootstrapped(state, action);
           sigmaQ = borl.sigmaQ_bootstrapped(state, action);
           
           if (size(borl.xQ_mismatch,1)>1)
               k_mismatch = borl.kernel_mismatch(repmat([state', action'], size(borl.xQ_mismatch,1),1), borl.xQ_mismatch);
               mismatchMu = k_mismatch'*borl.KQ_mismatch_inv_times_yQ;
               
               Q = muQ + mismatchMu + (1-justMu)*(sigmaQ + (borl.kernel_mismatch([state',action'],[state',action'])...
                                                            - k_mismatch'*borl.KQ_mismatch_inv*k_mismatch))*randn();
           else
               Q = muQ + (1-justMu)*sigmaQ*randn();   
           end
        end
        
        function Qvariance = getQvariance(borl, state, action, justMismatch)
            % Utility function, to monitor the variance in (bootstrapped + mismatch)/only mismatch GP
            
            [~, sigmaQ] = borl.musigmaQ_bootstrapped(state, action);
            k_mismatch = borl.kernel_mismatch(repmat([state', action'], size(borl.xQ_mismatch,1),1), borl.xQ_mismatch);
            Qvariance = sigmaQ*(1-justMismatch) + ...
                        (borl.kernel_mismatch([state',action'],[state',action']) - ...
                        k_mismatch'*borl.KQ_mismatch_inv*k_mismatch);
        end
    end
end