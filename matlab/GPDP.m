classdef GPDP < handle

    properties
        dynamics % function that takes a vector of states and a vector of actions and returns a vector of next states
        reward   % cost
        states
        actions
        gamma_
        muV
        sigmaf_v
        sigman_v
        l_v
        K_v
        K_v_inv
        muQ
        sigmaf_q
        sigman_q
        l_q
        K_q
        K_q_inv
        policy
    end
    
    methods 
        function gpdp = GPDP(dynamics, reward, states, actions, V, Q, gamma_)
            gpdp.dynamics = dynamics;
            gpdp.reward = reward;
            gpdp.states = states;
            gpdp.actions = actions;
            gpdp.gamma_ = gamma_;
            
            % Compute MAP hyperparameters
            % meanfunc = [];
            % covfunc = @covSEiso;
            % likfunc = @likGauss;
            % hyp = struct('mean',[],'cov',[0,0],'lik',-1);
            % hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, states, V);
            % gpdp.l_v = exp(hyp.cov(1))^2;
            % gpdp.sigman_v = exp(hyp.lik)^2;
            % gpdp.sigmaf_v = exp(hyp.cov(2))^2;
            gpdp.l_v = 0.2;
            gpdp.sigman_v = 10^(-4);
            gpdp.sigmaf_v = 1;
            gpdp.K_v = zeros(size(states,1),size(states,1));
            for s=1:1:size(states,1)
                gpdp.K_v(:,s) = gpdp.kernelSE_v(states,repmat(states(s,:),size(states,1),1));
            end
            gpdp.K_v = sparse(gpdp.K_v);
            gpdp.K_v_inv = inv(gpdp.K_v + gpdp.sigman_v*eye(size(states,1)));
            gpdp.muV = V;
            
            % hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, actions, zeros(size(actions,1),1));
            % gpdp.l_q = exp(hyp.cov(1))^2;
            % gpdp.sigman_q = exp(hyp.lik)^2;
            % gpdp.sigmaf_q = exp(hyp.cov(2))^2;
            gpdp.l_q = 0.2;
            gpdp.sigman_q = 10^(-4);
            gpdp.sigmaf_q = 1;
            gpdp.K_q = zeros(size(actions,1),size(actions,1));
            for a=1:1:size(actions,1)
                gpdp.K_q(:,a) = gpdp.kernelSE_q(actions,repmat(actions(a,:),size(actions,1),1));
            end
            gpdp.K_q = sparse(gpdp.K_q);
            gpdp.K_q_inv = inv(gpdp.K_q + gpdp.sigman_q*eye(size(actions,1)));
            gpdp.muQ = Q;
            % gpdp.policy = actions(randi(size(actions,1),size(states,1),1)',:);
            gpdp.policy = randi(size(actions,1),size(states,1),1)';
            
        end
        
        function k = kernelSE_v(gpdp,x,y)
            % x,y are nx1 inputs that represent states
            k = sparse(gpdp.sigmaf_v*exp(-1/(2*(gpdp.l_v))*sum((x-y).^2,2)) + gpdp.sigman_v*(prod(x==y,2)));
        end
        
        function k = kernelSE_q(gpdp,x,y)
            % x,y are nx1 inputs that represent actions
            k = sparse(gpdp.sigmaf_q*exp(-1/(2*(gpdp.l_q))*sum((x-y).^2,2)) + gpdp.sigman_q*(prod(x==y,2)));
        end
        
        function build_policy(gpdp, max_iterations, debug_)
            
            for iter=1:1:max_iterations
                if (debug_)
                    iter
                end
                % states_ = gpdp.dynamics(gpdp.states, gpdp.policy);
                states_ = gpdp.dynamics(gpdp.states, gpdp.actions(gpdp.policy,:));
                % r = gpdp.reward(states_, gpdp.policy);
                r = gpdp.reward(states_, gpdp.actions(gpdp.policy,:));
                V = gpdp.K_v_inv*gpdp.muV;
                for s=1:1:size(gpdp.states,1)
                    k = gpdp.kernelSE_v(repmat(states_(s,:),size(gpdp.states,1),1),gpdp.states);
                    gpdp.muQ(s,gpdp.policy(s)) = r(s) + gpdp.gamma_*(k'*V);
                end
                disp('Q-update done');
                
                % Update policy... minimize the mean but how?
                Q = zeros(size(gpdp.states,1),size(gpdp.actions,1));
                V = gpdp.K_v_inv*gpdp.muV;
                for a = 1:1:size(gpdp.actions,1)
                    states_ = gpdp.dynamics(gpdp.states, repmat(gpdp.actions(a,:),size(gpdp.states,1),1));
                    r = gpdp.reward(states_, repmat(gpdp.actions(a,:),size(gpdp.states,1),1));
                    for s=1:1:size(gpdp.states,1)
                        k = gpdp.kernelSE_v(repmat(states_(s,:),size(gpdp.states,1),1),gpdp.states);
                        Q(s,a) = r(s) + gpdp.gamma_*(k'*V);
                    end
                end
                [~,minA] = min(Q,[],2);
                % gpdp.policy = gpdp.actions(minA);
                gpdp.policy = minA;
                % Update V function points... when updated actions can be anything, update for V would be different
                gpdp.muV = gpdp.muQ([1:1:size(gpdp.states,1)]' + (minA-1)*size(gpdp.states,1));
                disp('Policy update done');
            end
            
        end
    
    end

end