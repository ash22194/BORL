classdef BORLAlg1 < handle
    properties
        env
        muQ      % bootstrapped mean function for Q values, takes in the state and the action
        sigmaQ   % bootstrapped sigma function for Q values, takes in the state and action
        policy
        actions
    end
    
    methods
        function borl = BORLAlg1(env)
            borl.env = env;
        end
        
        function policy = train_policy(borl, maxiter, horizon, debug_)
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
                while(~is_terminal && t<horizon)
                    t = t+1;
                    Q = borl.muQ(state,borl.actions) + randn(size(borl.actions,1),1).*borl.sigmaQ(state,borl.actions);
                    [minQ, minA] = min(Q);
                end
            end
        end
    end
end