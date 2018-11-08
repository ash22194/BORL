classdef BORLAlg1 < handle
    properties
        dynamics % function that takes a vector of states and a vector of actions and returns a vector of next states
        reward   % cost
        muQ      % bootstrapped mean function for Q values, takes in the state and the action
        sigmaQ   % bootstrapped sigma function for Q values, takes in the state and action
        policy
    end
    
    methods
        function borl = BORLAlg1(dynamics, reward)
            borl.dynamics = dynamics;
            borl.reward = reward;
            
        end
        
        function policy = train_policy(maxiter, horizon, debug)
            iter = 0;
            while(iter<maxiter)
                iter = iter+1;
                for t=1:1:horizon
                    
                end
            end
        end
    end
end