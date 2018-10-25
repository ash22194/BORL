classdef GPDP < handle

    properties
        env
        gamma_
        sigmae % scalar
        D      % Dictionary of states for initializing GPs
        gpq
        gpv
    end
    
    methods 
        function k = kernel(gptd,x,y)
            % x,y are nx1 inputs that represent states
            k = exp(-sum((x-y).^2,1)/(2*gptd.sigmak^2));
        end
        
    
    end






end