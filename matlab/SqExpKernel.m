classdef SqExpKernel < handle
    properties
        data
        K
        K_inv
        sigma_l
        sigma_n
        sigma_f
    end
    
    methods
        function k = SqExpKernel(data, sigma_l, sigma_n, sigma_f)
            k.data = data;
            k.K = zeros(size(data,1),size(data,1));
            k.sigma_l = sigma_l;
            k.sigma_n = sigma_n;
            k.sigma_f = sigma_f;
            for d=1:1:size(data,1)
                k.K(:,d) = k.distance(repmat(data(d,:),size(data,1),1), data);
            end
            k.K = sparse(k.K);
            k.K_inv = inv(k.K);
        end
        
        function d = distance(k, x1, x2)
            assert(size(x1,1)==size(x2,1), 'Numpoints in x1 and x2 must be same!');
            assert((size(x1,2)==length(k.sigma_l)) & (size(x2,2)==length(k.sigma_l)), strcat('Dimension of points in x1 and x2 must be ',num2str(length(k.sigma_l))));
            d = sparse(k.sigma_f*exp(-0.5*sum(((x1-x2)./repmat(k.sigma_l,size(x1,1),1)).^2,2)));
        end
        
        function addData(k, data)
            B = zeros(size(k.data,1),size(data,1));
            C = zeros(size(data,1),size(data,1));
            for d=1:1:size(data,1)
                B(:,d) = k.distance(k.data, repmat(data(d,:),size(k.data,1),1));
                C(:,d) = k.distance(data, repmat(data(d,:),size(data,1),1));
            end
            k.K = [k.K B;B' C];
            b = B'*k.K_inv;
            c = inv(C - b*B);
            e = k.K_inv*B;
            k.K_inv = [(k.K_inv + e*c*b) -e*c;-(e*c)' c];
        end
        
    end

end