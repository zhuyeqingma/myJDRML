function [outKernel, sigma_s2v]=CalEu2RieKernel(datax,cmmean,cminv,options)
% CalEu2RieKernel:	calculate the Euclidean-to-Riemannian kernel
%
%	Usage:
%       [outKernel, sigma_v2v]=CalRieKernel(CY1,CY2,options)
%
%             Input: 
%               datax             - Euclidean data (points)
%               cmmean            - mean datas of sets
%               cminv             - inverse of Cov. matrix  (set models, Riemannian data)
%                                    
%               options.t         - width of RBF kernel 
%                 
%                                    
%
%             Output:
%               outKernel         - RBF Euclidean-to-Riemannian kernel
%               sigma_s2v         - output the width of RBF kernel
%
%   Reference:
%
%   Zhiwu Huang, Ruiping Wang, Shiguang Shan,  Xilin Chen. 
%   Learning Euclidean-to-Riemannian Metric for Point-to-Set Classification.  
%   In Proc. CVPR 2014.
%
%   Written by Zhiwu Huang (zhiwu.huang@vipl.ict.ac.cn)
%



cmxnum = size(cmmean,2);
daxnum = size(datax,2);

D = ones(daxnum,cmxnum);
for i = 1 : daxnum
    for j = 1 : cmxnum
        datai = datax(:,i)';
        datamu = cmmean(:,j)';
        datainv = cminv(:,:,j);
        D(i,j) = sqrt((datai-datamu)*datainv*(datai-datamu)'); %Mahalanobis distance
    end
end
sigma_s2v = 0;
if ~isfield(options,'t')
    options.t = sqrt(mean(D(:)));
    sigma_s2v = options.t;
end
outKernel = exp(-D/(2*options.t^2));