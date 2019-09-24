function [outKernel, sigma_v2v]=CalRieKernel(CY1,CY2,options)
% CalRieKernel:	calculate the Riemannian kernel for Covariance matrices
%
%	Usage:
%       [outKernel, sigma_v2v]=CalRieKernel(CY1,CY2,options)
%
%             Input: 
%               CY1,CY2            - sets of Cov. matrices
%                                    
%               options.t          - width of RBF kernel 
%                 
%                                    
%
%             Output:
%               outKernel         - RBF Riemannian kernel
%               sigma_v2v         - output the width of RBF kernel
%
%   Reference:
%
%   Zhiwu Huang, Ruiping Wang, Shiguang Shan,  Xilin Chen. 
%   Learning Euclidean-to-Riemannian Metric for Point-to-Set Classification.  
%   In Proc. CVPR 2014.
%
%   Written by Zhiwu Huang (zhiwu.huang@vipl.ict.ac.cn)
%


if(isempty(CY2))
    CY2 = CY1;
end
number_sets1=size(CY1,3);
number_sets2=size(CY2,3);

D=zeros(number_sets1,number_sets2);
for tmpC1=1:number_sets1
    Y1=CY1(:,:,tmpC1);
    for tmpC2=1:number_sets2
        Y2=CY2(:,:,tmpC2);
        %riemannian kernel
        D(tmpC1,tmpC2)=trace(Y1*Y2);
%         Y = Y1-Y2;
%         D(tmpC1,tmpC2) = norm(Y(:)); %Log-Euclidean Riemannian metric      
    end
end
sigma_v2v = 0;
if ~isfield(options,'t')
    options.t = mean(D(:));
    sigma_v2v = options.t;
end
outKernel = exp(-D.^2/(2*options.t^2));%%%
outKernel=D;