function [MeanY1,LogY1,InvY1]= CalMeanLogInv(SY1)
% CalMeanLogInv: calculate the mean value, log and inverse of Covariance matrices
%
%	Usage:
%       [MeanY1,LogY1,InvY1]= CalMeanLogInv(SY1)
%
%             Input: 
%               SY1              - Data cell                               
%
%             Output:
%               MeanY1           - mean value
%               LogY1            - log of Cov. matrix
%               InvY1            - inverse of Cov. matrix
%
%   Reference:
%
%   Zhiwu Huang, Ruiping Wang, Shiguang Shan,  Xilin Chen. 
%   Learning Euclidean-to-Riemannian Metric for Point-to-Set Classification.  
%   In Proc. CVPR 2014.
%
%   Written by Zhiwu Huang (zhiwu.huang@vipl.ict.ac.cn)
%

num =length(SY1);
dim = size(SY1{1},1);
MeanY1 = zeros(dim,num);
LogY1 = zeros(dim,dim,num);
InvY1 = zeros(dim,dim,num);

for tmpC1=1:num
    Y1=SY1{tmpC1};
    Y1_mu = mean(Y1,2);
    MeanY1(:,tmpC1) = Y1_mu;
    
    Y1 = Y1-repmat(Y1_mu,1,size(Y1,2));
    Y1 = Y1*Y1'/(size(Y1,2)-1);
    lamda = 0.001*trace(Y1);
    Y1 = Y1+lamda*eye(size(Y1,1)); 
    
    LogY1 (:,:,tmpC1) = logm(Y1);

%     [U, S, V] = svd(Y1);
%     diags = full(diag(S));
% 
%     for j = 1 : length(diags)
%         diags(j) = log(diags(j)); %% log(C)
%     end
%     logs= diag(diags);
%     LY1(:,:,tmpC1)= U*logs*U';
    
    InvY1(:,:,tmpC1) = inv(Y1);    
    
end