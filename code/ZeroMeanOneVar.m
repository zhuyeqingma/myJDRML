function M = ZeroMeanOneVar(X,x_mean,x_var)
% ZeroMeanOneVar: normalize the data X
%
%
index = find(x_var<0.0000001);
x_var(index) = 1;
% x_mean(index) = 0;

num = size(X,2);
X_mean = repmat(x_mean, 1, num);
X_var = repmat(x_var, 1, num);
M = X - X_mean;
M = M ./ X_var;

return;
flag = isnan(M);
M(flag) = 0;