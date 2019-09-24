function [Wx, Wy] =  LERM(dataCell,options,dim)
% LERM:	Leaning Euclidean-to-Riemannian Metric
%
%	Usage:
%       [Wx Wy] =  LERM(dataCell,options)
%
%             Input: 
%               dataCell            
%                 data             - Data matrix of Euclidean and Riemannian samples.
%                                    Each column vector of fea is a data point.
%                 label            - labels of Euclidean and Riemannian samples.
%
%               options    
%                 betax, betay     - regularization parameter: lamda1_x, lamda1_y
%                 lamda2           - regularization parameter: lamda2
%                 intraKx,intraKy  - num. of within-class neighborhood:  k1
%                 interKx, interKy - num. of between-class neighborhood: k2
%                 nIter            - num. of iterations 
% 
%                                    
%
%             Output:
%               Wx, Wy - Projections of heterogeneous samples.
if ~isfield(options,'betax')
    betax = 0.2;%0.2, 0.10
else
    betax = options.betax;
end
if ~isfield(options,'betay')
    betay = 0.2;%0.2, 0.15
else
    betay = options.betay;
end
Xx = dataCell{1,1}.data;
Xy = dataCell{2,1}.data;
Lx = dataCell{1,1}.label;
Ly = dataCell{2,1}.label;


% sample sizes and dimensions
[dx, Nx] = size(Xx);
[dy, Ny] = size(Xy);



Lxx = repmat(Lx,1,Ny);
Lyy= repmat(Ly',Nx,1);
mask = Lyy==Lxx;

N1 = sum(mask(:));
N2 = Nx*Ny-N1;
U = ones(Nx,Ny);

UW = 1/N1*U;
UB = 1/N2*U;
UW(mask==0) = 0;
UB(mask==1) = 0;

U(mask==0) = -1;
% Lxx, lyy
for i =1:size(Lxx,1)
    U(i,i)=0;
end

myA=U;
myBx=diag(sum(myA,2));
myBy=diag(sum(myA,2));


% UW
% UB
% UW-UB
SxW = diag(sum(UW,2));
SyW = diag(sum(UW,1));
SxB = diag(sum(UB,2));
SyB = diag(sum(UB,1));

% Compute the affinity matrix
options.intraK = options.intraKx; 
options.interK = options.interKx; 
[LWx, LBx] = CalAffinityMatrix(Lx, options, Xx');%Ax, Xx

LWx = (betax/Nx)*LWx;
LBx = (betax/Nx)*LBx;

myAx=-(LWx-LBx);
for i =1:size(myAx,1)
    myAx(i,i)=0;
end
myLx=diag(sum(myAx,2))-myAx;



options.intraK = options.intraKy; 
options.interK = options.interKy; 
[LWy, LBy] = CalAffinityMatrix(Ly, options, Xy');  %Ay, Xy

LWy = (betay/Ny)*LWy;
LBy = (betay/Ny)*LBy;

myAy=-(LWy-LBy);
for i =1:size(myAy,1)
    myAy(i,i)=0;
end
myLy=diag(sum(myAy,1))-myAy;



% Compute the R matrices
RxW = SxW + 2*LWx;
RyW = SyW + 2*LWy;

RxB = SxB + 2*LBx;
RyB = SyB + 2*LBy;

% Compute the big M matrices
MW = [Xx*RxW*Xx' -Xx*UW*Xy'; -Xy*UW'*Xx' Xy*RyW*Xy'];
MB = [Xx*RxB*Xx' -Xx*UB*Xy'; -Xy*UB'*Xx' Xy*RyB*Xy'];

% Initialize the projections by solving the generalized eigen-decomposition problem
[W, d]= GenEig(MW,MB);

Wx = W(1:dx,:);
Wy = W(dx+1:end,:);


nIter = options.nIter;
lamda2 = options.lamda2;

lamda1=0.01;
for i = 1 : nIter
    %update Wx
    Wx = (Xx*(myBx+2*lamda1*myLx+lamda2*eye(Ny))*Xx')\(Xx*myA*Xy'*Wy);

    %update Wy
    Wy = (Xy*(myBy+2*lamda1*myLy+lamda2*eye(Ny))*Xy')\(Xy*myA*Xx'*Wx);
end

    


