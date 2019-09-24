function [Wx, Wy,M] =  myLERM_modifyM_21(dataCell,options,Wxo,Wyo,dim,my_feather)
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

options.intraK = options.intraKy; 
options.interK = options.interKy; 
[LWy, LBy] = CalAffinityMatrix(Ly, options, Xy');  %Ay, Xy
LWy = (betay/Ny)*LWy;
LBy = (betay/Ny)*LBy;

% Compute the R matrices
RxW = SxW + 2*LWx;
RyW = SyW + 2*LWy;
RxB = SxB + 2*LBx;
RyB = SyB + 2*LBy;

% Compute the big M matrices
MW = [Xx*RxW*Xx' -Xx*UW*Xy'; -Xy*UW'*Xx' Xy*RyW*Xy'];
MB = [Xx*RxB*Xx' -Xx*UB*Xy'; -Xy*UB'*Xx' Xy*RyB*Xy'];

nIter = options.nIter;
lamda2 = options.lamda2;
lamda1=0.01;

Wx=Wxo;
Wy=Wyo;

myvalue=[];
lamda2=0.3;
nIter=4;
lamda3=0.1;
myinterI=1;

V_U=eye(dim);
V_U_x=eye(dim);
V_U_y=randn(dim,dim);

M=V_U;
Mx=V_U_x;
My=V_U_y;

%---------------------
AA=RxW-RxB;
BXX=sum(AA,2);
BXX_A=diag(BXX);
BYY=sum(AA,1);
BYY_A=diag(BYY);
lamda1_1=1;
lamda1_2=1;
%--------------------

for i = 1 : nIter
    
        %compute the V  
        f1=Wx'*Xx*(RxW-RxB)*Xx'*Wx ;  % G2
        f2=Wy'*Xy*(RyW-RyB)*Xy'*Wy  ; % G3
        
        f3=Wx'*Xx*(SxW-SxB)*Xx'*Wx + Wy'*Xy*(SyW-SyB)*Xy'*Wy  -2*(Wx'*Xx*(UW-UB)*Xy'*Wy);     
        Jall_mo=lamda1_1*f1+lamda1_2*f2+f3;
        Jall_mx=lamda1_1*f1;
        Jall_my=lamda1_2*f2; 
        
    for interI=1:myinterI


        %------------------------------
        M=inv(V_U)*(-1)*Jall_mo';
        [eigvectorJ, eigvalueJ] = eig(M);   %  
        M=eigvectorJ*eigvectorJ';    % class - 1
        % update V_U
        myV_U=diag(M*M');
        V_U=inv(diag(myV_U))*0.5;
        %-------------------------------
        Mx=inv(V_U_x)*(-1)*Jall_mx';
        [eigvectorJx, eigvalueJx] = eig(Mx);   % 
        Mx=eigvectorJx*eigvectorJx';    % class - 1
        % update V_U
        myV_Ux=diag(Mx*Mx');
        V_U_x=inv(diag(myV_Ux))*0.5;

    end

   %update Wx
   %  A*X + X*B = C,
   BBx=M; 
   AAx=lamda2* inv(    Xx* (RxW-RxB+lamda2*eye(Ny))  *Xx'  )*(Xx*Xx');
   CCx=2*inv(    Xx* (RxW-RxB+lamda2*eye(Ny))  *Xx'  )    *Xx*(UW-UB)*Xy'*Wy*M;
   
   Wx = sylvester(AAx,BBx,CCx);
   
   %update Wy
   BBy=M;
   AAy=lamda2*inv(    Xy* (RyW-RyB+lamda2*eye(Ny))  *Xy'  )  *(Xy*Xy');
   CCy=2* inv(    Xy* (RyW-RyB+lamda2*eye(Ny))  *Xy'  )   *Xy*(UW-UB)*Xx'*Wx*M;
   Wy = sylvester(AAy,BBy,CCy); 

   
   
    f1=Wx'*Xx*(RxW-RxB)*Xx'*Wx;
    f2=Wy'*Xy*(RyW-RyB)*Xy'*Wy;
    f3=2*(Wx'*Xx*(UW-UB)*Xy'*Wy);
    f4=lamda2*Wx'*Xx*Xx'*Wx+lamda2*Wy'*Xy*Xy'*Wy;

end



