
clear;
clc;
addpath(genpath('dataset'))
load Extended_yaleB
load random_index_yaleB

class=28;
totalNum=9;
trainN=3;

TrainV.X=cell(class*trainN,1);
TrainV.y=zeros(class*trainN,1);
TestV.X=cell(class*(totalNum-trainN),1);
TestV.y=zeros(class*(totalNum-trainN),1);
testNum=class*(totalNum-trainN);

result=[];
for interj=1:5
ts=0;
for i=1:class
    for j=1:trainN
        ts=ts+1;
        TrainV.X{ts,1}=Extended_yaleB_PCA{i,random_index_yaleB{interj}(i,j)}  ;
        TrainV.y(ts)=i;
    end
end
ts=0;
for i=1:class
    for j=trainN+1:totalNum
        ts=ts+1;
        TestV.X{ts,1}=Extended_yaleB_PCA{i,random_index_yaleB{interj}(i,j)}    ;
        TestV.y(ts)=i;
    end
end

%% Generating Euclidean, Riemannian and Euclidean-to-Riemannian kernels
[M, LogC, InvC] = CalMeanLogInv(TrainV.X);
[M_t, LogC_t, InvC_t] = CalMeanLogInv(TestV.X);
options = [];
[K_S, sigma_s2s]= CalEuKernel(M',[],options);  
K_S_t= CalEuKernel(M_t',M',options);       
options = [];
[K_V, sigma_v2v] = CalRieKernel(LogC,[],options);   
options.t = sigma_v2v;
K_V_t = CalRieKernel(LogC_t,LogC,options);         
options = [];
[K_S_V, sigma_s2v]= CalEu2RieKernel(M,M,InvC,options);  
options.t = sigma_s2v;
K_S_V_t = CalEu2RieKernel(M_t,M,InvC,options);            
K_V_S_t = CalEu2RieKernel(M,M_t,InvC_t,options);        

%% Using different Mapping Modes
% Mapping Mode 1
% K_S_f = TrainS.X';  K_V_f = K_V; K_S_t_f = TestS.X'; K_V_t_f = K_V_t;
% Mapping Mode 2
 K_S_f = K_S; K_V_f = K_V; K_S_t_f = K_S_t; K_V_t_f = K_V_t;
% Mapping Mode 3
% K_S_f = [K_S  K_S_V]; K_V_f = [K_V  K_S_V'];  K_S_t_f = [K_S_t K_S_V_t]; K_V_t_f = [K_V_t K_V_S_t'];
%% Train LERM model and obtain projections

dim =class-1;
[Wxo,Wyo] = LERM_Train(K_S_f,K_V_f,TrainV.y,TrainV.y,dim);
my_feather=size(K_S_f,2);
[model,Wx,Wy,V] = myLERM_Train(K_S_f,K_V_f,TrainV.y,TrainV.y,Wxo,Wyo,dim,my_feather);
[Test_S_pro, Test_V_pro] = LERM_Proj(K_S_t_f,K_V_t_f,model,dim,V);
[Train_S_pro, Ttain_V_pro] = LERM_Proj(K_S_f,K_V_f,model,dim,V);
 sim_mat1 = pdist2(Test_S_pro',Train_S_pro','mahalanobis',V);
 sim_mat2 = pdist2(Test_V_pro',Ttain_V_pro','mahalanobis',V);
 sim_mat=sim_mat2 +0.001*sim_mat1;
 [sim, ind] = sort(sim_mat,2);
Gallery_L=TrainV.y';
Probe_L = TestV.y';
Gallery_L(ind(:,1));
correctNum = length(find((Probe_L-Gallery_L(ind(:,1)))==0));
fRate1 = correctNum/testNum;

result=[result fRate1];
end
resultmean=mean(result)

