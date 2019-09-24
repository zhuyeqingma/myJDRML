function [model,Wx,Wy,V] = myLERM_Train(kernel_x,kernel_y,label_x,label_y ,Wxo,Wyo,dim,my_feather)
[data_x, mu_x,sigma_x] = zscore(kernel_x);
[data_y, mu_y,sigma_y] = zscore(kernel_y);
options=[];
options.ReducedDim = 0;
[eigvector_x, eigvalue_x] = PCA(data_x, options);
[eigvector_y, eigvalue_y] = PCA(data_y, options);
pca_x = data_x*eigvector_x;
pca_y = data_y*eigvector_y;

%parameter setting
options = [];
options.betax = 0.01; %lamda1_x 
options.betay = 0.09; %lamda1_y
options.intraKx = 1; % k1 
options.interKx = 5; % k2
options.intraKy = 1; % k1
options.interKy = 5; % k2
options.nIter =3; % iteration num.
options.lamda2 = 0.1; % lamda2
dataCell = cell(2,1);
dataCell{1,1}.data = pca_x';
dataCell{2,1}.data = pca_y';
dataCell{1,1}.label = label_x;
dataCell{2,1}.label = label_y; 
[eigvector_lerm_x, eigvector_lerm_y,V] =  myLERM_modifyM_21(dataCell,options,Wxo,Wyo,dim,my_feather); % 2019-1-18

Wx=eigvector_lerm_x;
Wy=eigvector_lerm_y;
model.eigx = eigvector_lerm_x;
model.eigy = eigvector_lerm_y;
model.eigx_p = eigvector_x;
model.eigy_p = eigvector_y;
model.mux = mu_x;
model.sigx = sigma_x;
model.muy = mu_y;
model.sigy = sigma_y;

