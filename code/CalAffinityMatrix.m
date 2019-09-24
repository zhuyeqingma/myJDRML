function [LA, LB] = CalAffinityMatrix(gnd, options, data)
%
%       [LA, LB] = CalAffinityMatrix(gnd, options, data)
% 
%             Input:
%               data    - Data matrix. Each row vector of fea is a data point.
%
%               gnd     - Label vector.  
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                 intraK         = 0  
%                                     Sc:
%                                       Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                > 0  Sc:
%                                       Put an edge between two nodes if
%                                       they belong to same class and they
%                                       are among the intraK neareast neighbors of
%                                       each other in this class.  
%                 interK         = 0  Sp:
%                                       Put an edge between two nodes if and
%                                       only if they belong to different classes. 
%                                > 0
%                                     Sp:
%                                       Put an edge between two nodes if
%                                       they rank top interK pairs of all the
%                                       distance pair of samples belong to
%                                       different classes 
%
%
%             Output:
%               LA, LB - corresponding affinity matrices



bGlobal = 0;
if ~exist('data','var')
    bGlobal = 1;
    global data;
end

if (~exist('options','var'))
   options = [];
end


[nSmp,nFea] = size(data);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

intraK = 5;
if isfield(options,'intraK') 
    intraK = options.intraK;
end

interK = 20;
if isfield(options,'interK') 
    interK = options.interK;
end

tmp_T = cputime;

Label = unique(gnd);
nLabel = length(Label);


% D = EuDist2(data,[],0);
% D = data;
D = RBF_fast(data',data');


nIntraPair = 0;
if intraK > 0
    G = zeros(nSmp*(intraK+1),3);
    idNow = 0;
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        DClass = D(classIdx,classIdx);
        [dump idx] = sort(DClass,2,'descend'); 
        clear DClass dump;
        nClassNow = length(classIdx);
        nIntraPair = nIntraPair + nClassNow^2; 
        if intraK < nClassNow
            idx = idx(:,1:intraK+1);
        else
            idx = [idx repmat(idx(:,end),1,intraK+1-nClassNow)];
        end

        nSmpClass = length(classIdx)*(intraK+1);
        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[intraK+1,1]);
        G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
        G(idNow+1:nSmpClass+idNow,3) = 1;
        idNow = idNow+nSmpClass;
        clear idx
    end
    Sc = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    [I,J,V] = find(Sc);
    Sc = sparse(I,J,1,nSmp,nSmp);
    Sc = max(Sc,Sc');
    clear G
else
    Sc = zeros(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        nClassNow = length(classIdx);
        nIntraPair = nIntraPair + nClassNow^2;
        Sc(classIdx,classIdx) = 1;
    end
end

Sc = full(Sc);
LA = Sc.*D;
Dc = diag(sum(LA,2));
LA = -LA;
for i=1:size(Sc,1)
    LA(i,i) = LA(i,i) + Dc(i,i);
end


Sp = zeros(nSmp,nSmp);
if interK > 0 & (interK < (nSmp^2 - nIntraPair))
    minD = min(min(D))-0.9;
    D_t = D;
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        D_t(classIdx,classIdx) = minD;        
    end
    for i = 1 : nSmp        
        [dump,idx] = sort(D_t(i,:),'descend');
        idx = idx(1:interK);
        Sp(i,idx) = 1;
        Sp(idx,i) = 1;
    end
    
%     [dump,idx] = sort(D_t(:),'descend');
%     idx = idx(1:interK);
%     [I, J] = ind2sub([nSmp,nSmp],idx);
%     Sp = sparse(I,J,1,nSmp,nSmp);
%     Sp = max(Sp,Sp');
    
else
    Sp = ones(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        Sp(classIdx,classIdx) = 0;
    end
end

% save Sc.mat Sc;
% save Sp.mat Sp;

Sp = full(Sp);
LB = Sp.*D;
Dp = diag(sum(LB,2));
LB = - LB;
for i=1:size(Sp,1)
    LB(i,i) = LB(i,i) + Dp(i,i);
end


timeW = cputime - tmp_T;



