% min_{A>=0, A*1=1, F'*F=I}  trace(D'*A) + r*||A||^2 + 2*lambda*trace(F'*L*F)
% written by Feiping Nie on 2/9/2014
function [y, A, evs] = CANW(X, c, k, r, islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% r: paremeter, which could be set to a large enough value. If r<0, then it is determined by algorithm with k
% islocal:
%           1: only update the similarities of the k neighbor pairs, faster
%           0: update all the similarities
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations

% For more details, please see:
% Feiping Nie, Xiaoqian Wang, Heng Huang.
% Clustering and Projected Clustering with Adaptive Neighbors.
% The 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), New York, USA, 2014.

%%%X=Xiter,C=20,k=3
addpath(genpath('\funs'))

NITER = 30;
num = size(X,2);
if nargin < 5
    islocal = 1;
end;
if nargin < 4
    r = -1;
end;
if nargin < 3
    k = 15;
end;

%每个数据点之间的距离||xi-xj||_2^2
distX = L2_distance_1(X,X);
%distX = sqrt(distX);
%从小到大排序，相当于数据之间的相似度随着数字的增大而减小
[distX1, idx] = sort(distX,2);%每行元素从小到大排序

A = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);%相当于d_ij
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));%更新r。
    id = idx(i,2:k+2);%将距离按从小到大排序后对应的索引赋值给id
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    % A1(i,id) = 1/(2*k*rr(i))*(2*rr(i)-k*di+sum(di(1:k)));
end;
%A=abs(A);
if r <= 0
    r = mean(rr);
end
lambda = mean(rr);%求r的均值

A0 = (A+A')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;
[F, temp, evs]=eig2(L0, c, 0);

if sum(abs(evs(1:c+1))) < 1e-18%0.00000000001%1e-18
    error('The original graph has more than %d connected component', c);
end
% A=A0;

for iter = 1:NITER
    % distf = L2_distance_1(F',F');
    % distf=zeros(num);
    A = zeros(num);
    for i=1:num
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:num;
        end;
        % dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -dxi/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;
    %A=abs(A);
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    F_old = F;
    [F, temp, ev]=eig2(L, c, 0);
    %evs(:,iter+1) = ev;

    fn1 = sum(abs(ev(1:c)));
    fn2 = sum(abs(ev(1:c+1)));
    if fn1 > 0.00000000001
        lambda = 2*lambda;
    elseif fn2 < 0.00000000001
        lambda = lambda/2;  F = F_old;
    else
        break;
    end;

end;

%[labv, tem, y] = unique(round(0.1*round(1000*F)),'rows');
[clusternum, y]=conncomp(graph(sparse(A))); y = y';
if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', c)
end;