clear
clc
tic
load COIL100.mat
rng('default')
N=100;%总类数
nk=20;%选取的类数
m1=72;%每类所包含的图片数
[feank,gndnk]=xuancoil100(fea,gnd,N,nk,m1);
fea1= NormalizeFea(feank);
gnd1=gndnk;
ug=unique(gnd1);
X=fea1';
C=20;
m=1440;
n=1024;

l=22*C;

r1=2;%链接维数
r2=2;
r3=10;

iter=10;
NKN=10;
MIAPG=zeros(iter,NKN);
ACAPG=zeros(iter,NKN);
PUR=zeros(iter,NKN);

result=zeros(iter+1,6);

opt=[];
opt.k=C;
opt.eps=1e-5;
opt.Maxiter=100;
opt.lambda=100;        % 图正则化参数
opt.alpha_label=1000;  % 标签约束参数
opt.sim_alpha=0.5;     % 类内相似度权重参数 (新增)
K=8;
for j4=1:iter
    %构造图初始化
    m=1440;
    [Xiter,gnditer]=randpermData(X,gnd1,C,m,l);
    [n3,n1n2]=size(fea1);
    TY=reshape(Xiter,[sqrt(n1n2),sqrt(n1n2),n3]);
    % TY=tensor(TY);
    n=size(TY);
    A0=rand(r1,n(1),r2);
    B0=rand(r2,n(2),r3);
    C0=rand(r3,n(3),r1);
    core0={A0;B0;C0};
    % 构建图
    [~, W, ~] = CANW(Xiter, C, K);
    W=refineW(W,l,C);
    A=diag(sum(W,2));

    % 使用类内加权标签矩阵
    M_tilde = buildWeightedLabelMatrix_ISW(Xiter, gnditer, l, C, opt.sim_alpha);

    % === 构建选择矩阵 P (l x n) ===
    P = zeros(l, size(Xiter, 2));
    for j = 1:l
        P(j, j) = 1;
    end

    % 调用修改后的优化函数
    [U, V, core1, err, iter1] = cxgxabdntr_label_ISW(Xiter, TY, core0, opt, A, W, M_tilde, P);

    H = V;  % 样本表示矩阵
    Hnew = H(l+1:end, :);  % 未标记样本
    nClass = C;
    gndnew = gnditer(l+1:end);

    % K-means聚类
    for k5 = 1:NKN
        labelapg = kmeans(Hnew, nClass);
        result = ClusteringMeasure(gndnew, labelapg);
        MIAPG(j4, k5) = result(2);
        ACAPG(j4, k5) = result(1);
        PUR(j4, k5) = result(3);
    end
end

meanMIAPG = mean(mean(MIAPG));
meanACAPG = mean(mean(ACAPG));
meanPUR = mean(mean(PUR));
stdMIAPG = std2(MIAPG);
stdACAPG = std2(ACAPG);
stdPUR = std2(PUR);
re1 = [meanACAPG, stdACAPG, meanMIAPG, stdMIAPG, meanPUR, stdPUR, toc]

re2 = [meanACAPG, stdACAPG, meanMIAPG, stdMIAPG, meanPUR, stdPUR, toc, ...
    K, opt.lambda, opt.alpha_label, opt.sim_alpha];
% title = {'meanACAPG','stdACAPG','meanMIAPG','stdMIAPG','meanPUR','stdPUR',...
%     'toc','K','lambda','alpha_label','sim_alpha'};
% xlswrite("GMNTR_COIL100.xlsx", title)
% writematrix(re2, "GMNTR_COIL100.xlsx", "WriteMode", "append")
% writematrix(err',"GMNTR_COIL100_err.xlsx","WriteMode","append")

toc