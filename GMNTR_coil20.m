clear
clc
load('COIL20.mat')
rng('default')
N=20;%总类数
% fea1= NormalizeFea(fea);
fea1=fea;
ug=unique(gnd);
X=fea1';
C=20;
[n,m]=size(X);
l=22*C;

tic

r1=2;%链接维数
r2=2;
r3=10;

NKN=10;
iter=10;
MIAPG=zeros(iter,NKN);
ACAPG=zeros(iter,NKN);
PUR=zeros(iter,NKN);
rng('default')

result=zeros(iter+1,6);
opt=[];
opt.k=C;
opt.eps=1e-5;
opt.Maxiter=100;
opt.lambda=1000;        % 图正则化参数
opt.alpha_label=1000;  % 标签约束参数
opt.sim_alpha=0.5;     % 类内相似度权重参数 (新增)
K=3;

for j4=1:iter
    %构造图初始化
    % m=380;
    [Xiter,gnditer]=randpermData(X,gnd,C,m,l);
    [n3,n1n2]=size(fea1);
    TY=reshape(Xiter,[sqrt(n1n2),sqrt(n1n2),n3]);

    n=size(TY);
    A0=rand(r1,n(1),r2);
    B0=rand(r2,n(2),r3);
    C0=rand(r3,n(3),r1);
    core0={A0;B0;C0};

    [~, W, ~] = CANW(Xiter, C, K);
    W=refineW(W,l,C);%W_{ab}
    A=diag(sum(W,2));%D_{ab}

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
% xlswrite("GMNTR_COIL20.xlsx", title)
% writematrix(re2, "GMNTR_COIL20.xlsx", "WriteMode", "append")
% writematrix(err',"GMNTR_COIL20_err.xlsx","WriteMode","append")

Y = tsne(V(1:l,:));                  % 得到的矩阵为Nx2，N为N个样本，Y矩阵为320x2
MM=M_tilde(:,1:l);
Y=tsne(MM');
gscatter(Y(:,1), Y(:,2),gnditer(1:l));% 若无label输入，则画出的图没有色彩区分


toc