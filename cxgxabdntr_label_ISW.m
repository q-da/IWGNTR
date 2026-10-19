function [U_proj, V, core1, ob, iter] = cxgxabdntr_label_ISW(X, TY, core0, opt, A, W, M_tilde, P)
% 带类内加权标签矩阵约束的非负张量环分解
%Intra-class Weighted Graph Regularization for Non-negative Tensor Ring Decomposition
% 输入:
%   X       - 数据矩阵 (特征数 x 样本数)
%   TY      - 张量形式的数据
%   core0   - 初始核心张量
%   opt     - 参数结构体
%   A       - 度矩阵 D
%   W       - 相似度矩阵 S
%   M_tilde - 类内加权标签矩阵 (c x nSmp)
%   P       - 选择矩阵 (l x n)
%
% 输出:
%   U       - 组合矩阵
%   V       - 样本表示矩阵
%   core1   - 优化后的核心张量
%   ob      - 目标函数值历史
%   iter    - 实际迭代次数

eps = opt.eps;
Maxiter = opt.Maxiter;
lambda = opt.lambda;
alpha = opt.alpha_label;

n = size(X, 2);
c = size(M_tilde, 1);
l = size(P, 1);

% 计算拉普拉斯矩阵
L = A - W;

sizecore0 = size(core0, 1);
core1 = core0;

% 存储每个核心张量的size
sg = cell(sizecore0, 1);
for k1 = 1:sizecore0
    sg{k1, 1} = size(core0{k1, 1});
end

% 对每个核心张量进行mode-2展开
G22 = cell(sizecore0, 1);
for k2 = 1:sizecore0
    G22{k2, 1} = classical_mode_unfolding(core1{k2, 1}, 2);
end
G20 = G22;

% 对张量TY进行不同模式的展开
Xmod = cell(sizecore0, 1);
for k3 = 1:sizecore0
    Xmod{k3, 1} = mode_unfolding(TY, k3);
end

% 初始化投影矩阵 U_proj
r_N1 = size(core1{end}, 1);
r_N = size(core1{end}, 3);
d_H = r_N1 * r_N;
U_proj = rand(c, d_H);

% 提取标记样本对应的加权标签矩阵
M_label = M_tilde(:, 1:l);  % 只取前l列（标记样本）

GK4 = cell(sizecore0, 1);
GGK = cell(sizecore0, 1);

obj0 = 0;
ob = zeros(Maxiter + 1, 1);
ob(1) = objapggsntr_label_ISW(TY, core1, L, lambda, M_label, P, U_proj, alpha);

for k = 1:Maxiter
    for k4 = 1:sizecore0
        GK4{k4, 1} = subchain_matrix(core1, k4);
        
        if k4 == sizecore0
            % 样本维度的核心张量
            GU = GK4{k4, 1};
            UU = GU' * GU;
            V = G22{k4, 1};

            
            % 提取标记样本的表示
            V_label = P * V;
            
    
            
            % 更新 V
            numerator = Xmod{k4, 1} * GU + lambda * W * V +  alpha * P' * (M_label' * U_proj);
            denominator = V * UU + lambda * A * V +  alpha * P' *(P* V*(U_proj' * U_proj));
            V = V .* (numerator ./ max(denominator, eps));

              % 更新 U_proj
            numerator_U = M_label * V_label;
            denominator_U = U_proj * (V_label' * V_label);
            U_proj = U_proj .* (numerator_U ./ max(denominator_U, eps));

            
            % % 更新 U
            % U = U .* (M_label * V_label) ./ max(U * V_label'*V_label, eps);
            % 
           
            
            G22{k4, 1} = V;
            
        else
            % 非样本维度的核心张量
            GGK{k4, 1} = GK4{k4, 1}' * GK4{k4, 1};
            G22{k4, 1} = G22{k4, 1} .* (Xmod{k4, 1} * GK4{k4, 1}) ./ ...
                         max(G22{k4, 1} * GGK{k4, 1}, eps);
            GK4{k4, 1} = GK4{k4, 1} .* (Xmod{k4, 1}' * G22{k4, 1}) ./ ...
                         max(GK4{k4, 1} * G22{k4, 1}' * G22{k4, 1}, eps);
            G20{k4, 1} = G22{k4, 1};
        end
        
        G22{k4, 1} = max(0, G22{k4, 1});
        core1{k4, 1} = classical_mode_folding(G22{k4, 1}, 2, sg{k4, 1});
    end
    
    obj1 = objapggsntr_label_ISW(TY, core1, L, lambda, M_label, P, U_proj, alpha);
    ob(k + 1) = obj1;
    err1 = abs((obj1 - obj0));
    
    disp(['the ', num2str(k), ' obj is ', num2str(ob(k)), ' cost is ', num2str(err1)]);
    
    if err1 / max(obj1, 1) < eps
        break;
    else
        obj0 = obj1;
    end
end

iter = k;
end