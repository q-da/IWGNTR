function obj = objapggsntr_label_ISW(X, cores, L, lambda, M_label, P, U_proj, alpha)
% 带类内加权标签矩阵约束的目标函数
% 输入:
%   X       - 原始张量
%   cores   - 核心张量
%   L       - 拉普拉斯矩阵
%   lambda  - 图正则化参数
%   M_label - 类内加权标签矩阵 (c x l)
%   P       - 选择矩阵 (l x n)
%   U_proj  - 投影矩阵 (c x d_H)
%   alpha   - 标签约束参数
% 输出:
%   obj     - 目标函数值

N = size(cores, 1);

% 1. 重构误差项
Y = cores_2_tensor(cores);
XMY = X - Y;
XMY = tensor(XMY);
FXMY = ttt(XMY, XMY, 1:N);

% 2. 图正则化项
GN2 = classical_mode_unfolding(cores{N, 1}, 2);
WLW = GN2' * L * GN2;
TWLW = trace(WLW);

% 3. 类内加权标签约束项
H_label = P * GN2;  % 标记样本的表示 (l x d_H)

% 计算加权残差
residual = M_label - U_proj * H_label';
Tlabel = 0.5 * norm(residual, 'fro')^2;

% 总目标函数
obj = 0.5 * FXMY + 0.5 * lambda * TWLW + 0.5 * alpha * Tlabel;

end