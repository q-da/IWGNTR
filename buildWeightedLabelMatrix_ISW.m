function M_tilde = buildWeightedLabelMatrix_ISW(X, gnd, l, C, sim_alpha)
%% 构建类内加权标签矩阵 (Intra-class Similarity Weighting)
% 输入:
%   X         - 数据矩阵 (nfea × nSmp)
%   gnd       - 标签向量 (nSmp × 1)
%   l         - 有标签样本数
%   C         - 类别数
%   sim_alpha - 类内相似性权重参数
% 输出:
%   M_tilde   - 类内加权标签矩阵 (C × nSmp)

    [nfea, nSmp] = size(X);
    M_tilde = zeros(C, nSmp);
    
    % 获取唯一的类别标签
    ug = unique(gnd);
    
    %% 步骤1: 计算每个类的中心
    class_centers = zeros(nfea, C);
    for cc = 1:C
        % 找到属于第cc类的有标签样本
        class_idx = find(gnd(1:l) == ug(cc));
        if ~isempty(class_idx)
            class_samples = X(:, class_idx);
            class_centers(:, cc) = mean(class_samples, 2);
        else
            % 如果某类没有标签样本，使用所有有标签样本的中心
            warning('第%d类没有标签样本，使用全局中心替代', cc);
            class_centers(:, cc) = mean(X(:, 1:l), 2);
        end
    end
    
    %% 步骤2: 构建加权标签矩阵
    for cc = 1:C
        for j = 1:l
            if gnd(j) == ug(cc)
                x = X(:, j);
                x_norm = norm(x);
                
                if x_norm > eps
                    center = class_centers(:, cc);
                    
                    % 计算余弦相似度
                    if norm(center) > eps
                        sim = (x' * center) / (x_norm * norm(center));
                        sim = max(0, min(1, sim));  % 限制在[0,1]
                    else
                        sim = 0;
                    end
                    
                    % 改进的权重: 模长 × (1 + α × 相似度)
                    weight = x_norm * (1 + sim_alpha * sim);
                    M_tilde(cc, j) = weight;
                end
            end
        end
    end
    
    %% 显示统计信息
    fprintf('\n类内加权标签矩阵构建完成:\n');
    fprintf('  相似度权重参数 α = %.2f\n', sim_alpha);
    
    % 统计各类的权重
    for cc = 1:C
        class_weights = M_tilde(cc, 1:l);
        nonzeros = class_weights(class_weights > 0);
        if ~isempty(nonzeros)
            fprintf('  类别%d: 样本数=%d, 平均权重=%.2f, 范围=[%.2f, %.2f]\n', ...
                    cc, length(nonzeros), mean(nonzeros), min(nonzeros), max(nonzeros));
        end
    end
end