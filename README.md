# IWGNTR

**Intra-class Weighted Graph Regularization for Non-negative Tensor Ring Decomposition**

本仓库提供论文 **《Intra-class Weighted Graph Regularization for Non-negative Tensor Ring Decomposition》** 的 MATLAB 实现。方法简称为 **IWGNTR**。

## 项目简介

IWGNTR 面向非负张量环分解（Non-negative Tensor Ring Decomposition），引入类内加权图正则化，用于提升表示学习与聚类性能。仓库包含 IWGNTR 的主函数实现，以及用于 COIL-20 和 COIL-100 数据集的实验运行文件。

## 文件说明

| 文件 | 说明 |
|---|---|
| `cxgxabdntr_label_ISW.m` | **IWGNTR 主函数**，实现 IWGNTR 的核心算法 |
| `GMNTR_coil20.m` | COIL-20 数据集运行文件 |
| `GMNTR_coil100.m` | COIL-100 数据集运行文件 |

> 说明：`GMNTR_coil20` 和 `GMNTR_coil100` 为实验运行文件，分别用于在 COIL-20 与 COIL-100 数据上运行实验；IWGNTR 的主函数为 `cxgxabdntr_label_ISW.m`。

## 运行环境

- MATLAB
- 建议在仓库根目录下运行脚本，以确保函数路径正确
- 若使用其他目录，请将仓库根目录添加到 MATLAB 路径：
  ```matlab
  addpath(genpath(pwd));
