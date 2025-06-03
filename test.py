import dgl
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch
from utils import *


def get_second_order_neighbors(A):
    # 归一化邻接矩阵
    def normalize_adjacency_matrix(A):
        D_hat = torch.diag(torch.pow(A.sum(dim=1) + 1, -0.5))
        A_hat = A + torch.eye(A.size()[0])
        return D_hat @ A_hat @ D_hat

    # 归一化的邻接矩阵
    T_sym = normalize_adjacency_matrix(A)

    # 二阶邻居的影响力矩阵
    second_order_influence = torch.matmul(T_sym, T_sym)

    # 二阶邻居数量
    second_order_counts = (second_order_influence > 0).sum(dim=1) - 1  # subtract self influence

    return second_order_influence, second_order_counts


def aggregate_features_with_second_order(A, X, k):
    second_order_influence, second_order_counts = get_second_order_neighbors(A)
    N = A.shape[0]
    aggregated_features = torch.zeros_like(X)

    for i in range(N):
        # 获取节点i的二阶邻居及其影响力
        influences = second_order_influence[i]

        # 排除自身影响力
        influences[i] = 0

        # 获取二阶邻居的索引及其影响力
        neighbors = influences.nonzero(as_tuple=False).squeeze()
        neighbor_influences = influences[neighbors]

        # 根据影响力排序并选取前k个
        if len(neighbors) >= k:
            top_k_indices = torch.topk(neighbor_influences, k).indices
            top_k_neighbors = neighbors[top_k_indices]
            top_k_influences = neighbor_influences[top_k_indices]
        else:
            top_k_neighbors = neighbors
            top_k_influences = neighbor_influences

        # 计算加权聚合特征
        if len(top_k_neighbors) > 0:
            weights = top_k_influences / top_k_influences.sum()
            aggregated_features[i] = torch.sum(weights.unsqueeze(1) * X[top_k_neighbors], dim=0)

    return aggregated_features


# 示例邻接矩阵
A = torch.tensor([[0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1],
                  [1, 0, 0, 1, 0]], dtype=torch.float32)

# 示例节点特征矩阵
X = torch.tensor([[1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0],
                  [1, 0]], dtype=torch.float32)

k = 2
aggregated_features = aggregate_features_with_second_order(A, X, k)
print("Aggregated features based on second-order influence:")
print(aggregated_features)
