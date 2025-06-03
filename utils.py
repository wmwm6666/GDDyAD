import argparse
import os
import pickle
import networkx as nx
import scipy.sparse as sp
import yaml
import torch
import numpy as np
import time
import random
import math
from numpy.linalg import inv
import pickle
import os
import dgl

def node_features_windows(args,x):
    out = torch.tensor(0)
    for i in range(args.window_size-1):
        if i == 0:
            out = torch.stack((x,x),dim=0)
        else:
            out = torch.cat((out,x.unsqueeze(0)),dim=0)

        return out

def get_windows_Adjs(args,snap,data):
    begin = snap - args.window_size + 1
    end = snap + 1
    Adj_begin = sum_Adj(data['A'][0:begin+1])

    return torch.stack([Adj_begin,Adj_begin+sum_Adj(data['A'][begin+1:end])],dim=0)

def sum_Adj(Adj_sparse_list):
    Adj = Adj_sparse_list[0]
    for idx,A in enumerate(Adj_sparse_list):
        if idx == 0:
            continue
        else:
            Adj = Adj + A

    return Adj

def get_windows_node_subgraph(args, time_Adjs):
    '''

    :param args:
    :param input: [t,n,n]
    :return:
    '''
    subgraph_Adj_list = []
    subgraph_nodes_list = []
    for Adj in time_Adjs:
        subgraph_list, subgraph_Adj, subgraph_nodes = get_node_subgraph(args, Adj)
        subgraph_Adj_list.append(subgraph_Adj.to_dense())
        subgraph_nodes_list.append(subgraph_nodes)
    return torch.stack(subgraph_Adj_list,dim=0), torch.stack(subgraph_nodes_list, dim=0)

def get_windows_Adj_norm(adj_matrices):
    t, m, n, _ = adj_matrices.size()

    # Initialize normalized adjacency matrices
    normalized_adjs = torch.zeros(t, m, n, n)

    for i in range(t):
        for j in range(m):
            # Get the adjacency matrix for the current time and graph
            adj_matrix = adj_matrices[i, j]
            Adj_matrix = normalize_adjacency_matrix(adj_matrix)
            normalized_adjs[i,j] = Adj_matrix

    return normalized_adjs

def gdc(A, alpha, eps):
    N = A.shape[0]

    # Self-loops
    # A_loop = torch.eye(N) + A

    # # Symmetric transition matrix
    # D_loop_vec = A_loop.sum(dim=0)
    # D_loop_vec_invsqrt = 1 / torch.sqrt(D_loop_vec)
    # D_loop_invsqrt = torch.diag(D_loop_vec_invsqrt)
    # T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    T_sym = normalize_adjacency_matrix(A)

    # PPR-based diffusion
    S = alpha * torch.inverse(torch.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S * (S >= eps).float()

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(dim=0)
    T_S = S_tilde / D_tilde_vec

    return T_S


def get_node_subgraph(args, Adj, num=1, k = 9):
    '''
    :param args:
    :param Adj: n*n
    :param num:
    :param k:
    :return: subgraph_list [n * dglgraph] ,subgraph_Adj [n , sn , sn],selected_nodes [n , k+1]
    '''
    k = args.neighbor_num
    S = gdc(Adj, 0.85, 1e-6)
    # 获取非零元素的坐标
    edge_indices = Adj.nonzero(as_tuple=False).t()
    graph = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=Adj.shape[0])

    # 计算关联性得分
    correlation_scores = S.clone()
    diag = torch.diag(correlation_scores)
    a_diag = torch.diag_embed(diag)
    correlation_scores = correlation_scores - a_diag

    # 选择 Top-k 节点
    topk_indices = torch.topk(correlation_scores, k, dim=1).indices
    # 构建子图
    selected_nodes = torch.cat((torch.tensor([range(Adj.shape[0])]).T, topk_indices), dim=1)

    subgraph_list = [dgl.node_subgraph(graph, selected_node) for selected_node in selected_nodes]

    subgraph_Adj = torch.stack([subgraph.adjacency_matrix() for subgraph in subgraph_list],dim=0)

    return subgraph_list,subgraph_Adj,selected_nodes


def get_edge_subgraph_list(args,g,i,j,num=2):
    s_list = []
    types = ['rand','DFS','BFS']
    for ty in types:
        for _ in range(num):
            subgraph = get_edge_subgraph(args,g,i,j,ty)
            s_list.append(subgraph)
    return s_list

def get_edge_subgraph(args,g,i,j,type='rand'):
    nodes = [i,j]
    walk_length = int(args.subgraph_node_num * 0.5 + 1)
    if type == 'rand':
        random_walks = dgl.sampling.node2vec_random_walk(g, nodes=nodes,p=100, q=1, walk_length=walk_length)
    elif type == 'DFS':
        random_walks = dgl.sampling.node2vec_random_walk(g, nodes=nodes,p=100, q=0.1, walk_length=walk_length)
    elif type == 'BFS':
        random_walks = dgl.sampling.node2vec_random_walk(g, nodes=nodes,p=100, q=5, walk_length=walk_length)
    walks = []
    index = 0
    index_2 = 0
    while len(walks) < args.subgraph_node_num * 0.5:
        # if random_walks[0][index]==j or random_walks[0][index] in walks:
        #     index = index + 1
        #     continue
        walks.append(random_walks[0][index])
        index = index + 1
    while args.subgraph_node_num * 0.5 <= len(walks) < args.subgraph_node_num:
        # if random_walks[1][index_2] in walks:
        #     index_2 = index_2 + 1
        #     continue
        walks.append(random_walks[1][index_2])
        index_2 = index_2 + 1
    subgraph = dgl.node_subgraph(g, walks)
    subgraph = dgl.to_bidirected(subgraph)
    subgraph = dgl.remove_self_loop(subgraph)
    subgraph = dgl.add_self_loop(subgraph)
    return subgraph




def load_data(args):
    """Load dynamic network dataset"""
    print('Loading {} dataset...'.format(args.dataset))
    with open('data/percent/' + args.dataset + '_' + str(args.train_per) + '_' + str(args.anomaly_per) + '.pkl',
              'rb') as f:
        rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges = pickle.load(f)
    # 起始节点，终止节点，标签，   权重，  邻接链表 ， 训练大小     ，测试大小   ，节点数量 ，边数量
    # 获取每个节点的度数
    degrees = np.array([len(x) for x in headtail])
    num_snap = test_size + train_size

    # num_snap个时刻的边
    edges = [np.vstack((rows[i], cols[i])).T for i in range(num_snap)]

    # 获取num_snap个时刻的归一化矩阵,邻接矩阵和特征矩阵 adjs，Adjs,eigen_adjs：list：15
    adjs, Adjs, eigen_adjs = get_adjs(args,rows, cols, weights, nb_nodes)

    # X = torch.from_numpy(eigen_adjs[0][:,0:args.in_dim]).float()
    X = torch.rand((nb_nodes,args.in_dim),dtype=torch.float)

    labels = [torch.FloatTensor(label) for label in labels]

    snap_train = list(range(num_snap))[:train_size]
    snap_test = list(range(num_snap))[train_size:]

    idx = list(range(nb_nodes))
    index_id_map = {i: i for i in idx}
    idx = np.array(idx)

    return {'X': X, 'A': adjs, 'S': eigen_adjs, 'index_id_map': index_id_map, 'edges': edges,
            'y': labels, 'idx': idx, 'snap_train': snap_train, 'degrees': degrees,'Adjs':Adjs,
            'snap_test': snap_test, 'num_snap': num_snap}

def get_adjs(args, rows, cols, weights, nb_nodes):

    eigen_file_name = 'data/eigen/' + args.dataset + '_' + str(args.train_per) + '_' + str(args.anomaly_per) + '.pkl'
    if not os.path.exists(eigen_file_name):
        generate_eigen = True
        print('Generating eigen as: ' + eigen_file_name)
    else:
        generate_eigen = False
        print('Loading eigen from: ' + eigen_file_name)
        with open(eigen_file_name, 'rb') as f:
            # eigen_adjs_sparse是一个列表，记录的是数据集15个snap时刻的稀疏矩阵形式。
            eigen_adjs_sparse = pickle.load(f)
        eigen_adjs = []
        for eigen_adj_sparse in eigen_adjs_sparse:
            eigen_adjs.append(np.array(eigen_adj_sparse.todense()))
            # eigen_adjs.append(np.array(eigen_adj_sparse.todense()))

    adjs = []
    Adjs =[]
    if generate_eigen:
        eigen_adjs = []
        eigen_adjs_sparse = []

    for i in range(len(rows)):
        adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(nb_nodes, nb_nodes), dtype=np.float32)
        Adjs.append(preprocess_Adj(adj))
        adjs.append(preprocess_adj(adj))
        if True:
            if generate_eigen:
                eigen_adj = 0.15 * inv((sp.eye(adj.shape[0]) - (1 - 0.15) * adj_normalize(adj)).toarray())
                for p in range(adj.shape[0]):
                    eigen_adj[p, p] = 0.
                eigen_adj = normalize(eigen_adj)
                eigen_adjs.append(eigen_adj)
                eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

        else:
            eigen_adjs.append(None)

    if generate_eigen:
        with open(eigen_file_name, 'wb') as f:
            pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

    return adjs, Adjs, eigen_adjs

def normalize_adjacency_matrix(adj_matrix):
    # Step 1: Add self-weights to the adjacency matrix
    adj_hat = adj_matrix + torch.eye(adj_matrix.size(0))

    # Step 2: Compute the degree matrix
    degree_matrix = torch.sum(adj_hat, dim=1)
    d_inv_sqrt = torch.pow(degree_matrix, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # Handle division by zero
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.0
    # Step 3: Symmetrically normalize the adjacency matrix
    normalized_adj = torch.diag(d_inv_sqrt) @ adj_hat @ torch.diag(d_inv_sqrt)

    return normalized_adj

def normalize_tirble_multiple_adjacency_matrix(adj_matrix):
    adj_list = []
    for adj in adj_matrix:
        adj_list.append(normalize_adjacency_matrix(adj))
    return torch.stack(adj_list,dim=0)


def normalize_multiple_adjacency_matrix(adj_matrix):
    # Step 1: Add self-weights to the adjacency matrix
    adj_hat = adj_matrix + torch.eye(adj_matrix.size(-1))

    # Step 2: Compute the degree matrix
    degree_matrix = torch.sum(adj_hat, dim=-1)
    d_inv_sqrt = torch.pow(degree_matrix, -0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.0  # Handle division by zero
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # Step 3: Symmetrically normalize the adjacency matrix
    trans = torch.ones((1,d_inv_sqrt.size()[-1]))
    d_inv_sqrt = d_inv_sqrt.unsqueeze(dim = -1) @ trans
    indices = torch.arange(d_inv_sqrt.size()[-1])
    zero_tensor = torch.zeros_like(d_inv_sqrt)
    zero_tensor[:, :, indices, indices] = d_inv_sqrt[:, :, indices, indices]

    normalized_adj =  zero_tensor @ adj_hat @ zero_tensor

    return normalized_adj
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix. (0226)"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation. (0226)"""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
    adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj_np = np.array(adj.todense())
    adj_normalized = normalize_adj(adj_ + sp.eye(adj.shape[0]))
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return adj_normalized

def preprocess_Adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
    adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj_np = np.array(adj.todense())
    # adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_)
    return adj_normalized

def negative_sampling(data, edges):
    negative_edges = []
    node_list = data['idx']
    num_node = node_list.shape[0]
    for snap_edge in edges:
        num_edge = snap_edge.shape[0]

        negative_edge = snap_edge.copy()
        fake_idx = np.random.choice(num_node, num_edge)
        fake_position = np.random.choice(2, num_edge).tolist()
        fake_idx = node_list[fake_idx]
        negative_edge[np.arange(num_edge), fake_position] = fake_idx

        negative_edges.append(torch.Tensor(negative_edge))
    return negative_edges

def negative_sampling2(data, edges):
    negative_edges = []
    node_list = data['idx']
    num_node = node_list.shape[0]
    all_edge = None
    for idx,snap_edge in enumerate(edges):
        if idx==0:
            all_edge = snap_edge
        else:
            all_edge = np.concatenate((all_edge,snap_edge),axis=0)
        negative_edge = negative_sampling_snap(all_edge, num_node,num_samples=snap_edge.shape[0])

        negative_edges.append(torch.Tensor(negative_edge))
    return negative_edges


def negative_sampling_snap(edges, num_node,num_samples):
    negative_edges = []
    while len(negative_edges) < num_samples:
        # 随机采样两个节点
        node1 = np.random.choice(num_node,1)
        node2 = np.random.choice(num_node,1)

        # 确保采样的节点不在正样本边中
        if [node1, node2] not in edges and [node2, node1] not in edges and node1 != node2:
            negative_edges.append([node1, node2])

    return np.stack(negative_edges).squeeze()

def general_Adj(edges,snap):
    pass


def compute_zero_WL(node_list, link_list):
    WL_dict = {}
    for i in node_list:
        WL_dict[i] = 0
    return WL_dict

# batching + hop + int + time
def compute_batch_hop(node_list, edges_all, num_snap, Ss, k=5, window_size=1):
    # 这个方法输出一个list，目的是输出每个时间窗口每条边和其他节点的跳数。、
    '''
    :param node_list: 节点列表
    :param edges_all: 所有边
    :param num_snap: 时间片个数
    :param Ss: 所有特征矩阵
    :param k: 取的邻居个数
    :param window_size: 时间窗口大小
    :return: 每个时间窗口每条边和其他节点的跳数。
    '''
    batch_hop_dicts = [None] * (window_size-1)
    s_ranking = [0] + list(range(k+1))

    Gs = []
    for snap in range(num_snap):
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edges_all[snap])  # 这里做了改动G.add_edges_from(edges_all[snap])  # 这里做了改动
        Gs.append(G)

    # 这个从window_size - 1开始是因为要往前看window_size - 1个时间步
    for snap in range(window_size - 1, num_snap):
        batch_hop_dict = {}
        # S = Ss[snap]
        edges = edges_all[snap]

        # G = nx.Graph()
        # G.add_nodes_from(node_list)
        # G.add_edges_from(edges)

        for edge in edges:
            edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])
            batch_hop_dict[edge_idx] = []
            for lookback in range(window_size):
                # s = np.array(Ss[snap-lookback][edge[0]] + Ss[snap-lookback][edge[1]].todense()).squeeze()
                # s形状是（1899，）是此条边两个节点的特征向量的和 比如此条边为【375 376】
                s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]
                s[edge[0]] = -1000 # don't pick myself
                s[edge[1]] = -1000 # don't pick myself
                # 这个是选择特征值最大的k条边。【1898 638 626 627 628】
                top_k_neighbor_index = s.argsort()[-k:][::-1]

                # 选择完最大的k条边之后把原来的两个节点编号加入，indexs为(k+2,)  【375 376 1898 638 626 627 628】
                indexs = np.hstack((np.array([edge[0], edge[1]]), top_k_neighbor_index))

                for i, neighbor_index in enumerate(indexs):
                    # 第一个try是求起始节点和k个邻居节点（包括自己）的最短路径距离，如果没有路径则距离为99 ，第二个try是目标节点于其他节点的距离
                    try:
                        hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                    except:
                        hop1 = 99
                    try:
                        hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                    except:
                        hop2 = 99
                    # 获取邻居节点于目标边的两个节点的最小距离
                    hop = min(hop1, hop2)
                    # batch_hop_dict存储当前边的信息，key是边的编号'1_375_376'，value是一个列表，大小为（window_size*k+2，4），记录这条边对每个节点的编号，排名，距离，时间
                    batch_hop_dict[edge_idx].append((neighbor_index, s_ranking[i], hop, lookback))
        batch_hop_dicts.append(batch_hop_dict)

    return batch_hop_dicts

# Dict to embeddings
def dicts_to_embeddings(feats, batch_hop_dicts, wl_dict, num_snap, use_raw_feat=False):

    raw_embeddings = []
    wl_embeddings = []
    hop_embeddings = []
    int_embeddings = []
    time_embeddings = []

    for snap in range(num_snap):

        batch_hop_dict = batch_hop_dicts[snap]

        if batch_hop_dict is None:
            raw_embeddings.append(None)
            wl_embeddings.append(None)
            hop_embeddings.append(None)
            int_embeddings.append(None)
            time_embeddings.append(None)
            continue

        raw_features_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        time_ids_list = []

        for edge_idx in batch_hop_dict:

            neighbors_list = batch_hop_dict[edge_idx]
            edge = edge_idx.split('_')[1:]
            edge[0], edge[1] = int(edge[0]), int(edge[1])

            raw_features = []
            role_ids = []
            position_ids = []
            hop_ids = []
            time_ids = []

            for neighbor, intimacy_rank, hop, time in neighbors_list:
                if use_raw_feat:
                    raw_features.append(feats[snap-time][neighbor])
                else:
                    raw_features.append(None)
                role_ids.append(wl_dict[neighbor])
                hop_ids.append(hop)
                position_ids.append(intimacy_rank)
                time_ids.append(time)

            raw_features_list.append(raw_features)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)
            time_ids_list.append(time_ids)

        if use_raw_feat:
            raw_embedding = torch.FloatTensor(raw_features_list)
        else:
            raw_embedding = None
        wl_embedding = torch.LongTensor(role_ids_list)
        hop_embedding = torch.LongTensor(hop_ids_list)
        int_embedding = torch.LongTensor(position_ids_list)
        time_embedding = torch.LongTensor(time_ids_list)

        raw_embeddings.append(raw_embedding)
        wl_embeddings.append(wl_embedding)
        hop_embeddings.append(hop_embedding)
        int_embeddings.append(int_embedding)
        time_embeddings.append(time_embedding)

    return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings