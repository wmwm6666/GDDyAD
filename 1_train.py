import numpy as np
import torch
import argparse
import dgl
import dgl.function as fn
from matplotlib import pyplot as plt
from torch import optim
from sklearn.metrics import roc_curve, roc_auc_score, auc
from models import *
from utils import *
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import tqdm

def train(args):
    torch.cuda.empty_cache()  # 释放显存
    data = load_data(args)
    my_model = MyModel(args,data,use_atten=args.use_atten)
    optimizer = optim.Adam(my_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #
    criterion = nn.BCEWithLogitsLoss()
    # for t in range(0,data['snap_train']):
    #     data['']
    # raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = generate_embedding(args, data, data['edges'])
    negative_data = negative_sampling2(data, data['edges'][0:max(data['snap_train']) + 1])
    # Adjs = []
    nodes = data['idx'].shape[0]
    # Adj = torch.zeros((nodes, nodes), dtype=torch.float)
    for i in range(max(data['snap_train']) + 1):
        data['y'][i] = torch.cat((data['y'][i], torch.ones((negative_data[i].shape[0]))), dim=0)
        # Adj = Adj + data['Adjs'][i]
        # Adjs.append(Adj)
    # for i in range(max(data['snap_train']) + 1,data['num_snap']):
    #     Adj = Adj + data['Adjs'][i]
        # Adjs.append(Adj)
    # A = torch.stack(Adjs, dim=0)
    if args.use_dif:
        file_path = 'data/pre/' + args.dataset + '_' + str(args.train_per) + '_' + str(args.anomaly_per) + '.pt'
        # file_path = 'data/pre/' + args.dataset + '_' + str(args.train_per) + '_' + str(args.anomaly_per) + str(args.neighbor_num) +'.pt'
        # file_path = 'data/pre/' + args.dataset + '_' + str(args.train_per) + '_' + str(args.anomaly_per) + '_'+str( args.neighbor_num) +'_'+str( args.window_size) +'.pt'
        if os.path.exists(file_path):
            # 如果文件存在，加载数据
            graph_data = torch.load(file_path)
            print("Loaded data from file.")
            subgraph_adj_list = graph_data['subgraph_adj_list']
            subgraph_nodes_list = graph_data['subgraph_nodes_list']
            test_subgraph_adj_list = graph_data['test_subgraph_adj_list']
            test_subgraph_nodes_list = graph_data['test_subgraph_nodes_list']
        else:
            # 创建一个字典
            print('生成文件')
            subgraph_adj_list = []
            subgraph_nodes_list = []
            test_subgraph_adj_list = []
            test_subgraph_nodes_list = []

            for snap in tqdm.tqdm(range(args.window_size - 1, max(data['snap_train']) + 1)):
                # input = A[snap - args.window_size + 1:snap + 1, :, :]  # [ws,nodes,nodes]->[ws,n,c,sn,sn]
                # input = torch.stack(Adjs[snap - args.window_size + 1:snap + 1],dim=0)
                input = get_windows_Adjs(args, snap, data).to_dense()
                subgraph_Adj, subgraph_nodes = get_windows_node_subgraph(args, input)
                subgraph_adj = normalize_multiple_adjacency_matrix(subgraph_Adj)
                subgraph_adj_list.append(subgraph_adj)
                subgraph_nodes_list.append(subgraph_nodes)

            for snap in tqdm.tqdm(data['snap_test']):
                # input = A[snap - args.window_size + 1:snap + 1, :, :]  # [ws,nodes,nodes]->[ws,n,c,sn,sn]
                # input = torch.stack(Adjs[snap - args.window_size + 1:snap + 1], dim=0)
                input = get_windows_Adjs(args, snap, data).to_dense()
                subgraph_Adj, subgraph_nodes = get_windows_node_subgraph(args, input)
                subgraph_adj = normalize_multiple_adjacency_matrix(subgraph_Adj)
                test_subgraph_adj_list.append(subgraph_adj)
                test_subgraph_nodes_list.append(subgraph_nodes)

            data_dict = {'subgraph_adj_list': subgraph_adj_list, 'subgraph_nodes_list': subgraph_nodes_list,
                         'test_subgraph_adj_list': test_subgraph_adj_list,
                         'test_subgraph_nodes_list': test_subgraph_nodes_list}
            # 如果要测试参数敏感度，那就注释这一行以免占用太多硬盘
            torch.save(data_dict, file_path)
    else:
        pass
    print('$$$$ Start $$$$')
    max_AUC = 0
    max_epoch = None
    test_epoch_AUC = []
    test_AUC_dict = {}
    my_model.to(args.device)
    for epoch in range(args.max_epoch):
        train_loss = []
        # train_AUC_list = []
        for snap in range(args.window_size - 1, max(data['snap_train']) + 1):
            my_model.train()
            # input = A[snap - args.window_size + 1:snap + 1, :, :]  # [ws,nodes,nodes]->[ws,n,c,sn,sn]
            # subgraph_Adj, subgraph_nodes = get_windows_node_sungraph(args, input)

            # 这一块消融实验得改
            if args.use_dif:
                # print("this snap is ", snap)
                subgraph_nodes = subgraph_nodes_list[snap - args.window_size + 1]
                x = data['X'][subgraph_nodes].to(args.device)
                subgraph_adj = subgraph_adj_list[snap - args.window_size + 1].to(args.device)
                out = my_model(x, subgraph_adj, snap, negative_data)
            else:
                x = data['X'].to(args.device)
                x = node_features_windows(args,x).to(args.device)
                adj = get_windows_Adjs(args, snap, data).to_dense()
                adj = normalize_tirble_multiple_adjacency_matrix(adj).to(args.device)
                out = my_model(x, adj, snap, negative_data)

            out = out.float()
            optimizer.zero_grad()
            y = data['y'][snap].cuda()
            loss = criterion(out, y)
            # paraloss = torch.stack([torch.norm(params, 2).pow(2) for params in my_model.parameters()])
            # paraloss = paraloss.sum()
            # loss = loss+paraloss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print("epoch", epoch + 1, "loss: {:.5f}".format(np.array(train_loss).sum() / len(train_loss)))


        if (epoch + 1) % 10 == 0:
            my_model.eval()
            test_loss = []
            auc_list = []
            for snap in data['snap_test']:

                # 消融分支
                if args.use_dif:
                    subgraph_nodes = test_subgraph_nodes_list[snap - min(data['snap_test'])]
                    x = data['X'][subgraph_nodes].to(args.device)
                    subgraph_adj = test_subgraph_adj_list[snap - min(data['snap_test'])].to(args.device)
                    out = my_model(x, subgraph_adj, snap, None)
                else:
                    x = data['X'].to(args.device)
                    x = node_features_windows(args,x).to(args.device)
                    adj = get_windows_Adjs(args, snap, data).to_dense()
                    adj = normalize_tirble_multiple_adjacency_matrix(adj).to(args.device)
                    out = my_model(x, adj, snap, None)

                out = out.float()
                y = data['y'][snap].cuda().float()
                loss = criterion(out, y)
                test_loss.append(loss.cpu().item())
                # 计算AUC和绘制ROC曲线
                fpr, tpr, thresholds = roc_curve(y.cpu().detach().numpy(), out.cpu().detach().numpy())
                roc_auc = auc(fpr, tpr)
                if roc_auc is None:
                    auc_list.append(1)
                else:
                    auc_list.append(roc_auc)
                # 输出AUC分数
                print('snap:', snap, "loss:", loss.item(), "AUC:", roc_auc)
            this_AUC = np.array(auc_list).sum() / len(auc_list)
            if this_AUC > max_AUC:
                early_stop = 0
                max_AUC = this_AUC
                max_epoch = epoch
                test_AUC_dict[this_AUC] = auc_list
            else:
                early_stop = early_stop + 1
                if early_stop == args.early_stop:
                    break
            print("epoch", epoch + 1, "test_mean_loss:", np.array(test_loss).sum() / len(test_loss), "AUC:", this_AUC)

            test_epoch_AUC.append(this_AUC)
    f = open('log.txt', mode='a+', encoding='utf-8')
    f.writelines(
        args.dataset + '  ' + str(args.anomaly_per) + '  ' + str(args.train_per) + '  ' + str(args.window_size) + '  ' +
        str(args.neighbor_num) + '  '+ str(args.in_dim)+ '  '+ str(args.hidden_dim) + '  '+ str(args.out_dim)+'  '
        +  str(args.gru_dim) + '  ' + str(args.num_attention_heads) + '  ' +
        str(args.attention_hidden_size) + '  ' + str(args.score_size) + '  ' + str(args.dropout) + '  ' +
        str(max(test_epoch_AUC)) + '  ' + str(test_AUC_dict[max(test_epoch_AUC)]) + '  ' + str(max_epoch + 1) + '  '+ "\n")
    f.close()
    print("最大的AUC: ", max(test_epoch_AUC), '  epoch:', max_epoch + 1, '   AUC_list:',
          test_AUC_dict[max(test_epoch_AUC)])
    print('$$$$ Finish $$$$')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc', 'email'], default='uci')
    parser.add_argument('--anomaly_per', choices=[0.01, 0.05, 0.1], type=float, default=0.1)
    parser.add_argument('--train_per', type=float, default=0.5)
    parser.add_argument('--use_dif', type=bool, default=True)
    parser.add_argument('--use_atten', type=bool, default=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--res', type=bool, default=False)

    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--neighbor_num', type=int, default=19)

    parser.add_argument('--in_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--out_dim', type=int, default=96)
    parser.add_argument('--gru_dim', type=int, default=128)

    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--attention_hidden_size', type=int, default=96)

    parser.add_argument('--score_size', type=int, default=96)

    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0019)
    parser.add_argument('--weight_decay', type=float, default=7e-5)  #3e-4
    parser.add_argument('--dropout', type=float, default=0.08)
    parser.add_argument('--early_stop', type=int, default=8)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    # torch.set_printoptions(profile="full")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #train(args)
    # window_size_list = [2,3,5]
    # neighbor_num_list = [2,3,5,7,9,11,13,15,17,19]
    anomaly_per_list = [0.1, 0.05, 0.01]
    for anomaly_per in anomaly_per_list:
    # for window_size in window_size_list:
    #     for neighbor_num in neighbor_num_list:
    #         args.anomaly_per = 0.1
    #         args.window_size = window_size
    #         args.neighbor_num = neighbor_num
        args.anomaly_per = anomaly_per
        train(args)
    # train_per_list = [0.2,0.3,0.4,0.5,0.6]
    # for train_per in train_per_list:
    #     args.anomaly_per = 0.1
    #     args.train_per = train_per
    # train(args)



    # hidden_dim_list = [32,48,64,96]
    # out_dim_list = [32, 48, 64, 96, 128]
    # gru_dim_list = [32, 48, 64, 96, 128]
    # anomaly_per_list = [0.01,0.05, 0.1]
    # score_size_list = [32, 48, 64, 96, 128]
    # for hidden_dim in hidden_dim_list:
    #     for gru_dim in gru_dim_list:
    #         for out_dim in out_dim_list:
    #             for score_size in score_size_list:
    #                 for anomaly_per in anomaly_per_list:
    #                     args.anomaly_per = anomaly_per
    #                     args.gru_dim = gru_dim
    #                     args.hidden_dim = hidden_dim
    #                     args.out_dim = out_dim
    #                     args.attention_hidden_size = out_dim
    #                     args.score_size = score_size
    #                     train(args)






    # raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = generate_embedding(args, data,
    #                                                                                                     data['edges'])
    # 创建一个示例图
    # g = dgl.graph((data['edges'][0].transpose()[0],data['edges'][0].transpose()[1]),num_nodes=1899)
    # nodes = data['idx'].shape[0]
    # Adj = torch.zeros((nodes, nodes), dtype=torch.float)
    # for snap in range(args.window_size - 1, max(data['snap_train']) + 1):
    #     Adj = Adj + data['Adjs'][snap]
    #     subgraph_list, subgraph_Adj, subgraph_nodes = get_node_subgraph(args, Adj)
    #     subgraph_adj = torch.stack([normalize_adjacency_matrix(A.to_dense()) for A in subgraph_Adj], dim=0)
    #     x = data['X'][subgraph_nodes]
    #     gcn = GCN_Model(args)
    #     out = gcn(x, subgraph_adj)
    #     print(out.shape)
    #     break
    #
    # ne = negative_sampling2(data, data['edges'][0:max(data['snap_train']) + 1])
    # g = dgl.graph((data['edges'][0].T[0],data['edges'][0].T[1]), num_nodes=1899)
    # g = dgl.to_bidirected(g)
    # g.ndata['feat'] = torch.ones((g.num_nodes(),args.in_dim))
    #
    # edges = data['edges'][0]
    # batch_subg_list = [get_edge_subgraph_list(args,g,edge[0],edge[1]) for edge in edges] # [1000,6,DGL] list
    # # edge = edges[458]
    # # subg_list = get_edge_subgraph_list(args,g,edge[0],edge[1])
    # # print(subg_list[0])
    # # print(subg_list[0].edges())
    #
    # model = GCN_One_Subgraph(args)
    # model = model.to(args.device)
    # out = model(gs)
    # print(out.shape)
    print('end')
    # train(args)
    # 设置随机游走的参数

    '''
    num_walks = 10
    walk_length = 10

    # 执行随机游走
    random_walks = dgl.sampling.random_walk(g, nodes=[1,2],length=walk_length)

    # 随机游走结果是一个DGLGraph对象的列表，每个元素代表一个游走序列
    print(f"随机游走 :", random_walks[0][0].unique())

    subgraph = dgl.node_subgraph(g,random_walks[0][0].unique())

    print(subgraph)
    print(subgraph.nodes())
    print(subgraph.edges())
    '''
