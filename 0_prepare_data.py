from AnomalyGeneration import *
from scipy import sparse
import pickle
import time
import os
import argparse

def preprocessDataset(dataset):
    print('Preprocess dataset: ' + dataset)
    t0 = time.time()
    if dataset in ['digg', 'uci']:
        edges = np.loadtxt(
            'data/raw/' +
            dataset,
            dtype=float,
            comments='%',
            delimiter=' ')
        edges = edges[:, 0:2].astype(dtype=int)
    elif dataset in ['btc_alpha', 'btc_otc']:
        if dataset == 'btc_alpha':
            file_name = 'data/raw/' + 'soc-sign-bitcoinalpha.csv'
        elif dataset =='btc_otc':
            file_name = 'data/raw/' + 'soc-sign-bitcoinotc.csv'
        with open(file_name) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 3].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)
    elif dataset in ['email','email-Eu']:
        if dataset == 'email':
            file_name = 'data/raw/' + 'email-dnc.edges'
        elif dataset =='email-Eu':
            file_name = 'data/raw/' + 'email-Eu-core.txt'
        with open(file_name,encoding='utf-8-sig') as f:
            lines = f.read().splitlines()
        edges = [[int(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        # edges = edges[edges[:, 2].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)
    # 使起始节点序号小于目标节点
    for ii in range(len(edges)):
        x0 = edges[ii][0]
        x1 = edges[ii][1]
        if x0 > x1:
            edges[ii][0] = x1
            edges[ii][1] = x0

    edges = edges[np.nonzero([x[0] != x[1] for x in edges])].tolist() # 去除自连接
    aa, idx = np.unique(edges, return_index=True, axis=0) # 去重复
    edges = np.array(edges)
    edges = edges[np.sort(idx)]

    vertexs, edges = np.unique(edges, return_inverse=True) # 获取节点和边信息
    edges = np.reshape(edges, [-1, 2])
    print('vertex:', len(vertexs), ' edge: ', len(edges))
    np.savetxt(
        'data/interim/' +
        dataset,
        X=edges,
        delimiter=' ',
        comments='%',
        fmt='%d')
    print('Preprocess finished! Time: %.2f s' % (time.time() - t0))


def generateDataset(dataset, snap_size, train_per=0.5, anomaly_per=0.01):
    print('Generating data with anomaly for Dataset: ', dataset)
    if not os.path.exists('data/interim/' + dataset):
        preprocessDataset(dataset)
    edges = np.loadtxt(
        'data/interim/' +
        dataset,
        dtype=float,
        comments='%',
        delimiter=' ')
    edges = edges[:, 0:2].astype(dtype=int)
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

    t0 = time.time()
    synthetic_test, train_mat, train ,test = anomaly_generation(train_per, anomaly_per, edges, n, m, seed=1)

    print("Anomaly Generation finish! Time: %.2f s"%(time.time()-t0))
    t0 = time.time()

    train_mat = (train_mat + train_mat.transpose() + sparse.eye(n)).tolil()
    # headtail类似于链表存储，每个节点对应一个列表，列表里面的就是和他相连的节点编号。
    # rows是起始节点 cols是目标节点  labs是测试集的标签 weis是权重（全为1）m是边数量 n是节点数量
    # synthetic_test 是一个(6988,3)的形状，其中前两维是边，第三维是标签，0为normal，1为anomaly
    # 训练和测试大小为6919 其中加上注入的1% 5% 10% 所以会变大是6988 6988多出来的69条是1%的数据。
    headtail = train_mat.rows
    del train_mat

    train_size = int(len(train) / snap_size + 0.5)
    test_size = int(len(synthetic_test) / snap_size + 0.5)
    print("Train size:%d  %d  Test size:%d %d" %
          (len(train), train_size, len(synthetic_test), test_size))
    rows = []
    cols = []
    weis = []
    labs = []
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size

        row = np.array(train[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(train[start_loc:end_loc, 1], dtype=np.int32)
        # if ii!=0:
        #     row = np.concatenate((row, rows[-1]))
        #     col = np.concatenate((col, rows[-1]))
        lab = np.zeros_like(row, dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)

        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)

    print("Training dataset contruction finish! Time: %.2f s" % (time.time()-t0))
    t0 = time.time()

    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size

        row = np.array(synthetic_test[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(synthetic_test[start_loc:end_loc, 1], dtype=np.int32)
        # row = np.concatenate((synthetic_test[start_loc:end_loc, 0], rows[-1]))
        # col = np.concatenate((synthetic_test[start_loc:end_loc, 1], rows[-1]))

        # lab = np.concatenate((np.array(synthetic_test[start_loc:end_loc, 2], dtype=np.int32), labs[-1]))
        lab = np.array(synthetic_test[start_loc:end_loc, 2])
        wei = np.ones_like(row, dtype=np.int32)

        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)

    print("Test dataset finish constructing! Time: %.2f s" % (time.time()-t0))
    # headtail类似于邻接链表存储，每个节点对应一个列表，列表里面的就是和他相连的节点编号。
    # rows是起始节点 cols是目标节点  labs是测试集的标签 weis是权重（全为1）m是边数量 n是节点数量
    # synthetic_test 是一个(6988,3)的形状，其中前两维是边，第三维是标签，0为normal，1为anomaly
    # 训练和测试大小为6919 其中加上注入的1% 5% 10% 所以会变大是6988 6988多出来的69条是1%的数据。
    with open('data/percent/' + dataset + '_' + str(train_per) + '_' + str(anomaly_per) + '.pkl', 'wb') as f:
        pickle.dump((rows,cols,labs,weis,headtail,train_size,test_size,n,m),f,pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc','email'], default='uci')
    parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
    parser.add_argument('--train_per', type=float, default=0.2)
    args = parser.parse_args()

    snap_size_dict = {'uci':1000, 'digg':6000, 'btc_alpha':1000, 'btc_otc':1500, 'email':450, 'email-Eu':1000}

    if args.anomaly_per is None:
        anomaly_pers = [0.01, 0.05, 0.10]
    else:
        anomaly_pers = [args.anomaly_per]

    for anomaly_per in anomaly_pers:
        generateDataset(args.dataset, snap_size_dict[args.dataset], train_per=args.train_per, anomaly_per=anomaly_per)