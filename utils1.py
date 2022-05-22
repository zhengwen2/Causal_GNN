#-*- coding:UTF-8 -*-
# load data,数据处理，指标计算
import dgl
import scipy.sparse
import torch
import sys
import copy
import numpy as np
import scipy.sparse as sp

def load_data(g_data='cora',device=0):
    # gpu = lambda x: x
    # if torch.cuda.is_available() and device >= 0:
    #     dev = torch.device('cuda:%d' % device)
    #     gpu = lambda x: x.to(dev)

    raw_dir = '../data/' + g_data
    graph = (
        dgl.data.CoraGraphDataset(raw_dir=raw_dir) if g_data == 'cora'
        else dgl.data.CiteseerGraphDataset(raw_dir=raw_dir) if g_data == 'citeseer'
        else dgl.data.PubmedGraphDataset(raw_dir=raw_dir) if g_data == 'pubmed'
        else dgl.data.CoraFullDataset(raw_dir=raw_dir) if g_data == 'corafull'
        else dgl.data.CoauthorCSDataset(raw_dir=raw_dir) if g_data == 'coauthor-cs'
        else dgl.data.CoauthorPhysicsDataset(raw_dir=raw_dir) if g_data == 'coauthor-phy'
        else dgl.data.RedditDataset(raw_dir=raw_dir) if g_data == 'reddit'
        else dgl.data.AmazonCoBuyComputerDataset(raw_dir=raw_dir)
        if g_data == 'amazon-com'
        else dgl.data.AmazonCoBuyPhotoDataset(raw_dir=raw_dir) if g_data == 'amazon-photo'
        else None
    )[0]
    X = graph.ndata['feat']
    Y =  graph.ndata['label']
    # n_nodes = node_features.shape[0]
    # nrange = torch.arange(n_nodes)
    # n_features = node_features.shape[1]
    # n_labels = int(Y.max().item() + 1)
    src, dst = graph.edges()
    # n_edges = src.shape[0]
    # is_bidir = ((dst == src[0]) & (src == dst[0])).any().item()
    # print('BiDirection: %s' % is_bidir)
    
    # degree = n_edges * (2 - is_bidir) / n_nodes
    # print('degree: %.2f' % degree)
    return X, Y, src, dst


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    if "torch" in str(type(mx)):
        rowsum = np.array(mx.sum(1))
    else:
        rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    if "torch" in str(type(mx)):
        mx = r_mat_inv.dot(mx)
    else:
        mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

def load_data_gcn(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def process_dataset(dataset='cora',device = 0):

    # gpu = lambda x: x
    # if torch.cuda.is_available() and device >= 0:
    #     dev = torch.device('cuda:%d' % device)
    #     gpu = lambda x: x.to(dev)

    X, Y, src, dst = load_data(dataset,device)

    X = normalize(X)
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(Y)

    adj = sp.coo_matrix((np.ones(src.shape), (src, dst)),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test
    # return gpu(adj), gpu(features), gpu(labels), gpu(idx_train), gpu(idx_val), gpu(idx_test)