from cgi import test
from pyexpat import features
import dgl.data
import torch
import networkx as nx 
import scipy.sparse as sp
import numpy as np

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

def load_data(name):
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    # feature normalize
    feature = g.ndata['feat']
    label = g.ndata['label']
    # idx_train = g.ndata['train_mask']
    # idx_val = g.ndata['val_mask']
    # idx_test = g.ndata['test_mask']

    # get edges feature
    edges = g.edges()
    g = dgl.graph(edges)
    g = dgl.to_simple(g)
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g)
    g = g.to_networkx()

    adj = nx.adjacency_matrix(g)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    train_mask = np.zeros((adj.size()[0]), dtype=bool)
    val_mask = np.zeros((adj.size()[0]), dtype=bool)
    test_mask = np.zeros((adj.size()[0]), dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    

    idx_train = torch.from_numpy(train_mask)
    idx_val = torch.from_numpy(val_mask)
    idx_test = torch.from_numpy(test_mask)


    return adj, feature, label, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data_from_dgl(g_data='cora'):
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

    features = graph.ndata['feat']
    labels = graph.ndata['label']

    src, dst = graph.edges()

    features = normalize_features(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    adj = sp.coo_matrix((np.ones(src.shape), (src, dst)),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    # idx_train = range(500)
    # idx_val = range(500, 1000)
    # idx_test = range(1000, 1500)
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot