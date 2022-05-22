from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layer import GraphConvolution, GraphAttentionLayer

class Encoder(nn.Module):
    def __init__(self, nhid, nclass, nz) -> None:
        super().__init__()
        self.linear1 = nn.Linear(nhid + nclass, nz)
        self.linear2 = nn.Linear(nhid + nclass, nz)

    
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        mu = F.relu(self.linear1(x))
        logvar = F.relu(self.linear2(x))
        # variance = mu.mean(dim=1)
        # print(logvar.mean(dim=1).sum())
        # print(variance.sum())

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, ns, nhid) -> None:
        super().__init__()
        self.linear_x = nn.Linear(ns, nhid)

    
    def forward(self, s):
        x = F.sigmoid(self.linear_x(s))
        return x


class CausalGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nz, ns,dropout, alpha, base_model='gcn', nheads=8) -> None:
        super().__init__()
        self.dropout = dropout
        # concat: whether input elu layer
        # encoder 
        self.attention_z = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.encoder = Encoder(nhid, nclass, nz)

        # z->s
        self.attention_s = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.linear_s = nn.Linear(nhid + nz, ns)

        # decoder
        self.decoder = Decoder(ns, nfeat)

        # predict
        if base_model == 'gcn':
            self.base_model = GCN(nfeat, ns, nhid, nclass, dropout)
        else:
            self.base_model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads, ns)
        
        self.nz = nz





    def forward(self, feat, adj, y, index, stage=None, z_mean=None):
        # x = Wx
        h_z = self.attention_z(feat, adj)
        h_s = self.attention_s(feat, adj)
        # x = F.dropout(h_z, self.dropout, training=self.training)
        x = h_z[index, :]
        mu = [] 
        logvar = []
        z_sum = 0
        if stage == 'training':
        # encoder 
            y = y[index, :]
            mu, logvar = self.encoder(x, y)
            z = self.reparametrize(mu, logvar)
            z_sum = z.var(dim=1).sum()
            z_mean = torch.mean(z, dim=0).unsqueeze(0)
        else: 
            z = z_mean.repeat(x.size()[0], 1)

        # z->s
        x = h_s[index, :]
        x = torch.cat((z, x), dim=1)
        s = F.relu(self.linear_s(x))
        # s = F.dropout(s, self.dropout, training=self.training)

        # decoder
        recon_x = self.decoder(s)

        # predict
        output = self.base_model(s, feat, adj, index)
        return z_mean, output, recon_x, mu, logvar, z_sum
    
    def reparametrize(self, mu, logvar):
        # # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        # std = 0.5 * torch.exp(logvar)
        # # N(mu, std^2) = N(0, 1) * std + mu
        # z = torch.randn(std.size()) * std + mu
    
        # eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(logvar/2)
        return z



class GCN(nn.Module):
    def __init__(self, nfeat, ns, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.linear = nn.Linear(nhid+ns, nclass)
        self.dropout = dropout

    def forward(self, s, x, adj, index):
        # tranditional gcn
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        x = self.gc2(x, adj)


        # add s
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat((x[index, :], s), dim=1)
        # x = F.relu(self.linear(x))
        x = self.linear(x)

        return F.log_softmax(x, dim=1)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, ns):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False) 

        self.linear = nn.Linear(nhid * nheads + ns, nclass)

    def forward(self, s, x, adj, index):
        # tranditional GAT
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        # # add s
        x = torch.cat((x[index, :], s), dim=1)
        x = F.elu(self.linear(x))
        # x = x[index, :]

        return F.log_softmax(x, dim=1)

