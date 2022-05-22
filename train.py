import argparse
from lib2to3.pytree import Base
import torch
import numpy as np
import random
import yaml
import gc
import time
from utils import load_data, load_data_from_dgl, accuracy
from utils1 import process_dataset, accuracy
from model import CausalGNN
import torch.optim as optim
import torch.nn.functional as F

# output to a file

# set seed
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# input params
parser = argparse.ArgumentParser()
with open('param.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)['GCN']
    for key in config.keys():
        name = '--' + key
        parser.add_argument(name, type=type(config[key]), default=config[key])
args = parser.parse_args()

# use cuda
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set seed
set_rng_seed(args.seed)

# load data
adj, feature, label, idx_train, idx_val, idx_test = load_data_from_dgl('cora')

tmp = label.unsqueeze(1)
num = label.size()[0]
class_num = torch.max(label) + 1
label_train = torch.zeros(num, class_num).scatter_(1, tmp, 1)

adj = F.normalize(adj, p=1, dim=1)

# model
model =CausalGNN(nfeat=feature.shape[1],
                nhid=args.hidden,
                nclass=int(label.max())+1,
                dropout=args.dropout,
                nz=args.nz,
                ns=args.ns,
                alpha=args.alpha).to(device)

optimizer = optim.Adam(model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay)

BCE_loss = torch.nn.BCELoss(reduction='mean')

if args.cuda:
    adj = adj.to(device)
    feature = feature.to(device)
    label = label.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    label_train = label_train.to(device)



def loss_functiona(output, logvar, mu, feature, recon_x, z_sum):
    loss1 = F.nll_loss(output, label[idx_train])

    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    
    # x = feature.long().reshape(1, -1).squeeze(0)
    # recon_x = recon_x.reshape(1, -1).squeeze(0)

    reconstruction_loss = BCE_loss(recon_x, feature)

    loss = loss1 + KL_divergence * args.KL_loss + reconstruction_loss * args.BCE_loss + args.z_loss * z_sum
    
    return loss


def train_encoder(epoch, model):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj, label_train, idx_train, stage='training')
    loss_train = loss_functiona(output, logvar, mu, feature[idx_train], recon_x, z_sum)
    # loss_train = F.nll_loss(output, label[idx_train])
    acc_train = accuracy(output, label[idx_train])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return z_mean


def train_base_model(epoch, z_mean):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj, label_train, idx_train, stage='train_base_model', z_mean=z_mean)
    loss_train = F.nll_loss(output, label[idx_train])
    acc_train = accuracy(output, label[idx_train])
    loss_train.backward(retain_graph=True)
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, z_mean):
    model.eval()
    z, output, recon_x, mu, logvar, z_sum = model(feature, adj, label_train, idx_val, z_mean=z_mean)
    loss_val = F.nll_loss(output, label[idx_val])
    acc_val = accuracy(output, label[idx_val])
    z, output, recon_x, mu, logvar, z_sum = model(feature, adj, label_train, idx_test, z_mean=z_mean)
    loss_test = F.nll_loss(output, label[idx_test])
    acc_test = accuracy(output, label[idx_test])

    print("loss_val= {:.4f}".format(loss_val.item()),
          "acc_val= {:.4f}".format(acc_val.item()),
          "loss_test= {:.4f}".format(loss_test.item()),
          "acc_test= {:.4f}".format(acc_test.item()))
    return loss_val, loss_test, acc_val, acc_test


z_mean = []
for epoch in range(args.epoches):
    z_mean = train_encoder(epoch, model)
z_mean = z_mean.detach()
z_mean.requires_grad = False

min_loss = 1000000
best_acc_val = 0
best_acc_tst = 0
for epoch in range(args.epoch_base_model):
    train_base_model(epoch, z_mean)
    
# test
    if(epoch % 10 == 0):
        loss_val, loss_test, acc_val, acc_test = test(model, z_mean)
        if best_acc_val < acc_val:
            min_loss = loss_val
            best_acc_val = acc_val
            best_acc_tst = acc_test

print("best_acc_val:{:.4f}".format(best_acc_val))
print("best_acc_tst:{:.4f}".format(best_acc_tst))

    