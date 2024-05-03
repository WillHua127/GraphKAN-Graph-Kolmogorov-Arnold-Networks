import argparse
import random

from kan_layer import NaiveFourierKANLayer as KANLayer
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.datasets import Planetoid, WebKB
import torch_geometric.transforms as T
from torch_geometric.utils import *

class KanGNN(torch.nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, grid_feat, num_layers, use_bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_feat, hidden_feat)
        #self.lin_in = KANLayer(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(KANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.lins.append(nn.Linear(hidden_feat, out_feat, bias=False))
        #self.lins.append(KANLayer(hidden_feat, out_feat, grid_feat, addbias=False))

        # self.lins = torch.nn.ModuleList()
        # self.lins.append(nn.Linear(in_feat, hidden_feat, bias=use_bias))
        # for i in range(num_layers):
        #     self.lins.append(nn.Linear(hidden_feat, hidden_feat, bias=use_bias))
        # self.lins.append(nn.Linear(hidden_feat, out_feat, bias=use_bias))

    
    def forward(self, x, adj):
        x = self.lin_in(x)
        #x = self.lin_in(spmm(adj, x))
        for layer in self.lins[:self.num_layers-1]:
            x = layer(spmm(adj, x))
            #x = layer(x)
        x = self.lins[-1](x)
            
        return x.log_softmax(dim=-1)


def train(args, feat, adj, label, mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(feat, adj)
    pred, true = out[mask], label[mask]
    loss = F.nll_loss(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()

@torch.no_grad()
def eval(args, feat, adj, model):
    model.eval()
    with torch.no_grad():
        pred = model(feat, adj)
    pred = pred.argmax(dim=-1)
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', type=str, default='/home/sitao/kan_gnn/data/')
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--logger_path', type=str, default='logger/esm')
    # model
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--hidden_size', type=int, default=2)
    parser.add_argument('--grid_size', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=2)
    # training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    # optimizer
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #args.device = torch.device('cpu')


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])

    print(f'run experiments on {args.name} dataset')

    if args.name in {'Cora', 'Pubmed'}:
        dataset = Planetoid(args.path, args.name, transform=transform)[0]
    elif args.name in {'Cornell'}:
        dataset = WebKB(args.path, args.name, transform=transform)[0]
    
    in_feat = dataset.num_features
    out_feat = max(dataset.y) + 1

    model = KanGNN(
                   in_feat=in_feat,
                   hidden_feat=args.hidden_size, 
                   out_feat=out_feat, 
                   grid_feat=args.grid_size,
                   num_layers=args.n_layers,
                  ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)

    adj      = dataset.adj_t.to(args.device)
    feat     = dataset.x.float().to(args.device)
    label    = dataset.y.to(args.device)

    trn_mask, val_mask, tst_mask = random_disassortative_splits(label, out_feat)
    trn_mask, val_mask, tst_mask = trn_mask.to(args.device), val_mask.to(args.device), tst_mask.to(args.device)
    for epoch in range(args.epochs):
        trn_acc, trn_loss = train(args, feat, adj, label, trn_mask, model, optimizer)
        pred = eval(args, feat, adj, model)
        val_acc = int((pred[val_mask] == label[val_mask]).sum()) / int(val_mask.sum())
        tst_acc = int((pred[tst_mask] == label[tst_mask]).sum()) / int(tst_mask.sum())

        print(f'Epoch: {epoch:04d}, Trn_loss: {trn_loss:.4f}, Trn_acc: {trn_acc:.4f}, Val_acc: {val_acc:.4f}, Test_acc: {tst_acc:.4f}')

    



