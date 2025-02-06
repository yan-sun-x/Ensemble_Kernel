from __future__ import division
from __future__ import print_function

import time
from typing import List, Dict
import argparse

import torch
from torch import linalg as LA
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.nn import Linear, Parameter, ModuleList
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from models.ensemble import EnsembleKernels, project_simplex
from utils.get_data import getDataSet, getAllKernels, loadDS
from utils.get_metric import cal_acc, eva_svc
from utils.log import Log


def UMKL_loss(X, K, gamma):
    loss = 0.5 * LA.matrix_norm(X - K @ X) ** 2
    loss += gamma * torch.sum(K * torch.cdist(X, X, p=2) ** 2)
    return loss


class GCNConv(MessagePassing):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix. 
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, dtype=x.dtype)
        # deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out


class GCN(nn.Module):
    def __init__(self, gcn_input_dim, gcn_num_layers, pool='sum'):
        super(GCN, self).__init__()
        
        self.gcn_num_layers = gcn_num_layers

        dim_list = [gcn_input_dim for _ in range(gcn_num_layers+1)]

        self.gcn_list = ModuleList([GCNConv(dim_list[i], dim_list[i+1]) for i in range(gcn_num_layers)])
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool

    def forward(self, x, edge_index, batch):
        for layer in range(self.gcn_num_layers):
            x = self.gcn_list[layer](x, edge_index)

        xpool = self.pool(x, batch)
        return xpool


def train(model: torch.nn.Module, 
          model2: torch.nn.Module, 
          optimizer: Optimizer, 
          dataloader: torch.utils.data.DataLoader, 
          epoch: int, 
          pre_epochs: int) -> Log:
    
    t = time.time()
    for data in dataloader:
        if epoch < pre_epochs:
            model.train()
            output = model(data.x, data.edge_index, data.batch)
        else:
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.batch)
    
        model2.train()
        k_new = model2()

        optimizer.zero_grad()
        loss_train = UMKL_loss(output, k_new, gamma=1)
        loss_train.backward()
        optimizer.step()

        model2.weights.data = project_simplex(model2.weights.detach())
        k_new = model2()

    log = Log(k_new.detach().numpy(), model2.weights.detach().tolist(), loss_train.item(), time.time() - t)
    return log
    


def autoGraphKernel(args: argparse.Namespace, kernelNameList: List[str]) -> Dict[str, List]:
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed
                               )
    # Load data
    dataloader, num_features = loadDS(args.dataset)
    G, y = getDataSet(args.dataset)
    class_num = len(np.unique(y))
    K_list = getAllKernels(G, args.dataset, kernelNameList)
    kernels = [torch.from_numpy(k) for k in K_list]

    # Model and optimizer
    model = GCN(gcn_input_dim=num_features, gcn_num_layers=4, pool='mean')
    model2 = EnsembleKernels(kernels, args.init_type)
    optimizer = optim.Adam(list(model.parameters()) + list(model2.parameters()),
                        lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    results = {'Accuracy': [], "Normalized_Mutual_Info": [], 'Adjusted_Rand_Index': [], 'SVC_cv10': [], 'Weights': [], 'Loss':[ ]}
    
    t_total = time.time() - time.time()

    for epoch in range(args.epochs):
        
        log = train(model, model2, optimizer, dataloader, epoch, args.pre_epochs)
        
        y_pred = SpectralClustering(n_clusters = class_num, 
                                random_state = 0,
                                affinity = 'precomputed').fit_predict(log.k_new)

        acc_score = cal_acc(y, y_pred)
        nmi = metrics.normalized_mutual_info_score(y, y_pred)
        ari = metrics.cluster.adjusted_rand_score(y, y_pred)
        svm_results = eva_svc(log.k_new, y_pred)

        print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(log.loss_train),
                'time: {:.4f}s'.format(log.duration),
                "acc_score: {:.4f}; nmi: {:.4f}; ari: {:.4f} (SC)".format(acc_score, nmi, ari),
                "acc: {:.4f}$\pm${:.4f} (SVM)\n".format(svm_results[0], svm_results[1])
                )

        results['Accuracy'].append(acc_score)
        results['Normalized_Mutual_Info'].append(nmi)
        results['Adjusted_Rand_Index'].append(ari)
        results['SVC_cv10'].append(f'{svm_results[0]}$\pm${svm_results[1]}')
        results['Weights'].append(log.weight)
        results['Loss'].append(log.loss_train)
        t_total += log.duration

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(t_total))

    results['Best'] = [('Accuracy', max(results['Accuracy']), np.argmax(results['Accuracy']).item()), 
                       ('Normalized_Mutual_Info', max(results['Normalized_Mutual_Info']), np.argmax(results['Normalized_Mutual_Info']).item()), 
                       ('Adjusted_Rand_Index', max(results['Adjusted_Rand_Index']), np.argmax(results['Adjusted_Rand_Index']).item()) 
                    ]

    results['Time'] = (t_total, len(G), len(K_list))
    return results