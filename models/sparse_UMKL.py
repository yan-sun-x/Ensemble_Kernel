from __future__ import division
from __future__ import print_function

import time
from typing import List, Dict
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from models.ensemble import EnsembleKernels, project_simplex
from utils.get_data import getDataSet, getAllKernels
from utils.get_metric import cal_acc, eva_svc
from utils.log import Log


def sparse_UMKL_loss(W: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    loss = torch.sum(W * torch.cdist(K, K, p=2) ** 2)
    return loss


def get_W(K_list: List[torch.Tensor], n_neighbors: int, N: int) -> torch.Tensor:
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
    W = np.zeros((N, N))
    for k in K_list:
        neigh.fit(k)
        A = neigh.kneighbors_graph(k)
        W += A.toarray()

    return torch.from_numpy(W)


def train(model: torch.nn.Module, 
          optimizer: Optimizer, 
          w_matrix: torch.Tensor) -> Log:

    t = time.time()
    model.train()
    optimizer.zero_grad()

    k_new = model()
    loss_train = sparse_UMKL_loss(w_matrix, k_new)
    loss_train.backward()
    optimizer.step()
    model.weights.data = project_simplex(model.weights.detach())
    k_new = model()
        
    log = Log(k_new.detach().numpy(), model.weights.detach().tolist(), loss_train.item(), time.time() - t)
    return log
    


def autoGraphKernel(args: argparse.Namespace, kernelNameList: List[str]) -> Dict[str, List]:
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    G, y = getDataSet(args.dataset)
    class_num = len(np.unique(y))
    K_list = getAllKernels(G, args.dataset, kernelNameList)
    kernels = [torch.from_numpy(k) for k in K_list]
    
    # Model and optimizer
    model = EnsembleKernels(kernels, args.init_type)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    w_matrix = get_W(K_list, args.n_neighbors, len(G))
    t_total = time.time() - t_total

    # Train model
    results = {'Accuracy': [], "Normalized_Mutual_Info": [], 'Adjusted_Rand_Index': [], 'SVC_cv10': [], 'Weights': [], 'Loss':[ ]}
    
    for epoch in range(args.epochs):
        
        log = train(model, optimizer, w_matrix)
        
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