from __future__ import division
from __future__ import print_function

import time
from typing import List, Dict, Union
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from models.ensemble import EnsembleKernels, project_simplex
from utils.get_data import getDataSet, getAllKernels
from utils.get_metric import cal_acc, eva_svc
from utils.log import Log
from sklearn.model_selection import train_test_split


def power_kl_loss(ker_mat: torch.Tensor, pow: int = 2, fixed_ground_truth: Union[None, torch.Tensor] = None, forward_KL: bool = True) -> torch.Tensor:

    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    Q = ker_mat / ker_mat.sum(1, keepdim=True)
    if fixed_ground_truth is None:
        P = ker_mat ** pow #/ ker_mat.sum(0)
        P = P / P.sum(1, keepdim=True)
    else:
        P = fixed_ground_truth

    if forward_KL:
        return loss(P.log(), Q)
    else:
        return loss(Q.log(), P)


def multi_kl_loss(ker_mat: torch.Tensor, k_list: List[torch.Tensor], forward_KL: bool = True) -> torch.Tensor:

    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    tot_loss = torch.zeros(1)
    Q = ker_mat / ker_mat.sum(1, keepdim=True)
    for k in k_list:
        P = k / k.sum(1, keepdim=True)
        if forward_KL:
            tot_loss += loss(P, Q)
        else:
            tot_loss += loss(Q, P)
    
    tot_loss /= len(k_list)
    return tot_loss


def power_ce_loss(ker_mat: torch.Tensor, pow: int = 2, fixed_ground_truth: Union[None, torch.Tensor] = None) -> torch.Tensor:
    
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    Q = ker_mat / ker_mat.sum(1, keepdim=True)
    if fixed_ground_truth is None:
        P = ker_mat ** pow #/ ker_mat.sum(0)
        P = P / P.sum(1, keepdim=True)
    else:
        P = fixed_ground_truth

    return loss(P, Q)


def multi_ce_loss(ker_mat: torch.Tensor, k_list: List[torch.Tensor]) -> torch.Tensor:

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    tot_loss = torch.zeros(1)
    Q = ker_mat / ker_mat.sum(1, keepdim=True)
    for k in k_list:
        P = k / k.sum(1, keepdim=True)
        tot_loss += loss(P, Q)
    
    tot_loss /= len(k_list)
    return tot_loss


def train(model: torch.nn.Module, 
          optimizer: Optimizer, 
          loss_fun: str, 
          power=2, 
          kernels=None,
          fixed_ground_truth=None,
          forward_loss=True
          ) -> Log:
    
    t = time.time()
    model.train()
    optimizer.zero_grad()

    k_new = model()
    if loss_fun == 'PCE':
        loss_train = power_ce_loss(k_new, power, fixed_ground_truth)
    elif loss_fun == 'MCE':
        loss_train = multi_ce_loss(k_new, kernels)
    elif loss_fun == 'PKL':
        loss_train = power_kl_loss(k_new, power, fixed_ground_truth, forward_loss)
    elif loss_fun == 'MKL':
        loss_train = multi_kl_loss(k_new, kernels, forward_loss)
    else:
        raise ValueError('Invalid loss function!')
    
    loss_train.backward()
    optimizer.step()
    model.weights.data = project_simplex(model.weights.detach())
    duration = time.time() - t
    k_new = model()

    log = Log(k_new.detach().numpy(), model.weights.detach().tolist(), loss_train.item(), duration)
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
    if args.perturb:
        for i in range(len(kernels)):
            noise = torch.randn_like(kernels[i]) * args.noise_std
            kernels[i] += (noise + noise.T) * 0.5

    # Model and optimizer
    model = EnsembleKernels(kernels, args.init_type)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    results = {'Accuracy': [], 
               "Normalized_Mutual_Info": [], 
               'Adjusted_Rand_Index': [], 
               'SVC_cv10': [], 
               'Weights': [], 
               'Loss':[]}
    p_list = []
    t_total = time.time() - time.time()
    fixed_ground_truth = None

    for epoch in range(args.epochs):

        try:
            if args.loss_fun in ['PKL', 'PCE']:
                if args.set_fixed_ground_truth and (epoch==0):
                    log = train(model, optimizer, args.loss_fun, power=args.power, forward_loss=args.forward_loss)
                    fixed_ground_truth = log.k_new ** args.power
                    fixed_ground_truth = fixed_ground_truth / fixed_ground_truth.sum(1)
                    fixed_ground_truth = torch.from_numpy(fixed_ground_truth)
                log = train(model, optimizer, args.loss_fun, power=args.power, fixed_ground_truth=fixed_ground_truth, forward_loss=args.forward_loss)

            elif args.loss_fun in ['MKL', 'MCE']:
                log = train(model, optimizer, args.loss_fun, kernels=kernels, forward_loss=args.forward_loss)

            else:
                raise ValueError('Invalid loss function!')

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
            p_list.append((log.k_new / log.k_new.sum(1)).tolist()) # record probablistic simplex
            t_total += log.duration
        except:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(t_total))

    results['Best'] = [('Accuracy', max(results['Accuracy']), np.argmax(results['Accuracy']).item()), 
                       ('Normalized_Mutual_Info', max(results['Normalized_Mutual_Info']), np.argmax(results['Normalized_Mutual_Info']).item()), 
                       ('Adjusted_Rand_Index', max(results['Adjusted_Rand_Index']), np.argmax(results['Adjusted_Rand_Index']).item()),
                    ]
    results['Time'] = (t_total, len(G), len(K_list))
    return results, p_list


def autoGraphKernelTest(args: argparse.Namespace, kernelNameList: List[str]) -> Dict[str, List]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    G, y = getDataSet(args.dataset)
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=args.test_size, random_state=args.seed)
    print(G_train.shape, G_test.shape, y_train.shape, y_test.shape)
    print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

    class_num = len(np.unique(y))
    train_dataname = args.dataset + '_train_' + str(args.test_size)
    K_list = getAllKernels(G_train, train_dataname, kernelNameList)
    print('kernels', K_list[0].shape)
    kernels = [torch.from_numpy(k) for k in K_list]

    if args.perturb:
        for i in range(len(kernels)):
            noise = torch.randn_like(kernels[i]) * args.noise_std
            kernels[i] += (noise + noise.T) * 0.5
    
    # Model and optimizer
    model = EnsembleKernels(kernels, args.init_type)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    results = {'Accuracy': [], 
               "Normalized_Mutual_Info": [], 
               'Adjusted_Rand_Index': [], 
               'SVC_cv10': [], 
               'Weights': [], 
               'Loss':[]}
    p_list = []
    t_total = time.time()
    fixed_ground_truth = None

    for epoch in range(args.epochs):

        try:
            if args.loss_fun in ['PKL', 'PCE']:
                if args.set_fixed_ground_truth and (epoch==0):
                    log = train(model, optimizer, args.loss_fun, power=args.power, forward_loss=args.forward_loss)
                    fixed_ground_truth = log.k_new ** args.power
                    fixed_ground_truth = fixed_ground_truth / fixed_ground_truth.sum(1)
                    fixed_ground_truth = torch.from_numpy(fixed_ground_truth)
                log = train(model, optimizer, args.loss_fun, power=args.power, fixed_ground_truth=fixed_ground_truth, forward_loss=args.forward_loss)

            elif args.loss_fun in ['MKL', 'MCE']:
                log = train(model, optimizer, args.loss_fun, kernels=kernels, forward_loss=args.forward_loss)

            else:
                raise ValueError('Invalid loss function!')

            y_pred = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(log.k_new)
            acc_score = cal_acc(y_train, y_pred)
            nmi = metrics.normalized_mutual_info_score(y_train, y_pred)
            ari = metrics.cluster.adjusted_rand_score(y_train, y_pred)
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
            p_list.append((log.k_new / log.k_new.sum(1)).tolist()) # record probablistic simplex
        except:
                break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    results['Best'] = [('Accuracy', max(results['Accuracy']), np.argmax(results['Accuracy']).item()), 
                       ('Normalized_Mutual_Info', max(results['Normalized_Mutual_Info']), np.argmax(results['Normalized_Mutual_Info']).item()), 
                       ('Adjusted_Rand_Index', max(results['Adjusted_Rand_Index']), np.argmax(results['Adjusted_Rand_Index']).item()) 
                    ]

    # Test
    test_dataname = args.dataset + '_test_' + str(args.test_size)
    K_test_list = getAllKernels(G_test, test_dataname, kernelNameList)
    kernels_test = [torch.from_numpy(k) for k in K_test_list]
    kernels_test = np.stack(kernels_test)

    best_weights = results['Weights'][results['Best'][0][2]] # Take the best weights from ACC
    best_weights = np.array(best_weights)
    k_test_mul = kernels_test * best_weights[:,None,None]
    k_test_new = np.sum(k_test_mul, axis=0)

    if args.loss_fun == 'PCE':
        loss_test = power_ce_loss(torch.from_numpy(k_test_new), args.power, fixed_ground_truth)
    elif args.loss_fun == 'MCE':
        loss_test = multi_ce_loss(torch.from_numpy(k_test_new), kernels)
    elif args.loss_fun == 'PKL':
        loss_test = power_kl_loss(torch.from_numpy(k_test_new), args.power, fixed_ground_truth, args.forward_loss)
    elif args.loss_fun == 'MKL':
        loss_test = multi_kl_loss(torch.from_numpy(k_test_new), kernels, args.forward_loss)
    else:
        raise ValueError('Invalid loss function!')

    y_pred_test = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(k_test_new)

    acc_score_test = cal_acc(y_test, y_pred_test)
    nmi_test = metrics.normalized_mutual_info_score(y_test, y_pred_test)
    ari_test = metrics.cluster.adjusted_rand_score(y_test, y_pred_test)
    svm_results_test = eva_svc(k_test_new, y_pred_test)

    print('Test: acc_score: {:.4f}; nmi: {:.4f}; ari: {:.4f} (SC)'.format(acc_score_test, nmi_test, ari_test))
    print('Test: acc: {:.4f}$\pm${:.4f} (SVM)'.format(svm_results_test[0], svm_results_test[1]))

    results['Test'] = {
                        'Loss': loss_test.item(),
                        'Accuracy': acc_score_test, 
                        'Normalized_Mutual_Info': nmi_test, 
                        'Adjusted_Rand_Index': ari_test, 
                        'SVC_cv10': f'{svm_results_test[0]}$\pm${svm_results_test[1]}'
                        }

    return results, p_list
