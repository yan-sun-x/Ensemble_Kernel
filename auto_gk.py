import numpy as np
from grakel.datasets import fetch_dataset
import os.path as osp
import json
import warnings
warnings.filterwarnings("ignore")

from utils.get_metric import *
from utils.check import *
from utils.kernels import get_kernel
# import eigen_grad 
import eigen_grad_torch


def getDataSet(dataName = 'MSRC_9', adjustByLabels = True):
    print(f'Fetching {dataName} dataset...')
    data = fetch_dataset(dataName.upper(), verbose=False, prefer_attr_nodes=False)
    print(f'Finish fetching!')
    G, y = data.data, data.target
    G = np.array(G)
    
    if adjustByLabels:
        G = np.row_stack([G[np.where(y == num)[0].astype(np.uint64)] for num in np.unique(y)])
        y = np.concatenate([y[np.argwhere(y == num).ravel()] for num in np.unique(y)])
    
    return G, y


def getAllKernels(G, dataName, kernelNameList = []):
    K_list = []
    for kernelName in kernelNameList:
        print('Get kernel %s...'%(kernelName))
        K_list.append(get_kernel(kernelName, G, dataName))
    return K_list


def get_Knew(prev_w, K_list):
    T = len(prev_w)
    p, q = K_list[0].shape

    K_new = np.zeros((p, q))
    for t in range(T):
        K_new += prev_w[t] * K_list[t]
    return K_new


def combination_result(kwargs):
    dataName = kwargs['data_name']
    kernelNameList = kwargs['kernel_name_list']
    weights = kwargs['init_type']

    G, y = getDataSet(dataName)
    class_num = len(np.unique(y))
    K_list = getAllKernels(G, dataName, kernelNameList)
    k_new = get_Knew(weights, K_list)
    y_pred = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(k_new)

    acc_score = cal_acc(y, y_pred)
    nmi = metrics.normalized_mutual_info_score(y, y_pred)
    print("%.4f(acc); %.4f(nmi)"%(acc_score, nmi))


def autoGraphKernel(kwargs):

    dataName = kwargs['data_name']
    kernelNameList = kwargs['kernel_name_list']

    G, y = getDataSet(dataName)
    K_list = getAllKernels(G, dataName, kernelNameList)

    init_type = kwargs['init_type']
    stepsize  = kwargs['stepsize']
    num_iter  = kwargs['num_iter']
    weight_decay = kwargs.get('weight_decay',0.0)
    add_kl_loss = kwargs.get('add_kl_loss', False)
    add_eigen_loss = kwargs.get('add_eigen_loss', True)

    class_num = len(np.unique(y))
    K_new, K_best, weights_best, results_dict = eigen_grad_torch.train(K_list, 
                                                                    init_type, 
                                                                    y, 
                                                                    class_num, 
                                                                    stepsize, 
                                                                    num_iter,
                                                                    weight_decay,
                                                                    add_kl_loss,
                                                                    add_eigen_loss)
    
    results_dict['Check_Prop'] = {"isSymmetry": check_symmetric(K_new),\
                            "isTriangIneq": check_tria_ineq(K_new),\
                            "isPSD": check_psd(K_new)}

    save_root = kwargs['save_root']
    np.savetxt(osp.join(save_root, 'weights.txt'), weights_best)
    np.save(osp.join(save_root, 'K_new.npy'), K_new)
    np.save(osp.join(save_root, 'K_best.npy'), K_best)

    with open(osp.join(save_root, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    indi_dict = individual_results(kernelNameList, K_list, y, class_num)
    indi_dict["Accuracy"]['Joint'] = max(results_dict["Accuracy"])
    indi_dict["Normalized_Mutual_Info"]['Joint'] = max(results_dict["Normalized_Mutual_Info"])
    with open(osp.join(save_root, 'results_compared.json'), 'w') as f:
        json.dump(indi_dict, f, indent=4)


