from typing import Set
import argparse

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from utils.get_data import getDataSet, getAllKernels


def cal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def eva_svc(K: np.ndarray, y_true: np.ndarray) -> Set[float]:
    '''
    Uses the SVM classifier to perform classification
    '''
    clf = SVC(kernel="precomputed", tol=1e-6, probability=True)
    acc_score = cross_val_score(clf, K, y_true, cv=5)
    return (np.mean(acc_score).item(), np.std(acc_score).item())


def individual_results(args: argparse.Namespace, kernel_name: str) -> dict:
    
    G, y = getDataSet(args.dataset)
    class_num = len(np.unique(y))
    K_list = getAllKernels(G, args.dataset, kernel_name)

    class_num = len(np.unique(y))
    print('===== Individual acc_score | nmi=====')
    results_dict = {"Accuracy (SVM)": {}, "Accuracy (SC)": {}, "Normalized_Mutual_Info (SC)": {}, 'Adjusted_Rand_Index (SC)': {}}
    i = 0
    for K_candidate in K_list:
        y_pred = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(K_candidate)

        acc_score = cal_acc(y, y_pred)
        nmi = metrics.normalized_mutual_info_score(y, y_pred)
        ari = metrics.cluster.adjusted_rand_score(y, y_pred)
        svm_results = eva_svc(K_candidate, y)
        
        results_dict["Accuracy (SVM)"][kernel_name[i]] = "{:.4f}\pm${:.4f}".format(svm_results[0], svm_results[1])
        results_dict["Accuracy (SC)"][kernel_name[i]] = acc_score
        results_dict["Normalized_Mutual_Info (SC)"][kernel_name[i]] = nmi
        results_dict["Adjusted_Rand_Index (SC)"][kernel_name[i]] = ari
        print("%s: %.4f(acc); %.4f(nmi); %.4f(ari); %.4f (%.4f)"%(kernel_name[i], acc_score, nmi, ari, svm_results[0], svm_results[1]))
        i += 1
    return results_dict