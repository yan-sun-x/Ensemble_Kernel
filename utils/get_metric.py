import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics


def cal_acc(y_true, y_pred):
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


def individual_results(kernel_name, K_list, y, class_num):
    print('===== Individual acc_score | nmi=====')
    results_dict = {"Accuracy": {}, "Normalized_Mutual_Info": {}}
    i = 0
    for K_candidate in K_list:
        y_pred = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(K_candidate)

        acc_score = cal_acc(y, y_pred)
        nmi = metrics.normalized_mutual_info_score(y, y_pred)
        results_dict["Accuracy"][kernel_name[i]] = acc_score
        results_dict["Normalized_Mutual_Info"][kernel_name[i]] = nmi
        print("%s: %.4f(acc); %.4f(nmi)"%(kernel_name[i], acc_score, nmi))
        i += 1
    return results_dict

