import sys
import os.path as osp
# Add the parent directory to the Python path
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import yaml
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from utils.get_data import getDataSet, getAllKernels
from utils.get_metric import *
from utils.check import *

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from utils.get_metric import cal_acc, eva_svc

from MKLpy.algorithms import AverageMKL, EasyMKL, FHeuristic, GRAM, RMKL, MEMO, PWMK, CKA


def oneStepMKL(KL, Y):
    num_class=len(np.unique(Y))
    # if num_class > 2: raise ValueError('Combine_kernels requires binary classification problems')

    mkl_dict = {'AverageMKL': AverageMKL(),
                'EasyMKL': EasyMKL(lam=0.1),
                'GRAM': GRAM(),
                # 'R-MKL': RMKL(C=1),
                'MEMO': MEMO(theta=0.0, min_margin = 1e-4, solver = 'auto'),
                # 'PWMK': PWMK(delta = .4, cv=3),
                'FHeuristic': FHeuristic(),
                # 'CKA': CKA(),
                }
    # results_dict = {"Accuracy (SVM)": {}, "Accuracy (SC)": {}, "Normalized_Mutual_Info (SC)": {}, 'Adjusted_Rand_Index (SC)': {}}
    results_dict = {"Accuracy (SVM)": {}}

    for name, mkl in mkl_dict.items():
        
        solution = mkl.combine_kernels(KL, Y)
        K_candidate = solution.ker_matrix
        svm_results = eva_svc(K_candidate, y)

        results_dict["Accuracy (SVM)"][name] = "{:.4f}\\pm${:.4f}".format(svm_results[0], svm_results[1])
        print("%s: %.4f (%.4f)"%(name, svm_results[0], svm_results[1]))
    
    return results_dict


def equal_weight(K_list, y):
    num_class=len(np.unique(y))
    # if num_class > 2: raise ValueError('Combine_kernels requires binary classification problems')
    K = np.zeros_like(K_list[0])
    for i in range(len(K_list)):
        K += K_list[i]
    K /= len(K_list)
    
    y_pred = SpectralClustering(n_clusters = num_class, 
                                random_state = 0,
                                affinity = 'precomputed').fit_predict(K)
    acc_score = cal_acc(y, y_pred)
    nmi = metrics.normalized_mutual_info_score(y, y_pred)
    ari = metrics.cluster.adjusted_rand_score(y, y_pred)
    svm_results = eva_svc(K, y)

    print('Equal Weight:', acc_score, nmi, ari, svm_results)
    return {"Equal Weight": 
                {   
                    "Accuracy (SVM)": "{:.4f}\\pm${:.4f}".format(svm_results[0], svm_results[1]),
                    "Accuracy (SC)": "{:.4f}".format(acc_score),
                    "Normalized_Mutual_Info (SC)": "{:.4f}".format(nmi),
                    'Adjusted_Rand_Index (SC)': "{:.4f}".format(ari)
                }
            }



if __name__ == '__main__':
    
    dataSet = sys.argv[1]
    number = sys.argv[2]
    
    test_path =  f"./experiment/{dataSet}/{number}"
    print("Start with", test_path)

    with open(osp.join(test_path, 'para.yml'), 'r') as f:
        kwargs = yaml.safe_load(f)

    dataName = kwargs['data_name']
    kernelNameList = kwargs['kernel_name_list']

    G, y = getDataSet(dataName)
    K_list = getAllKernels(G, dataName, kernelNameList)

    results = oneStepMKL(K_list, y)
    # results = equal_weight(K_list, y)

    with open(osp.join(test_path, 'results_baseline.json'), 'w') as f:
        json.dump(results, f, indent=4)


