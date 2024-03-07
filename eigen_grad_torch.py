import numpy as np
import scipy
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from utils.get_metric import cal_acc
from typing import List
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Parameter



def cal_Lap_eigen(K, class_num):
    L = scipy.sparse.csgraph.laplacian(K.cpu().detach().numpy(), normed=True)
    eigenValues, eigenVectors = scipy.linalg.eigh(L, subset_by_index=[class_num-1, class_num])
    eigenValues = torch.from_numpy(eigenValues)
    eigenVectors = torch.from_numpy(eigenVectors)
    return (eigenValues, eigenVectors[0], eigenVectors[1])


class EnsembleKernels(torch.nn.Module):
    def __init__(self, kernels: List[torch.Tensor], init_type, class_num:int):
        super(EnsembleKernels, self).__init__()
        # initialize weights
        n = len(kernels) # n: number of kernels
        m, u = kernels[0].size() # m, u: size of one kernel
        self.input_dim = m
        self.query = nn.Linear(m, u)
        self.key = nn.Linear(m, u)
        self.value = nn.Linear(m, u)
        self.softmax = nn.Softmax(dim=2)
        if init_type == 'uniform':
            self.weights = torch.ones(n)/n
            
        elif init_type == 'eigen':
            self.weights = torch.zeros(n)
            for i, k in enumerate(kernels):
                eigenValues, _, _ = cal_Lap_eigen(k, class_num)
                self.weights[i] = torch.diff(eigenValues)
            self.weights /= self.weights.sum()

        elif init_type == 'eigen_inv':
            self.weights = torch.zeros(n)
            for i, k in enumerate(kernels):
                eigenValues, _, _ = cal_Lap_eigen(k, class_num)
                self.weights[i] = torch.diff(eigenValues)
            self.weights = self.weights.sum() - self.weights
            self.weights /= self.weights.sum()

        elif init_type == 'random':
            self.weights = torch.distributions.dirichlet.Dirichlet(torch.ones(n)).sample()

        elif type(init_type)==list:
            self.weights = torch.Tensor(init_type)
        else:
            raise ValueError('Init_type is wrong.')
        
        self.weights = Parameter(self.weights)
        self.kernels = torch.stack(kernels).to(torch.float32)

    def forward(self):
        queries = self.query(self.kernels)
        keys = self.key(self.kernels)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = self.kernels * attention
        k_mul = weighted * self.weights[:,None,None]
        k_new = torch.sum(k_mul, dim=0)
        return k_new



def kl_loss(s):
    '''
    - s: 	(n, n) pairwise similarity scores
    '''
    loss = torch.nn.KLDivLoss(reduction="batchmean")
    weight = s**2 / s.sum(0)
    return loss(s, (weight.t()/weight.sum(1)).t())


def gk_L(K, coef = 1):

    n = K.shape[0]
    D = torch.sum(K, axis = 0)

    D15 = torch.diag(torch.pow(D, -1.5))
    D05 = torch.diag(torch.pow(D, -0.5))
    D1 = torch.diag(torch.pow(D, -1))

    U0 = - 0.5 * D15 @ K @ D05 * coef
    U1 = - 0.5 * D05 @ K @ D15 * coef

    U0 = torch.tile(U0.sum(axis = 1), (n, 1))
    U1 = (torch.tile(U1.sum(axis = 0), (n, 1))).t()
    grad_K = - (U0 + U1 + D1 * coef)
    
    return grad_K



def cal_grad_w(kernels, k_new, class_num, return_grad_A = False):

    eigenValues, eigenV_kp1, eigenV = cal_Lap_eigen(k_new, class_num)
    eigen_k = eigenValues[0]
    eigen_gap = torch.diff(eigenValues)
    eigen_gap_ratio = eigen_gap/eigen_k
    eigenV_kp1_trace = (eigenV_kp1 @ eigenV_kp1).t()
    eigenV_trace = (eigenV @ eigenV).t()
    inv_trace1 = torch.pow(eigen_gap, -1) * (eigenV_kp1_trace - eigenV_trace)
    inv_trace2 = torch.pow(eigen_k,-1) * (eigenV_trace)

    grad_LK = gk_L(k_new, (inv_trace1 - inv_trace2))
    grad_Lw = torch.stack(kernels)
    grad = grad_Lw * grad_LK 
    grad = grad.sum(axis = 1).sum(axis = 1) 

    if return_grad_A:
        grad_A = grad_LK
        return grad.flatten(), grad_A, eigen_gap_ratio
        
    return grad.flatten(), eigen_gap_ratio



def project_simplex(x):
    """ Take a vector x (with possible nonnegative entries and non-normalized)
        and project it onato the unit simplex.
    """

    xsorted, _ = torch.sort(x, descending=True)
    # remaining entries need to sum up to 1
    sum_ = 1.0
    lambda_a = (torch.cumsum(xsorted, dim=0) - sum_) / torch.arange(1.0, len(xsorted)+1.0)
    for i in range(len(lambda_a)-1):
        if lambda_a[i] >= xsorted[i+1]:
            astar = i
            break
    else:
        astar = -1

    p = torch.maximum(x-lambda_a[astar],  torch.zeros(1))
    return p


def train(K_list, init_type, y, class_num = 2, stepsize = 1e-4, num_iter = 200, weight_decay=0., add_kl_loss=False, add_eigen_loss=True):
    # TODO: how to further simplify the ensemble process, do we need to get all of the kernels ? 

    kernels = [torch.from_numpy(k) for k in K_list]
    model = EnsembleKernels(kernels, init_type, class_num)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), \
                                lr=stepsize,        \
                                weight_decay = 0.0)
    pbar = tqdm(range(1, num_iter + 1))
    results_dict = {'Accuracy': [], "Normalized_Mutual_Info": [], 'Weights': [], 'Loss':[ ]}
    score_best = 0.0
    obj = 0
    weights_best = None
    k_best = None
    for epoch in pbar:
        k_new = model()

        if add_kl_loss:
            train_loss = kl_loss(k_new)
            train_loss.backward()
            optimizer.step()

        if add_eigen_loss:
            grad, obj = cal_grad_w(kernels, k_new, class_num) #0, torch.zeros(1)
            model.weights.data = project_simplex(model.weights.detach() + 1e-3 * stepsize * grad)
        
        else:
            model.weights.data = project_simplex(model.weights.detach())

        
        y_pred = SpectralClustering(n_clusters = class_num, 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(k_new.detach().numpy())

        acc_score = cal_acc(y, y_pred)
        nmi = metrics.normalized_mutual_info_score(y, y_pred)
        results_dict['Accuracy'].append(acc_score)
        results_dict['Normalized_Mutual_Info'].append(nmi)
        results_dict['Weights'].append(model.weights.detach().tolist())
        if add_kl_loss:
            if add_eigen_loss:
                results_dict['Loss'].append(train_loss.item()-1e-3 * obj.item())
            else:
                results_dict['Loss'].append(train_loss.item())
        else:
            results_dict['Loss'].append(-1e-3 * obj.item())


        if score_best < acc_score: 
            weights_best = model.weights.detach().tolist()
            k_best = k_new.detach().numpy()
            score_best = acc_score

        if (epoch%10 == 0): 
            weight_array = model.weights.detach().numpy()
            weight_norm = np.linalg.norm(weight_array)
            weight_sum = np.sum(weight_array)
            print(f'----- At No.{epoch+1} iteration: sum: {np.round(weight_sum, 4)} norm: {np.round(weight_norm, 4)}-----')
            print(f"Total Loss: {results_dict['Loss'][-1]}")
            print(f"acc_score: {acc_score}; nmi: {nmi}\n")
    
    k_new = k_new.detach().numpy()
    print(k_new)
    return k_new, k_best, weights_best, results_dict

