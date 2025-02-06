from utils.get_metric import cal_acc, eva_svc
from typing import List, Union, Set
import scipy
import numpy as np

import torch#; torch.manual_seed(2024)
from torch.nn import Parameter


def project_simplex(x: torch.Tensor) -> torch.Tensor:
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


def cal_Lap_eigen(K: torch.Tensor, class_num: int) -> torch.Tensor:
    L = scipy.sparse.csgraph.laplacian(K.cpu().detach().numpy(), normed=True)
    eigenValues, _ = scipy.linalg.eigh(L, subset_by_index=[class_num-1, class_num])
    eigenDiff = torch.from_numpy(np.diff(eigenValues))
    return eigenDiff


class EnsembleKernels(torch.nn.Module):

    def __init__(self, kernels: List[torch.Tensor], init_type: Union[str, list]='uniform', class_num: int = 2):
        super(EnsembleKernels, self).__init__()
        # initialize weights
        n = len(kernels) # n: number of kernels

        if init_type == 'uniform':
            self.weights = torch.ones(n)/n

        elif init_type == 'random':
            self.weights = torch.distributions.dirichlet.Dirichlet(torch.ones(n)).sample()

        elif init_type == 'eigen':
            self.weights = torch.zeros(n)
            for i, k in enumerate(kernels):
                self.weights[i] = cal_Lap_eigen(k, class_num)
            self.weights /= self.weights.sum()

        elif init_type == 'eigen_inv':
            self.weights = torch.zeros(n)
            for i, k in enumerate(kernels):
                self.weights[i] = cal_Lap_eigen(k, class_num)
            self.weights = self.weights.sum() - self.weights
            self.weights /= self.weights.sum()


        elif type(init_type)==list:
            self.weights = torch.Tensor(init_type)
        else:
            raise ValueError('Init_type is wrong.')
        
        self.weights = Parameter(self.weights)
        self.kernels = torch.stack(kernels).to(torch.float32)
        self.X = torch.cat(kernels, dim=1).to(torch.float32)

    def forward(self) -> torch.Tensor:
        k_mul = self.kernels * self.weights[:,None,None]
        k_new = torch.sum(k_mul, dim=0)
        return k_new