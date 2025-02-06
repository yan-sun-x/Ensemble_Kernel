from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, PyramidMatch, WeisfeilerLehmanOptimalAssignment
from grakel.kernels import SubgraphMatching, GraphletSampling, ShortestPath, RandomWalk, NeighborhoodHash
import numpy as np
import os.path as osp


def get_kernel(name: str, G, dataName):

    cached_path = osp.join('./cache', osp.join(dataName))
    if osp.exists(osp.join(cached_path, "%s.npy"%(name))): 
        # print("Retrieved from cache")
        return np.load(osp.join(cached_path, "%s.npy"%(name)))
    
    name_split = name.split('_') # e.g. 'K_VH'
    abbr = name_split[1]
    if abbr == 'VH':
        kernel =  get_VH(G)

    elif abbr == 'PM':
        assert (len(name) == 8)
        L_value, d_value = int(name_split[2]), int(name_split[3])
        kernel = get_PM(G, L_value, d_value)

    elif abbr == 'NH':
        assert (len(name) == 8)
        R_value, bits_value = int(name_split[2]), int(name_split[3])
        kernel =  get_NH(G, R_value, bits_value)

    elif abbr == 'WL':
        assert (len(name) >= 6)
        iter_num = int(name_split[2])
        kernel = get_WL(G, iter_num)
    
    elif abbr == 'WLOA':
        assert (len(name) >= 6)
        iter_num = int(name_split[2])
        kernel = get_WL(G, iter_num)

    elif abbr == 'GS':
        assert (len(name) == 6)
        k_num = int(name_split[2])
        kernel = get_GS(G, k_num)

    elif abbr == 'SP':
        kernel = get_SP(G)

    elif abbr == 'RW':
        lambda_value = float(name_split[2])
        kernel = get_RW(G, lambda_value)

    if not osp.exists(cached_path): 
        import os
        os.mkdir(cached_path)
    np.save(osp.join(cached_path, "%s.npy"%(name)), kernel)
    # print("Save in cache")
    return kernel

def get_VH(G):
    wl_kernel = VertexHistogram(normalize=True, sparse='auto')
    K = wl_kernel.fit_transform(G)
    return K

def get_PM(G, L_value, d_value):
    wl_kernel = PyramidMatch(normalize=True, L = L_value, d = d_value)
    K = wl_kernel.fit_transform(G)
    return K


def get_NH(G, R_value = 3, bits_value = 8):
    wl_kernel = NeighborhoodHash(normalize=True, R = R_value, bits = bits_value)
    K = wl_kernel.fit_transform(G)
    return K


def get_WL(G, iter_num = 1):
    wl_kernel = WeisfeilerLehman(n_iter=iter_num, normalize=True, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform(G)
    return K

def get_WLOA(G, iter_num = 1):
    wl_kernel = WeisfeilerLehmanOptimalAssignment(n_iter=iter_num, normalize=True, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform(G)
    return K


def get_GS(G, k_num = 4):
    wl_kernel = GraphletSampling(normalize=True, k=k_num)
    K = wl_kernel.fit_transform(G)
    return K


def get_SP(G, algo = 'auto'):
    wl_kernel = ShortestPath(normalize=True, algorithm_type=algo)
    K = wl_kernel.fit_transform(G)
    return K


def get_RW(G, lambda_value = 0.1):
    wl_kernel = RandomWalk(lamda=lambda_value, normalize=True, kernel_type="exponential")
    K = wl_kernel.fit_transform(G)
    return K