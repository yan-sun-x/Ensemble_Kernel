import os.path as osp

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from grakel.datasets import fetch_dataset

from utils.kernels import get_kernel

def getDataSet(dataName = 'MSRC_9', adjustByLabels = True):
    print(f'Fetching {dataName} dataset...')
    data = fetch_dataset(dataName, verbose=False, prefer_attr_nodes=False, produce_labels_nodes=True)
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
        gk = get_kernel(kernelName, G, dataName)
        if np.all(gk>=0):
            K_list.append(gk)
        else:
            print('Skip %s'%(kernelName))
    print('Get all %s kernels!'%(len(K_list)))
    return K_list



'''
Available datasets:
- AIDS
- BZR_MD
- COX2_MD
- PTC_MM
- SYNTHETIC
- ...

A graph is used to model pairwise relations (edges) between objects (nodes). 
A single graph in PyG is described by an instance of torch_geometric.data.Data, which holds the following attributes by default:

- data.x: Node feature matrix with shape [num_nodes, num_node_features]
- data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
- data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
- data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
- data.pos: Node position matrix with shape [num_nodes, num_dimensions]
'''

global_path = "/Users/mathilda/pycharmProjects/graph/data"


def loadDS(DS:str, batch_size:int = -1, ave_num_nodes = 10):

    class MyCustomTransform():
        '''
        Add two properties (split from data.x):
        - data.node_attr: Node feature matrix with shape [num_nodes, num_node_attribute]
        - data.node_label: Node feature matrix with shape [num_nodes, num_node_label]
        '''
        def __call__(self, data):
            # print(data.x.shape)
            # num_node = int(np.random.normal(ave_num_nodes, 2))
            # data.x = data.x[torch.randint(len(data.x), (num_node,)),:]
            # data.edge_index = torch.from_numpy(np.random.random_integers(0, num_node-1, size=(2, num_node**2//3)))
            # print(data.x.shape, data.edge_index.shape)
            if data.x is None:
                data.x = torch.ones(data.num_nodes,1)
            else:
                data.node_attr = data.x[:, :dataset.num_node_attributes]
                data.node_label = data.x[:, dataset.num_node_attributes:]
            return data

    path = osp.join(global_path, DS)
    dataset = TUDataset(path, name=DS, use_node_attr=True).shuffle()
    dataset.transform = MyCustomTransform() # add two properties
    if batch_size == -1: batch_size = len(dataset)
    return DataLoader(dataset, batch_size=batch_size), dataset.num_features


def get_graph_idx(data):
    # Get the index for list of graph embedding
    graph_idx = [0]
    graph_idx += [len(data[i].node_attr) for i in range(len(data))]
    graph_idx = np.cumsum(np.array(graph_idx))
    return graph_idx


def get_edge_idx(data):
    # Get the index for list of edge embedding
    edge_idx = [0]
    edge_idx += [data[i].edge_index.shape[1] for i in range(len(data))]
    edge_idx = np.cumsum(np.array(edge_idx))
    return edge_idx