import numpy as np
import os
import torch
from scipy import sparse
import itertools
from torch_geometric.data import Data, DataLoader, Dataset
import config
import scipy
import random

path = '/GNN_NPY_DATASETS/SEED/data_dependent'

def fre_stack(data, pseudo_label):
    data = data.T
    P_hat = np.array(list(itertools.permutations(list(range(5)), 5)))  # Full Permutation
    selected = P_hat[pseudo_label]

    ret = np.vstack((data[selected[0]], data[selected[1]]))
    for i in range(2, len(selected)):
        ret = np.vstack((ret, data[selected[i]]))

    return ret.T

def spa_stack(data, pseudo_label):
    k_permutations = np.load('max_hamming_set_10_128.npy')
    selected = k_permutations[pseudo_label]

    EEG_dic = {}
    EEG_dic[0] = data[:5]
    EEG_dic[1] = np.vstack((data[5:8], data[14:17]))
    EEG_dic[2] = np.vstack((data[23:26], data[32:35]))
    EEG_dic[3] = np.vstack((np.vstack((data[41:44], data[50:52])), data[60]))
    EEG_dic[4] = np.vstack((data[8:11], np.vstack((data[17:20], data[26:29]))))
    EEG_dic[5] = np.vstack((data[35:38], data[44:47]))
    EEG_dic[6] = np.vstack((data[52:55], data[57:60]))
    EEG_dic[7] = np.vstack((data[11:14], data[20:23]))
    EEG_dic[8] = np.vstack((data[29:32], data[38:41]))
    EEG_dic[9] = np.vstack((np.vstack((data[47:50], data[55:57])), data[61]))

    ret = np.vstack((EEG_dic[selected[0]], EEG_dic[selected[1]]))
    for i in range(2, len(selected)):
        ret = np.vstack((ret, EEG_dic[selected[i]]))

    return ret

def adjacency():
    row_ = np.array(
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
         13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
         27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
         41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
         54, 55, 55, 56, 57, 58, 59,
         60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
         20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
         34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
         40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
         54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])

    col_ = np.array(
        [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
         13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
         27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
         48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
         59, 55, 60, 56, 61, 61, 58,
         59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
         11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
         25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
         39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
         53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])

    # te_r = np.array([1,2,3])
    # tr_c = np.array([4,5,6])
    # data_ = np.ones(3).astype('float32')
    # B = scipy.sparse.csr_matrix((data_, (row_, col_)), shape=(62, 62))

    weight_ = np.ones(236).astype('float32')
    A = scipy.sparse.csr_matrix((weight_, (row_, col_)), shape=(62, 62))
    
    return row_, col_, weight_

def data_reader(index):
    train_data = np.load(path + '/' + 'train_dataset_{}.npy'.format(index))
    train_label = np.load(path + '/' + 'train_labelset_{}.npy'.format(index))
    test_data = np.load(path + '/' + 'test_dataset_{}.npy'.format(index))
    test_label = np.load(path + '/' + 'test_labelset_{}.npy'.format(index))

    return train_data, train_label, test_data, test_label

def create_graph(data, label, shuffle=False, batch_size=100, drop_last=True):
    row_, col_, weight_ = adjacency()
    edge_index = torch.from_numpy(np.vstack((row_, col_))).long()
    edge_attr = weight_
    edge_attr = torch.from_numpy(edge_attr)
    graph = []

    for i in range(data.shape[0]):
        x = data[i]
        x = torch.from_numpy(x).type(torch.float32)
        
        y = torch.tensor(label[i], dtype=torch.long)
        graph.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return DataLoader(graph, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=config.num_workers)

def create_jigsaw(stack, data, jigsaw_parts=120, shuffle=False, batch_size=100, drop_last=True, num_jigsaw=1):
    row_, col_, weight_ = adjacency()
    edge_index = torch.from_numpy(np.vstack((row_, col_))).long()
    edge_attr = weight_
    edge_attr = torch.from_numpy(edge_attr)
    graph = []
    
    for i in range(data.shape[0]):
        for j in range(num_jigsaw):
            pseudo = np.random.randint(0, jigsaw_parts)
            x = stack(data[i], pseudo_label=pseudo)         
            y = pseudo
            x = torch.from_numpy(x).type(torch.float32)
            y = torch.tensor(y)

            graph.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return DataLoader(graph, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=config.num_workers)

def create_contrastive(fstack, sstack, data, timeseed, fjigsaw_parts=120, sjigsaw_parts=128, shuffle=True, batch_size=100, drop_last=True, num_jigsaw=1):
    '''
    combine fre & spa data agumentation
    '''
    row_, col_, weight_ = adjacency()
    edge_index = torch.from_numpy(np.vstack((row_, col_))).long()
    edge_attr = weight_
    edge_attr = torch.from_numpy(edge_attr)
    graph = []
    
    if shuffle:
        random.seed(timeseed)
        random.shuffle(data)
    for i in range(data.shape[0]):
        for j in range(num_jigsaw):
            spseudo = np.random.randint(0, sjigsaw_parts)
            x = sstack(data[i], pseudo_label=spseudo)
            fpseudo = np.random.randint(0, fjigsaw_parts)
            x = fstack(x, pseudo_label=fpseudo)

            x = torch.from_numpy(x).type(torch.float32)

            y = i   # this data don't need label
            graph.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return DataLoader(graph, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=config.num_workers)
