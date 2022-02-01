import numpy as np
import torch
import math

from torch.utils.data import TensorDataset

def split_by_polygon_id(dataset, perc, return_sums=False):
    classes = np.unique(dataset[:][1])

    train_indices = []
    val_indices = []
    ctrain_sizes = []
    cval_sizes = []
    for cc in classes:
        cind = torch.nonzero(dataset[:][1] == cc)
        ctensors = dataset[cind]
        cpolys = torch.unique(ctensors[2])
        cnum_polys = len(cpolys)
        ctrain_poly_num = math.ceil(cnum_polys * perc)
        cindices = cpolys[torch.randperm(len(cpolys))]
        ctrain_polys = cindices[:ctrain_poly_num]

        ctrain_indices = torch.any(
            ctensors[2] == torch.unsqueeze(ctrain_polys, 0), axis=1
        )
        cval_indices = torch.logical_not(ctrain_indices)
        train_indices.append(cind[ctrain_indices])
        val_indices.append(cind[cval_indices])
        if return_sums:
            ctrain_sizes.append(torch.sum(ctrain_indices))
            cval_sizes.append(torch.sum(cval_indices))

    train_indices = torch.squeeze(torch.cat(train_indices, dim=0))
    val_indices = torch.squeeze(torch.cat(val_indices, dim=0))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    if return_sums:
        return train_dataset, val_dataset, torch.Tensor(ctrain_sizes), torch.Tensor(cval_sizes)

    return train_dataset, val_dataset