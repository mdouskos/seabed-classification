import numpy as np
import torch
import math

from torch.utils.data import TensorDataset


def positional_encoding(data, pos, d_emb, sigma, div_term=None):
    # if type(sigma) is float:
    #     sigma = np.tile(sigma, len(pos))
    if div_term is None:
        # d_emb_fact = 1 / (d_emb // 2)
        # div_term = np.expand_dims(sigma, 1) * np.expand_dims(
        #     np.arange(0, d_emb // 2) * d_emb_fact, 0
        # )
        # div_term = (
        #     2 * np.pi * sigma * np.tile(np.arange(0, d_emb // 2) * d_emb_fact, (2, 1))
        # )
        div_term = np.random.randn(2, d_emb // 2) * sigma
    p_emb_kern = np.einsum("ij,ik->jk", pos, div_term)

    # out_data = np.concatenate([np.sin(p_emb_kern), np.cos(p_emb_kern), data], 1)
    out_data = data + np.sin(p_emb_kern) + np.cos(p_emb_kern)
    return out_data, div_term


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
        if cnum_polys == ctrain_poly_num:
            ctrain_poly_num = cnum_polys - 1
        # cindices = cpolys[torch.randperm(len(cpolys))]
        cindices = cpolys[:]
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
        return (
            train_dataset,
            val_dataset,
            torch.Tensor(ctrain_sizes),
            torch.Tensor(cval_sizes),
        )

    return train_dataset, val_dataset