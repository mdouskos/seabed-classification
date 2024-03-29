import numpy as np
import math


def normalize_data(X, X_ref=None, normalization_type="std"):
    if X_ref is None:
        X_ref = X
    if normalization_type == "std":
        X_norm = (X - np.mean(X_ref, axis=0)) / np.std(X_ref, axis=0)
    elif normalization_type == "minmax":
        X_norm = (X - np.min(X_ref, axis=0)) / (
            np.max(X_ref, axis=0) - np.min(X_ref, axis=0)
        )
    elif normalization_type == "range":
        X_norm = X / 255
    else:
        raise ValueError(f"Unrecognlized normalization type {normalization_type}")
    return X_norm


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    # oldshape = source.shape
    # source = source.ravel()
    # template = template.ravel()

    result = np.zeros_like(source)

    for dim in range(source.shape[1]):
        # get the set of unique pixel values and their corresponding indices and
        # counts
        _, bin_idx, s_counts = np.unique(
            source[:, dim], return_inverse=True, return_counts=True
        )
        t_values, t_counts = np.unique(template[:, dim], return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        result[:, dim] = interp_t_values[bin_idx]  # .reshape(oldshape)

    return result


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


def split_by_polygon_id(X, y, polys, perc, shuffle=True, return_sums=False):
    classes = np.unique(y)

    train_indices = []
    val_indices = []
    ctrain_sizes = []
    cval_sizes = []
    for cc in classes:
        cind = np.nonzero(y == cc)[0]
        cpolys = np.unique(polys[cind])
        cnum_polys = len(cpolys)
        ctrain_poly_num = math.ceil(cnum_polys * perc)
        if cnum_polys == ctrain_poly_num:
            ctrain_poly_num = cnum_polys - 1
        if shuffle:
            cindices = cpolys[np.random.permutation(len(cpolys))]
        else:
            cindices = cpolys[:]
        ctrain_polys = cindices[:ctrain_poly_num]

        ctrain_indices = np.any(
            polys[cind, np.newaxis] == ctrain_polys[np.newaxis, :], axis=1
        )
        cval_indices = np.logical_not(ctrain_indices)
        train_indices.append(cind[ctrain_indices])
        val_indices.append(cind[cval_indices])
        if return_sums:
            ctrain_sizes.append(np.sum(ctrain_indices))
            cval_sizes.append(np.sum(cval_indices))

    train_indices = np.squeeze(np.concatenate(train_indices))
    val_indices = np.squeeze(np.concatenate(val_indices))

    training_data = [X[train_indices], y[train_indices], polys[train_indices]]
    validation_data = [X[val_indices], y[val_indices], polys[val_indices]]

    if return_sums:
        return (
            training_data,
            validation_data,
            np.array(ctrain_sizes),
            np.array(cval_sizes),
        )

    return training_data, validation_data
