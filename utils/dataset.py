import pathlib 

import numpy as np
import torch
import math

import scipy

import rioxarray

import matplotlib.pyplot as plt

from skimage.morphology import dilation, disk

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

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

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

    return interp_t_values[bin_idx]  # .reshape(oldshape)

def read_csv(datafile):
    ifile = open(datafile, 'r')
    data_all = []
    ifile.readline()
    for line in ifile:
        file_data = line.strip().split(',')
        data_all.append(file_data)
    return np.array(data_all, dtype=np.float)
    

def process_data(
    input_file,
    gt_file,
    polygons_file=None,
    bathymetry_file=None,
    normalize=False,
    plot=False,
):
    data_dict = dict.fromkeys(
        [
            "data",
            "gt",
            "data_raster",
            "gt_raster",
            "bathy_raster",
            "mask_gt",
            "mask_data",
            "polygons",
            "input_shape",
        ]
    )

    print("Reading data...", end=" ")
    input_raster = rioxarray.open_rasterio(input_file)
    data_dict["data_raster"] = input_raster
    if input_raster.data.dtype != np.uint8:
        input_raster.data = input_raster.data.astype(np.uint8)
    mask_input = np.all(input_raster.data != input_raster.rio.nodata, axis=0)
    mask_valid = mask_input
    data_dict["input_shape"] = input_raster.data.shape[1:]
    print("done!")

    print("Reading ground truth...", end=" ")
    file_ext = pathlib.Path(gt_file).suffix
    if file_ext == '.tif':
        gt_raster = rioxarray.open_rasterio(gt_file).isel(band=0)
        # gt_raster.data = dilation(gt_raster.data, ball(9))
    elif file_ext == ".csv" or file_ext == ".txt":
        # build single-band dataset based on input data
        gt_raster = input_raster.isel(band=0)

        # read sample points and corresponding classes from CSV file
        gt_coords = read_csv(gt_file)

        # store sample classes
        gt_cls = gt_coords[:, -1].copy()

        # make coordinates homogeneous
        gt_coords[:, -1] = 1

        # initialize gt data array
        gt_data = np.zeros_like(input_raster)[0]

        # prepare inverse transfromation mapping sample points to pixel coords
        T = input_raster.rio.transform()
        Tnp = np.array(T).reshape(3, 3)
        Tnp_inv = np.linalg.inv(Tnp)
        
        # compute and filter pixel coordinates of samples
        gt_indx = (Tnp_inv@gt_coords.T)[:2, :].astype(dtype=np.int16)
        mask_gt_ind = np.all(gt_indx >= 0, 0) * \
            (gt_indx[0, :] < gt_data.shape[1]) * (gt_indx[1, :] < gt_data.shape[0])
        gt_indx = gt_indx[..., mask_gt_ind]

        # assign classes to the data array
        gt_indx_flat = np.ravel_multi_index([gt_indx[1, :], gt_indx[0, :]], gt_data.shape)
        np.put(gt_data, gt_indx_flat, gt_cls[mask_gt_ind])

        # expand gt regions
        gt_raster.data = dilation(gt_data, disk(10))

    
    data_dict["gt_raster"] = gt_raster
    if gt_raster.data.dtype != np.uint8:
        gt_raster.data = gt_raster.data.astype(np.uint8)
    mask_gt = gt_raster.data != gt_raster.rio.nodata
    mask_train = mask_gt.copy()
    # num_classes = np.unique(gt_raster.data).size - 1
    print("done!")

    mask_train = np.logical_and(mask_train, mask_input)

    if polygons_file:
        print("Reading polygon IDs...", end=" ")
        poly_raster = rioxarray.open_rasterio(polygons_file).isel(band=0)
        if poly_raster.data.dtype != np.uint8:
            poly_raster.data = poly_raster.data.astype(np.uint8)
        mask_poly = poly_raster.data != poly_raster.rio.nodata
        assert np.all(
            mask_gt == mask_poly
        ), "GT mask does not correspond to polygon IDs mask"
        print("done!")

    if bathymetry_file:
        print("Reading bathy...", end=" ")
        bathy_raster = rioxarray.open_rasterio(bathymetry_file).isel(band=0)
        data_dict["bathy_raster"] = bathy_raster
        # BPI = (
        #     bathy_raster.data
        #     - scipy.ndimage.uniform_filter(bathy_raster.data, 17, mode="constant")
        # ).astype(np.int)
        mask_bathy = bathy_raster.data != bathy_raster.rio.nodata
        mask_train = np.logical_and(mask_train, mask_bathy)
        mask_valid = np.logical_and(mask_valid, mask_bathy)
        print("done!")

    data_dict["gt"] = gt_raster.data[mask_train] - 1
    data_pixels = input_raster.data.reshape((3, -1))[
        :, mask_train.flatten()
    ].transpose()
    data_dict["mask_gt"] = mask_train
    data_dict["mask_data"] = mask_valid

    if polygons_file:
        data_dict["polygons"] = poly_raster.data[mask_train]
    if bathymetry_file:
        data_pixels = np.append(
            data_pixels,
            bathy_raster.data.flatten()[mask_train.flatten(), np.newaxis],
            # BPI.flatten()[mask_train.flatten(), np.newaxis],
            axis=1,
        )
    if normalize:
        data_pixels_n = (data_pixels - np.mean(data_pixels, axis=0)) / np.std(
            data_pixels, axis=0
        )
        # data_pixels_n = (data_pixels - np.min(data_pixels, axis=0)) / (
        #     np.max(data_pixels, axis=0) - np.min(data_pixels, axis=0)
        # )
    else:
        data_pixels_n = data_pixels

    data_dict["data"] = data_pixels_n

    if plot:
        plt.imshow(input_raster.data.transpose(1, 2, 0).astype(np.uint8))
        if bathymetry_file:
            bathy_show = bathy_raster.data
            bathy_show[~mask_bathy] = np.nan
            plt.imshow(bathy_show, alpha=0.5)
        gt_show = gt_raster.data.astype(np.float32)
        gt_show[~mask_train] = np.nan
        plt.imshow(gt_show, cmap="Set1")
        plt.show()

    return data_dict


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
        cindices = cpolys[torch.randperm(len(cpolys))]
        # cindices = cpolys[:]
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