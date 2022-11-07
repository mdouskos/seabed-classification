from typing import Union, Type, Literal
from pathlib import Path
from copy import deepcopy

import logging

import rioxarray

import json
import numpy as np

from skimage.morphology import dilation, disk

import pandas


from utils.dataset import positional_encoding

from scipy.ndimage import uniform_filter, convolve1d

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SurveyData:
    def __init__(
        self,
        data_description: Union[str, Type[Path], dict],
        dataset_dir: Union[str, Type[Path]] = None,
        mode: str = "bs",
    ):
        if type(data_description) == str:
            data_description = Path(data_description)

        if isinstance(data_description, Path):
            assert (
                data_description.suffix == ".json"
            ), "Invalid input dataset description"
            # Load description from file
            with open(data_description) as f:
                data_description = json.load(f)

        self.dataset_dir = dataset_dir
        self.data_description = data_description
        self.mode = mode
        self.__read_all_data()

    def plot(self):
        plt.imshow(self.backscatter.data.transpose(1, 2, 0).astype(np.uint8))
        if hasattr(self, "bathymetry"):
            bathy_show = self.bathymetry.data
            bathy_show[~self.bathymetry_mask] = np.nan
            plt.imshow(bathy_show, alpha=0.5)
        gt_show = self.groundtruth.data.astype(np.float32)
        gt_show[~self.groundtruth_mask] = np.nan
        plt.imshow(gt_show, cmap="Set1")
        plt.show()

    def get_data(
        self,
        normalize: Literal["none", "minmax", "std"] = "none",
        pe_dim: int = 0,
        pe_sigma: float = 1.0,
        bpi_radius: int = 17,
        dilation_radius: int = 10,
        morph_element: Literal["square", "disk"] = "square",
    ):
        data_out = dict()

        if "samples" in self.data_description:
            assert hasattr(self, "ids_cls_map"), "Polygon ID to class map missing"
            assert hasattr(self, "polygons"), "Polygon ID raster is missing"
            self.groundtruth = deepcopy(self.polygons)
            self.groundtruth.data = np.zeros_like(self.polygons.data)
            # expand gt regions
            if morph_element == "disk":
                self.polygons.data = dilation(self.polygons.data, disk(dilation_radius))
            elif morph_element == "square":
                self.polygons.data = convolve1d(
                    convolve1d(
                        self.polygons.data,
                        np.ones((2 * dilation_radius)),
                        mode="constant",
                    ).T,
                    np.ones((2 * dilation_radius)),
                    mode="constant",
                ).T
            else:
                raise ValueError(f"Morph element {morph_element} not supported")

            for polyid, clsid in self.ids_cls_map.T:
                self.groundtruth.data[self.polygons.data == polyid] = clsid
            self.groundtruth_mask = self.groundtruth.data != self.groundtruth.rio.nodata

        mask_valid_gt = np.logical_and(self.backscatter_mask, self.groundtruth_mask)
        data_pixels = self.backscatter.data.reshape((3, -1))[
            :, self.backscatter_mask.flatten()
        ].transpose()
        mask_gt_seq = self.groundtruth_mask.flatten()[self.backscatter_mask.flatten()]
        # data_valid_gt = data_pixels[mask_gt_seq]

        if hasattr(self, "bathymetry"):
            if self.mode == "bathy":
                bathydata = self.bathymetry.data
            elif self.mode == "bpi":
                bathydata = (
                    self.bathymetry.data
                    - uniform_filter(self.bathymetry.data, bpi_radius, mode="constant")
                ).astype(np.int)
            data_pixels = np.append(
                data_pixels,
                bathydata.flatten()[self.backscatter_mask.flatten(), np.newaxis],
                axis=1,
            )

        if pe_dim > 0:
            logger.info(
                f"Using positional embedding with dim {pe_dim} and sigma {pe_sigma}"
            )
            pos_data = np.where(self.backscatter_mask)
            # sigma_vec = sigma/np.array(raster_shape)
            # pos_vec = pos_data/np.expand_dims(raster_shape, 1)
            data_pixels, div_term = positional_encoding(
                data_pixels, pos_data, pe_dim, pe_sigma
            )
            self.div_term = div_term

        data_out["name"] = self.data_description["name"]
        data_out["data"] = data_pixels[mask_gt_seq, :]
        data_out["data_all"] = data_pixels
        data_out["gt"] = (
            self.groundtruth.data[
                np.logical_and(self.backscatter_mask, self.groundtruth_mask)
            ]
            - 1
        )
        data_out["polygons"] = self.polygons.data[mask_valid_gt]
        data_out["backscatter"] = self.backscatter
        data_out["backscatter_mask"] = self.backscatter_mask
        return data_out

    def __read_all_data(self):
        self.__read_raster("backscatter", dtype=np.uint8)
        if "samples" in self.data_description:
            self.__read_samples(dilation_r=3)
        elif "groundtruth" in self.data_description:
            self.__read_gt()

        if self.mode in ("bathy", "bpi"):
            self.__read_raster("bathymetry", n_bands=1)

    def __read_gt(self):
        self.__read_raster("groundtruth", n_bands=1, dtype=np.uint8)
        self.__read_raster("polygons", n_bands=1, dtype=np.uint8)

        assert np.all(
            self.groundtruth_mask == self.polygons_mask
        ), "GT mask does not correspond to polygon IDs mask"

    def __read_raster(self, key, n_bands=3, dtype=None):
        assert key in self.data_description, f"Unknown key: {key}"
        raster_file = self.data_description[key]
        assert Path(raster_file).suffix == ".tif"
        logger.info("Reading " + raster_file + "...")
        raster_file = (
            self.dataset_dir
            / Path(self.data_description["directory"])
            / self.data_description[key]
        )
        raster = rioxarray.open_rasterio(raster_file)
        if n_bands == 1:
            raster = raster.isel(band=0)
        if dtype and raster.data.dtype != dtype:
            raster.data = raster.data.astype(dtype)
        if n_bands > 1:
            mask = np.all(raster.data != raster.rio.nodata, axis=0)
        else:
            mask = raster.data != raster.rio.nodata
        logger.info("Done!")
        self.__setattr__(key, raster)
        self.__setattr__(key + "_mask", mask)

    def __read_samples(
        self,
        key: str = "samples",
        dilation_r: int = 10,
        Xfld: str = "X",
        Yfld: str = "Y",
        CIDfld: str = "Class ID",
    ):
        assert hasattr(self, "backscatter")
        samples_file = self.data_description[key]
        assert Path(samples_file).suffix in (".csv", ".txt")
        samples_file = (
            self.dataset_dir / Path(self.data_description["directory"]) / samples_file
        )
        logger.info("Reading sample file...")
        # build single-band dataset based on input data
        gt_raster = self.backscatter.isel(band=0)
        gt_raster.data = np.zeros_like(gt_raster.data)
        gt_data_shape = gt_raster.data.shape

        # read sample points and corresponding classes from CSV file
        gt_df = pandas.read_csv(samples_file)

        self.n_samples = len(gt_df)
        # store sample classes
        gt_cls = gt_df[[CIDfld]].to_numpy()[:, 0]
        gt_ids = np.arange(1, len(gt_cls) + 1)

        # make coordinates homogeneous
        gt_coords = gt_df[[Xfld, Yfld]].to_numpy()
        gt_coords = np.hstack((gt_coords, np.ones((len(gt_df), 1))))

        # prepare inverse transfromation mapping sample points to pixel coords
        T = gt_raster.rio.transform()
        Tnp = np.array(T).reshape(3, 3)
        Tnp_inv = np.linalg.inv(Tnp)

        # compute and filter pixel coordinates of samples
        gt_indx = (Tnp_inv @ gt_coords.T)[:2, :].astype(dtype=np.int16)
        mask_gt_ind = (
            np.all(gt_indx >= 0, 0)
            * (gt_indx[0, :] < gt_data_shape[1])
            * (gt_indx[1, :] < gt_data_shape[0])
        )
        gt_indx = gt_indx[..., mask_gt_ind]

        # assign classes to the data array
        gt_indx_flat = np.ravel_multi_index(
            [gt_indx[1, :], gt_indx[0, :]], gt_data_shape
        )
        # np.put(gt_data, gt_indx_flat, gt_cls[mask_gt_ind])
        np.put(gt_raster.data, gt_indx_flat, gt_ids[mask_gt_ind])

        logger.info("Done!")

        self.polygons = gt_raster
        self.ids_cls_map = np.vstack((gt_ids[mask_gt_ind], gt_cls[mask_gt_ind]))
