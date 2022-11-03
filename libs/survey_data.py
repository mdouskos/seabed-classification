from typing import Union, Type
from pathlib import Path

import logging

import rioxarray

import json
import numpy as np

from skimage.morphology import dilation, disk

import pandas

# import matplotlib.pyplot as plt

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

    def get_data(self, normalize=True):
        data_out = dict()
        mask_valid_gt = np.logical_and(self.backscatter_mask, self.groundtruth_mask)
        data_pixels = self.backscatter.data.reshape((3, -1))[
            :, mask_valid_gt.flatten()
        ].transpose()

        if hasattr(self, "polygons"):
            data_out["polygons"] = self.polygons.data[mask_valid_gt]
        # if bathymetry_file:
        #     data_pixels = np.append(
        #         data_pixels,
        #         bathy_raster.data.flatten()[mask_train.flatten(), np.newaxis],
        #         # BPI.flatten()[mask_train.flatten(), np.newaxis],
        #         axis=1,
        #     )
        if normalize:
            data_pixels_n = (data_pixels - np.mean(data_pixels, axis=0)) / np.std(
                data_pixels, axis=0
            )
            # data_pixels_n = (data_pixels - np.min(data_pixels, axis=0)) / (
            #     np.max(data_pixels, axis=0) - np.min(data_pixels, axis=0)
            # )
        else:
            data_pixels_n = data_pixels

        data_out["data"] = data_pixels_n
        data_out["gt"] = self.groundtruth.data[mask_valid_gt] - 1
        data_out["data_raster"] = self.backscatter
        data_out["mask_data"] = self.backscatter_mask
        data_out["input_shape"] = self.backscatter.shape[1:]
        data_out["gt_raster"] = self.groundtruth
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

        # read sample points and corresponding classes from CSV file
        gt_df = pandas.read_csv(samples_file)

        self.n_samples = len(gt_df)
        # store sample classes
        gt_cls = gt_df[[CIDfld]].to_numpy()
        gt_ids = np.arange(1, len(gt_cls) + 1)

        # make coordinates homogeneous
        gt_coords = gt_df[[Xfld, Yfld]].to_numpy()
        gt_coords = np.hstack((gt_coords, np.ones((len(gt_df), 1))))

        # initialize gt data array
        gt_data = np.zeros_like(gt_raster.data)

        # prepare inverse transfromation mapping sample points to pixel coords
        T = gt_raster.rio.transform()
        Tnp = np.array(T).reshape(3, 3)
        Tnp_inv = np.linalg.inv(Tnp)

        # compute and filter pixel coordinates of samples
        gt_indx = (Tnp_inv @ gt_coords.T)[:2, :].astype(dtype=np.int16)
        mask_gt_ind = (
            np.all(gt_indx >= 0, 0)
            * (gt_indx[0, :] < gt_data.shape[1])
            * (gt_indx[1, :] < gt_data.shape[0])
        )
        gt_indx = gt_indx[..., mask_gt_ind]

        # assign classes to the data array
        gt_indx_flat = np.ravel_multi_index(
            [gt_indx[1, :], gt_indx[0, :]], gt_data.shape
        )
        # np.put(gt_data, gt_indx_flat, gt_cls[mask_gt_ind])
        np.put(gt_data, gt_indx_flat, gt_ids[mask_gt_ind])

        # expand gt regions
        gt_raster.data = dilation(gt_data, disk(dilation_r))

        groundtruth_mask = gt_raster.data != gt_raster.rio.nodata
        logger.info("Done!")

        self.groundtruth = gt_raster
        self.groundtruth_mask = groundtruth_mask