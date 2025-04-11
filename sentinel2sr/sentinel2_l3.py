#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
# Copyright: (c) 2025 Open Earth Platform Initiative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Driver for Sentinel2 L3 Mosaics
"""

import os
from enum import Enum
from typing import List, Optional, Tuple
from sensorsio import regulargrid
from regulargrid import read_as_numpy

import numpy as np
import rasterio as rio


class Sentinel2L3:
    """
    Class for Sentinel2 L3 Mosaics
    """

    def __init__(
        self,
        product_path: str,
        year: int,
        quartile: str,
        offsets: Optional[Tuple[float, float]] = None
    ):
        """
        Constructor

        :param product_path: Path to product
        :param offsets: Shifts applied to image orgin (as computed by StackReg for instance)
        """
        # Store product DIR
        self.product_dir = os.path.abspath(os.path.join(product_path, os.pardir))
        self.product_name = os.path.basename(product_path)

        # Strip zip extension if exists
        if self.product_name.endswith(".tif") or self.product_name.endswith(".tiff"):
            self.product_name = self.product_name[:-4]

        # Store offsets
        self.offsets = offsets
        self.radiometric_offset = 0

        # Get
        self.satellite = Sentinel2L3.Satellite("SENTINEL2")

        # Get tile
        self.tile = self.product_name

        # Get acquisition date
        self.year = year
        self.quartile = quartile

        with rio.open(self.build_band_path(Sentinel2L3.B2)) as dataset:
            # Get bounds
            self.bounds = dataset.bounds
            self.transform = dataset.transform
            # Get crs
            self.crs = dataset.crs

    def __repr__(self):
        return f"{self.satellite}, {self.year}, {self.quartile}, {self.tile}"

    class Satellite(Enum):
        """
        Enum class representing Sentinel2 satellite id
        """

        S2A = "SENTINEL2A"
        S2B = "SENTINEL2B"
        S2 = "SENTINEL2"

    # Aliases
    S2A = Satellite.S2A
    S2B = Satellite.S2B

    class Band(Enum):
        """
        Enum class representing Sentinel2 spectral bands
        """

        B1 = "B01"
        B2 = "B02"
        B3 = "B03"
        B4 = "B04"
        B5 = "B05"
        B6 = "B06"
        B7 = "B07"
        B8 = "B08"
        B8A = "B8A"
        B9 = "B09"
        B10 = "B10"
        B11 = "B11"
        B12 = "B12"
        VIRTUAL = "VIRTUAL"

    # Aliases
    B1 = Band.B1
    B2 = Band.B2
    B3 = Band.B3
    B4 = Band.B4
    B5 = Band.B5
    B6 = Band.B6
    B7 = Band.B7
    B8 = Band.B8
    B8A = Band.B8A
    B9 = Band.B9
    B10 = Band.B10
    B11 = Band.B11
    B12 = Band.B12
    VIRTUAL_BAND = Band.VIRTUAL

    class Mask(Enum):
        """
        Enum class for Sentinel2 L2A masks
        """

        CLASSI = "CLASSI_B00"

    # Band groups
    GROUP_10M = [B2, B3, B4, B8]
    GROUP_VIRTUAL = [B1, B5, B6, B7, B8A, B9, B10, B11, B12]
    ALL_MASKS = [Mask.CLASSI]

    # Resolution
    RES = {
        B2: 10,
        B3: 10,
        B4: 10,
        B8: 10,
    }

    def build_band_path(self, band: Band) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        band_path = os.path.join(self.product_dir, f"{self.product_name}.tif")

        # Raise
        if not os.path.exists(band_path):
            raise FileNotFoundError(
                f"Could not find band {band.value} in directory {self.product_dir}"
            )
        return band_path

    def read_as_numpy(
        self,
        bands: List[Band],
        scale: float = 10000,
        crs: Optional[str] = None,
        resolution: float = 10,
        no_data_value: float = np.nan,
        bounds: Optional[rio.coords.BoundingBox] = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype("float32"),
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, str,]:
        """Read bands from Sentinel2 products as a numpy
        ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.
        :param bands: The list of bands to read
        :param scale: Scale factor applied to reflectances (r_s = r /
        scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the
        resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a
        list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from
        image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor

        :return: The image pixels as a np.ndarray of shape [bands,
        width, height],

        The masks pixels as a np.ndarray of shape [masks, width,
        height],
        The x coords as a np.ndarray of shape [width],
        the y coords as a np.ndarray of shape [height],
        the crs as a string
        """

        if len(bands):

            img_files = [self.build_band_path(self.B2)]
            np_arr, xcoords, ycoords, out_crs = read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                offsets=self.offsets,
                output_no_data_value=no_data_value,
                input_no_data_value=-32768,
                bounds=bounds,
                algorithm=algorithm,
                separate=True,
                dtype=dtype,
                scale=scale,
            )

            # Skip first dimension
            np_arr = np.swapaxes(np_arr, 0, 1)
            np_arr = np_arr[0, ...]

            # Handle no-data values
            np_arr[np_arr == -32768] = 1

            # Add virtual bands
            # TODO: fix proper virtual band handling
            if any(b in Sentinel2L3.GROUP_VIRTUAL for b in bands):
                virtual_bands = np.zeros((6, np_arr.shape[1], np_arr.shape[2]))
                np_arr = np.concatenate((np_arr, virtual_bands), axis=0)
                #bands_arr = np.zeros((len(bands), np_arr.shape[2], np_arr.shape[3]))
                #for idx, b in enumerate(bands):
                #    if b not in Sentinel2L3.GROUP_VIRTUAL:
                #        np.insert(bands_arr, idx, np_arr[], axis=0)
                #np_arr = bands_arr

            # Apply radiometric offset
            np_arr = (np_arr + self.radiometric_offset) / scale


        # Return plain numpy array
        return np_arr, xcoords, ycoords, out_crs
