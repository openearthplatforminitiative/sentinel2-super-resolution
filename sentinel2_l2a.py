#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
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
Driver for Sentinel2 L2A products using sen2cor
"""

import os
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import rasterio as rio
import xarray as xr
from dateutil.parser import parse as parse_date

from sensorsio import regulargrid, storage


class Sentinel2L2A:
    """
    Class for Sentinel2 L2A products with sen2cor
    """

    def __init__(
        self,
        product_dir: str,
        offsets: Optional[Tuple[float, float]] = None,
        parse_xml: bool = True,
        s3_context: Optional[storage.S3Context] = None,
    ):
        """
        Constructor

        :param product_dir: Path to product directory
        :param offsets: Shifts applied to image orgin (as computed by StackReg for instance)
        :param parse_xml: If True (default), parse additional information from xml metadata file
        """
        # Store s3 context
        self.s3_context = s3_context

        # Store product DIR
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        # Strip zip extension if exists
        if self.product_name.endswith(".zip") or self.product_name.endswith(".ZIP"):
            self.product_name = self.product_name[:-4]

        # Strip out the .safe
        self.product_name = self.product_name[:-5]

        # Look for xml file
        self.xml_file = self.build_xml_path()

        # Store offsets
        self.offsets = offsets

        # Get
        self.satellite = Sentinel2L2A.Satellite("SENTINEL" + self.product_name[1:3])

        # Get tile
        self.tile = self.product_name[39:44]

        # Get acquisition date
        self.date = parse_date(self.product_name[11:26])
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday

        with rio.open(self.build_band_path(Sentinel2L2A.B2)) as dataset:
            # Get bounds
            self.bounds = dataset.bounds
            self.transform = dataset.transform
            # Get crs
            self.crs = dataset.crs

        # Parse xml file if requested
        if parse_xml:
            self.parse_xml()

    def __repr__(self):
        return f"{self.satellite.value}, {self.date}, {self.tile}"

    def parse_xml(self):
        """
        Parse metadata file
        """
        with storage.agnostic_open(
            self.product_dir, self.xml_file, self.s3_context
        ) as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # Parse cloud cover
            quality_node = root.find(".//*/Cloud_Coverage_Assessment")
            if quality_node is not None:
                # The value is a percentage in metadata file
                self.cloud_cover = int(float(quality_node.text))
            # Parse orbit number
            orbit_node = root.find(".//*/SENSING_ORBIT_NUMBER")
            if orbit_node is not None:
                # the orbit provide in the metadata file is already the relative orbit number
                self.orbit = int(orbit_node.text)
                self.relative_orbit_number = self.orbit

            # Do we need to apply radiometric offset (latest product format)
            if root.find(".//*/Radiometric_Offset_List") is not None:
                self.radiometric_offset = -1000
            else:
                self.radiometric_offset = 0

    class Satellite(Enum):
        """
        Enum class representing Sentinel2 satellite id
        """

        S2A = "SENTINEL2A"
        S2B = "SENTINEL2B"

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

    class Mask(Enum):
        """
        Enum class for Sentinel2 L2A masks
        """

        CLASSI = "CLASSI_B00"

    # Band groups
    GROUP_10M = [B2, B3, B4, B8]
    GROUP_20M = [B5, B6, B7, B8A, B11, B12]
    GROUP_60M = [B1, B9, B10]
    ALL_MASKS = [Mask.CLASSI]

    # Resolution
    RES = {
        B1: 60,
        B2: 10,
        B3: 10,
        B4: 10,
        B5: 20,
        B6: 20,
        B7: 20,
        B8: 10,
        B8A: 20,
        B9: 60,
        B10: 60,
        B11: 60,
        B12: 60,
    }

    def build_xml_path(self) -> str:
        """
        Return path to root xml file
        """
        xml_path = storage.agnostic_regex(
            self.product_dir,
            "*MTD_MSIL2A.xml",
            s3_context=self.s3_context,
            use_gdal_adressing=False,
        )
        # Raise
        if len(xml_path) == 0:
            raise FileNotFoundError(
                f"Could not find root XML file in product directory {self.product_dir}"
            )
        return xml_path[0]

    def build_band_path(self, band: Band) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        band_path = storage.agnostic_regex(
            self.product_dir,
            f"GRANULE/*/IMG_DATA/R{Sentinel2L2A.RES[band]}m/*_{band.value}*.jp2",
            s3_context=self.s3_context,
            use_gdal_adressing=True,
        )

        # Raise
        if len(band_path) == 0:
            raise FileNotFoundError(
                f"Could not find band {band.value} in directory {self.product_dir}"
            )
        return band_path[0]

    def build_mask_path(self, mask: Mask) -> str:
        """
        Build path to a band for product
        :param mask: The band to build path for as a Sentinel2.Mask enum value


        :return: The path to the band file
        """
        mask_path = storage.agnostic_regex(
            self.product_dir,
            f"GRANULE/*/QI_DATA/MSK_*{mask.value}.jp2",
            s3_context=self.s3_context,
            use_gdal_adressing=True,
        )
        # Raise
        if len(mask_path) == 0:
            raise FileNotFoundError(
                f"Could not find mask \
            {mask.value} in \
            product directory {self.product_dir}"
            )
        return mask_path[0]

    def read_as_numpy(
        self,
        bands: List[Band],
        masks: Optional[List[Mask]] = None,
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
        if masks is None:
            masks = self.ALL_MASKS

        if len(bands):
            img_files = [self.build_band_path(b) for b in bands]
            np_arr, xcoords, ycoords, out_crs = regulargrid.read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                offsets=self.offsets,
                output_no_data_value=no_data_value,
                input_no_data_value=0,
                bounds=bounds,
                algorithm=algorithm,
                separate=True,
                dtype=dtype,
                scale=scale,
            )

            # Skip first dimension
            np_arr = np_arr[0, ...]

            # Apply radiometric offset
            np_arr = (np_arr + self.radiometric_offset) / scale

        # Read masks if needed
        np_arr_msk = None
        if len(masks) != 0:
            mask_files = [self.build_mask_path(m) for m in masks]
            np_arr_msk, _, _, _ = regulargrid.read_as_numpy(
                mask_files,
                crs=crs,
                resolution=resolution,
                offsets=self.offsets,
                output_no_data_value=no_data_value,
                input_no_data_value=-10000,
                bounds=bounds,
                algorithm=rio.enums.Resampling.nearest,
                separate=True,
                dtype=np.uint8,
                scale=None,
            )
            # Skip first dimension
            np_arr_msk = np_arr_msk[0, ...]

        # Return plain numpy array
        return np_arr, np_arr_msk, xcoords, ycoords, out_crs

    def read_as_xarray(
        self,
        bands: List[Band],
        masks: Optional[List[Mask]] = None,
        scale: float = 10000,
        crs: Optional[str] = None,
        resolution: float = 10,
        no_data_value: float = np.nan,
        bounds: Optional[rio.coords.BoundingBox] = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype("float32"),
    ) -> xr.Dataset:
        """Read bands from Sentinel2 products as a numpy

        ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param scale: Scale factor applied to reflectances (r_s = r /
        scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use
        WarpedVRT)
        :param resolution: Resolution of data. If different from the
        resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a
        list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from
        image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height]

        """
        if masks is None:
            masks = self.ALL_MASKS

        np_arr, np_arr_msk, xcoords, ycoords, crs = self.read_as_numpy(
            bands,
            masks,
            scale,
            crs,
            resolution,
            no_data_value,
            bounds,
            algorithm,
            dtype,
        )

        variables = {}
        for i, band in enumerate(bands):
            variables[band.value] = (["t", "y", "x"], np_arr[None, i, ...])
        if np_arr_msk is not None:
            for i, mask in enumerate(masks):
                variables[mask.value] = (["t", "y", "x"], np_arr_msk[None, i, ...])

        xarr = xr.Dataset(
            variables,
            coords={"t": [self.date], "x": xcoords, "y": ycoords},
            attrs={"tile": self.tile, "crs": crs},
        )
        return xarr
