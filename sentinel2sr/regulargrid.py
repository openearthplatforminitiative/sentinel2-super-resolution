#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
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
This module contains function for the reading and resampling of
regular gridded data such as images, through rasterio
"""

from typing import List, Optional, Tuple

import numpy as np
import rasterio as rio
from affine import Affine  # type: ignore
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


def create_warped_vrt(filename: str,
                      resolution: float,
                      dst_bounds: Optional[BoundingBox] = None,
                      dst_crs: Optional[str] = None,
                      src_nodata: Optional[float] = None,
                      nodata: Optional[float] = None,
                      shifts: Optional[Tuple[float, float]] = None,
                      resampling: Resampling = Resampling.cubic,
                      dtype=None) -> WarpedVRT:
    """
    Create a warped vrt from filename, to change srs and resolution

    :param filename: Name of the image file
    :param resolution: Target resolution
    :param dst_bounds: Target bounds
    :param dst_crs: Target crs
    :param src_nodata: Value for missing data in source image
    :param nodata: Exposed value for missing data in VRT
    :param shifts: Shifts to apply to src origin for registration purposes
    :param resampling: Resampling method

    :return: A WarpedVRT object
    """

    with rio.open(filename) as src:
        target_bounds = None
        target_crs = src.crs
        if dst_crs is not None:
            target_crs = dst_crs
        if dst_bounds is not None:
            target_bounds = dst_bounds
        else:
            if target_crs != src.crs:
                target_bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
            else:
                target_bounds = src.bounds

        src_transform = src.transform
        if shifts is not None:
            src_res = src_transform[0]
            src_transform = Affine(src_res, 0.0, src_transform[2] - shifts[0], 0.0, -src_res,
                                   src_transform[5] - shifts[1])

        # Compute optimized transform wrt. resolution and new crs
        left, bottom, right, top = target_bounds
        dst_width = int(np.floor((right - left) / resolution))
        dst_height = int(np.floor((top - bottom) / resolution))
        dst_transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

        vrt_options = {
            'resampling': resampling,
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            'crs': target_crs,
            'src_transform': src_transform
        }
        if src_nodata is not None:
            vrt_options['src_nodata'] = src_nodata
        elif src.nodata is not None:
            vrt_options['src_nodata'] = src.nodata
        if nodata is not None:
            vrt_options['nodata'] = nodata

        if dtype is not None:
            vrt_options['dtype'] = dtype

        vrt = WarpedVRT(src, **vrt_options)

        return vrt


def read_as_numpy(img_files: List[str],
                  crs: Optional[str] = None,
                  resolution: float = 10,
                  offsets: Optional[Tuple[float, float]] = None,
                  input_no_data_value: Optional[float] = None,
                  output_no_data_value: float = np.nan,
                  bounds: Optional[rio.coords.BoundingBox] = None,
                  algorithm=rio.enums.Resampling.cubic,
                  separate: bool = False,
                  dtype=np.float32,
                  scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    :param vrts: A list of WarpedVRT objects to stack
    :param dtype: dtype of the output Tensor
    :param separate: If True, each WarpedVRT is considered to offer a single band
    TODO
    """
    assert len(img_files) > 0

    # Check if we need resampling or not
    need_warped_vrt = offsets is not None
    # If we change image bounds
    nb_bands = None
    out_crs = crs

    if out_crs is None:
        out_crs = rio.open(img_files[0]).crs

    for img_file in img_files:
        with rio.open(img_file) as rio_dataset:
            if nb_bands is None:
                nb_bands = rio_dataset.count
            else:
                if nb_bands != rio_dataset.count:
                    raise ValueError("All image files need to have the same number of bands")
            if bounds is not None and rio_dataset.bounds != bounds:
                need_warped_vrt = True
                out_bounds = rio.coords.BoundingBox(*bounds)
            else:
                out_bounds = rio.coords.BoundingBox(*rio_dataset.bounds)
            # If we change projection
            if out_crs != rio_dataset.crs:
                need_warped_vrt = True
            if rio_dataset.transform[0] != resolution:
                need_warped_vrt = True

    # If warped vrts are needed, create them
    if need_warped_vrt:
        datasets = [
            create_warped_vrt(f,
                              resolution,
                              dst_bounds=out_bounds,
                              dst_crs=out_crs,
                              nodata=input_no_data_value,
                              src_nodata=input_no_data_value,
                              resampling=algorithm,
                              shifts=offsets) for f in img_files
        ]

    else:
        datasets = [rio.open(f, 'r') for f in img_files]

    axis = 0
    # if vrts are bands of the same image
    if separate:
        axis = 1

    np_stack: np.ndarray = np.stack([ds.read(masked=True) for ds in datasets], axis=axis)

    # Close datasets
    for rio_dataset in datasets:
        rio_dataset.close()

    # If scaling is required, apply it
    if scale is not None:
        np_stack_mask = np_stack == input_no_data_value
        np_stack = np_stack / scale
        np_stack[np_stack_mask] = output_no_data_value

    # Convert to float before casting to final dtype
    np_stack = np_stack.astype(dtype)

    xcoords: np.ndarray = np.linspace(out_bounds.left + 0.5 * resolution,
                                      out_bounds.right - 0.5 * resolution, np_stack.shape[3])

    ycoords: np.ndarray = np.linspace(out_bounds.top - 0.5 * resolution,
                                      out_bounds.bottom + 0.5 * resolution, np_stack.shape[2])

    return np_stack, xcoords, ycoords, out_crs
