"""
Inference script
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import rasterio as rio  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]
from sensorsio.utils import bb_snap
from sentinel2_l3 import Sentinel2L3
from tqdm import tqdm
from tilefinder.toRGB import toRGB
import yaml


@dataclass(frozen=True)
class Chunk:
    source_area: rio.coords.BoundingBox
    target_area: rio.coords.BoundingBox


@dataclass(frozen=True)
class ModelParameters:
    model: str
    bands: list[str]
    margin: int
    factor: float
    name: str


def read_model_parameters(cfg: str):
    """
    Read yaml file describing model
    """
    with open(cfg, "r") as f:
        cfg_dict = yaml.safe_load(f)

        return ModelParameters(
            model=cfg_dict["model"],
            bands=cfg_dict["bands"],
            margin=cfg_dict["margin"],
            factor=cfg_dict["factor"],
            name=cfg_dict["name"],
        )

def generate_chunks(
    roi: rio.coords.BoundingBox,
    tile_size_in_meters: float,
    margin_in_meters: float,
) -> list[Chunk]:
    """
    Class initializer
    """
    # Find number of chunks in each dimension

    print((roi.right - roi.left) / tile_size_in_meters)

    nb_chunks_x = np.ceil((roi.right - roi.left) / tile_size_in_meters)
    nb_chunks_y = np.ceil((roi.top - roi.bottom) / tile_size_in_meters)

    # Compute upper left corners of chunks
    chunks_x = roi.left + tile_size_in_meters * np.arange(0, nb_chunks_x)
    chunks_y = roi.bottom + tile_size_in_meters * np.arange(0, nb_chunks_y)
    # Generate the 2d grid of chunks upper left center
    chunks_x, chunks_y = np.meshgrid(chunks_x, chunks_y)



    # Flatten both list
    chunks_x = chunks_x.ravel()
    chunks_y = chunks_y.ravel()

    # Generate output chunk list
    chunks: list[Chunk] = []

    for cx, cy in zip(chunks_x, chunks_y):
        # Target area should not exceed roi
        target_area = rio.coords.BoundingBox(
            cx,
            cy,
            min(cx + tile_size_in_meters, roi.right),
            min(cy + tile_size_in_meters, roi.top),
        )
        #print(target_area)
        # Source area is target area padded with margin
        source_area = rio.coords.BoundingBox(
            left=target_area.left - margin_in_meters,
            right=target_area.right + margin_in_meters,
            bottom=target_area.bottom - margin_in_meters,
            top=target_area.top + margin_in_meters,
        )
        chunks.append(Chunk(source_area, target_area))

    return chunks

_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def run(model_yaml,
        input,
        output_dir,
        region_of_interest_pixel=None,
        region_of_interest=None,
        loglevel=logging.INFO,
        tilesize=1000,
        num_threads=8
        ):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    """
    setup_logging(loglevel)
    model_parameters = read_model_parameters(model_yaml)

    s2_ds = Sentinel2L3(input, year=2024, quartile="Q3")
    # Bands that will be processed
    bands = [Sentinel2L3.Band(b) for b in model_parameters.bands]
    level = "_L3_"

    source_resolution = 10.0
    target_resolution = source_resolution / model_parameters.factor

    tile_size_in_meters = target_resolution * tilesize
    margin_in_meters = target_resolution * model_parameters.margin

    _logger.info(f"Will process {s2_ds}")
    _logger.info(f"Bounds: {s2_ds.bounds}, {s2_ds.crs}")
    _logger.info(f"Will use model {model_yaml}")
    _logger.info(
        f"Will process the following bands {bands} at {source_resolution} meter resolution"
    )
    _logger.info(f"Target resolution is {target_resolution} meter")

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    so.use_deterministic_compute = True

    # Execute on gpu only if available
    ep_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Create inference session
    onnx_model_path = os.path.join(Path(model_yaml).parent.resolve(), model_parameters.model)
    ort_session = ort.InferenceSession(onnx_model_path, sess_options=so, providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry("log_severity_level", "3")

    # Read roi
    roi = s2_ds.bounds
    if region_of_interest_pixel is not None:
        logging.info(f"Pixel ROI set, will use it to define target ROI")

        if (
            region_of_interest_pixel[2] <= region_of_interest_pixel[0]
            or region_of_interest_pixel[3] <= region_of_interest_pixel[1]
        ):
            logging.error(
                "Inconsistent coordinates for region_of_interest_pixel parameter:"
                " expected line_start col_start line_end col_end, with line_end > line_start"
                " and col_end > col_start"
            )
            sys.exit(1)
        roi_pixel = rio.coords.BoundingBox(*region_of_interest_pixel)
        roi = rio.coords.BoundingBox(
            left=s2_ds.bounds.left + 10 * roi_pixel.left,
            bottom=s2_ds.bounds.top - 10 * roi_pixel.top,
            right=s2_ds.bounds.left + 10 * roi_pixel.right,
            top=s2_ds.bounds.top - 10 * roi_pixel.bottom,
        )
    elif region_of_interest is not None:
        logging.info(
            f"ROI set, will use it to define target ROI. Note that provided ROI will be snapped to the 10m Sentinel-2 sampling grid."
        )
        roi = bb_snap(rio.coords.BoundingBox(*region_of_interest), align=10)

    # Adjust roi according to margin_in_meters
    roi = rio.coords.BoundingBox(
        left=max(roi.left, s2_ds.bounds.left + margin_in_meters),
        bottom=max(roi.bottom, s2_ds.bounds.bottom + margin_in_meters),
        right=min(roi.right, s2_ds.bounds.right - margin_in_meters),
        top=min(roi.top, s2_ds.bounds.top - margin_in_meters),
    )

    _logger.info(f"Will generate following roi : {roi}")
    chunks = generate_chunks(roi, tile_size_in_meters, margin_in_meters)
    _logger.info(f"Will process {len(chunks)} image chunks")

    # Output tiff profile
    geotransform = (roi[0], target_resolution, 0.0, roi[3], 0.0, -target_resolution)
    transform = Affine.from_gdal(*geotransform)

    profile = {
        "driver": "GTiff",
        "height": int((roi[3] - roi[1]) / target_resolution),
        "width": int((roi[2] - roi[0]) / target_resolution),
        "count": 3,
        "dtype": np.uint8,
        "crs": s2_ds.crs,
        "transform": transform,
        "nodata": 0,
        "tiled": True,
    }

    # Ensure that ouptut directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Derive output file name
    out_sr_file = os.path.join(
        output_dir,
        str(s2_ds.satellite.value)
        + "_"
        + f"{s2_ds.year}-{s2_ds.quartile}"
        + level
        + "T"
        + s2_ds.tile
        + f"_{model_parameters.name}_"
        + str(target_resolution).replace(".", "m").rstrip("0")
        + "_sisr.tif",
    )
    _logger.info(f"Super-resolved output image: {out_sr_file}")

    with rio.open(out_sr_file, "w", **profile) as rio_ds:
        for chunk in tqdm(
            chunks, total=len(chunks), desc="Super-resolution in progress ..."
        ):
            data_array = s2_ds.read_as_numpy(
                bounds=chunk.source_area,
                bands=bands,
                resolution=10,
                scale=1.0,
                no_data_value=-32768,
                algorithm=rio.enums.Resampling.cubic,
            )[0]

            print(chunk.source_area)

            data_array = data_array.astype(np.float32)

            output = ort_session.run(
                None, {"input": data_array[None, ...]}, run_options=ro
            )
            output = output[0][0, ...]

            output[np.isnan(output)] = -10000

            # Crop margin out
            if model_parameters.margin != 0:
                cropped_output = output[
                    :,
                    model_parameters.margin: -model_parameters.margin,
                    model_parameters.margin: -model_parameters.margin,
                ]
            else:
                cropped_output = output

            # Find location to write in ouptut image
            window = rio.windows.Window(
                int(np.ceil((chunk.target_area.left - roi.left) / target_resolution)),
                int(np.floor((roi.top - chunk.target_area.top) / target_resolution)),
                int(np.floor((chunk.target_area.right - chunk.target_area.left) / target_resolution)),
                int(np.ceil((chunk.target_area.top - chunk.target_area.bottom) / target_resolution)),
            )

            print(chunk.target_area.right)
            print(chunk.target_area.left)
            print((chunk.target_area.right - chunk.target_area.left) / target_resolution)

            # Color correct and contrast enhance
            corrected_output = toRGB(cropped_output[0:3])

            # Write ouptut image
            rio_ds.descriptions = tuple(['Blue',
                                         'Green',
                                         'Red'])
            rio_ds.write(corrected_output, window=window)


if __name__ == '__main__':
    run("model/s2v2x2_spatrad.yaml", "data/e46af4b23474a1f85bed9b783f418795.tif", output_dir="../results/")
