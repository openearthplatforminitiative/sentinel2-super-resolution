# Sentinel2-SR

Simple repo for AI upscaling of sentinel 2 imagery. Original code is from 
[Sentinel2 Superresolution](https://github.com/Evoland-Land-Monitoring-Evolution/sentinel2_superresolution).
Modified to allow upscaling of Sentinel2 Quarterly Mosaics. 

## Files

The sentinel2sr package contains 4 files:
* **regulargrid**: File directly taken from the [sensorsio](https://github.com/CNES/sensorsio) repo. This allows us to
avoid the sensorsio dependency, which has some difficult transative dependencies. Used to read tiff file data.
* **run_l3_4band.py**: Modified version of the [Sentinel2 Superresolution](https://github.com/Evoland-Land-Monitoring-Evolution/sentinel2_superresolution) main run file. The modifications
allow us to use Quarterly Mosaics instead of standard sentinel images.
* **sentinel2_l3.py**: Modified file from [sensorsio](https://github.com/CNES/sensorsio) repo that allows us to load
Quarterly Mosaics into the sensorsio framework.
* **toRGB**: Postprocessing step that transforms 16-bit surface reflections to 8-bit RGB values and does some contrast 
enhancement to downtune some especially bright areas.


## Usage

## Installation
Install dependencies
```pip install -r requirements.txt```

If you have a GPU, also install ```onnxruntime-gpu```. Only GPUs supporting CUDA is supported.

## Running
The most basic usage of the package looks like this:

```python
from sentinel2sr import run

model = "s2v2x2_spatrad"
in_file = "path/to/input.tif"
out_dir = "path/to/output/folder/"
upscaled = run(model, in_file, output_dir=out_dir)
```

The run function supports the following arguments:

```
model_yaml - The model to use for super-resolution task. See Model section for more info.
input - The input tiff file to be upscaled. 4 bands are expected to be present in the input tiff, in the order [B02, B03, B04, B08]
output_dir - The folder in which the output file wil be stored.
region_of_interest_pixel(Optional) - Select only specific region of input image to be super-resolved, using pixel regions.  
region_of_interest(Optional) - Select only specific region of input image to be super-resolved, using UTM coordinates. 
loglevel(Optional) - The loglevel for python logs. Default INFO.
tilesize(Optional) - Tile size used in chunking, provided in pixels. Default 1000."
num_threads(Optional) - Number of threads used for model inference. Default 8.
```

## Models
The following models are available for superresolving Sentinel images: 

* carn_3x3x64g4sw_bootstrap
* s2v2x2_spatrad
* wsx2_spatrad
* wsx4_spatrad


### CARN
CARN (carn_3x3x64g4sw_bootstrap) is a form of neural network designed specifically for super resolution tasks. 
This makes the model the safest in terms of real-life accuracy, since it will not generate information that 
wasn’t present in the original photo. The downside is that the upsampled images may not look too different from 
the original photo. The model is trained on the [Sen2Venus](https://zenodo.org/records/14603764) dataset.
 

### ESRGAN
ESRGAN is a generative model which excels at superresolving images. While the model is generative, it differs from 
transformer models such as DALL-E in that it generally generates more consistent and safe images.

Three different ESRGAN models are available. ```s2v2x2_spatrad``` generates images at 
5m resolution. The model has been trained on the [Sen2Venus](https://zenodo.org/records/14603764) dataset, 
the same one as the CARN model.

```wsx2_spatrad``` and ```wsx4_spatrad``` have been trained on the [WorldStrat](https://worldstrat.github.io/) 
dataset. The two models differ in that one generates images with 5m resolution, and the other with 2.5m resolution. 
These two models are the most unsafe in terms of accuracy, and they struggle especially with urban areas. 
For rural areas however, the models perform alot better, though they might still generate information not present in 
the original images.  


## RGB color correction

The RGB color transformations in the *toRGB* file has some variables that can be tweaked to change the colors of the
final images:

* maxR: Max reflectance. Increasing this variable darkens the final images, but expands the range of bright areas.
* midR: Mid reflectance. Increasing this variable darkens the final images, but increases contrast.
* sat: Saturation factor. Increasing this variable increases the saturation of the final images.
* gamma: Gamma factor. Increasing this variable increases the brightness of the final images.

