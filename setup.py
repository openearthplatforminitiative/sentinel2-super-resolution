from io import open
from setuptools import find_packages, setup

setup(
    name="sentinel2-super-resolution",
    version="0.1.0",
    description="AI upscaling of sentinel 2 imagery",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/openearthplatforminitiative/sentinel2-super-resolution",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=True,
    install_requires=open("requirements.txt").read().splitlines()
)
