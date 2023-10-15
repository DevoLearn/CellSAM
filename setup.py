# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import find_packages, setup

setup(
    name="cellsam",
    version="0.0.0",
    author="sushmanth",
    python_requires=">=3.9",
    install_requires=["monai", "matplotlib", "scikit-image", "SimpleITK>=2.2.1", "nibabel", "tqdm", "scipy", "ipympl", "opencv-python", "jupyterlab", "ipywidgets"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
