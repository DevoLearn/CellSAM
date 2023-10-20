# CellSAM
This is the official repository for CellSAM: Segment Anything in Microscopy Images of C. elegans.

## Installation
1. Create a virtual environment `conda create -n cellsam python=3.10 -y` and activate it `conda activate cellsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/DevoLearn/CellSAM.git`
4. Enter the CellSAM folder `cd CellSAM` and run `pip install -e .`

## Get Started
Download the [model checkpoint](https://drive.google.com/drive/folders/1AF8AFw3dpppg_U79kGsXXl85sy4DNR3m?usp=share_link) and place it at e.g., `work_dir/CellSAM/cellsam_vit_b`

We provide three ways to quickly test the model on your microscopy images:

1. **Command line**

```bash
python CellSAM_Inference.py # segment the demo image
```

Segment other images with the following flags:
```bash
-i input_img
-o output path
--box bounding box of the segmentation target
```

2. **Jupyter-notebook**

We provide a step-by-step tutorial on [CoLab](https://colab.research.google.com/drive/1HXntUbgstm8UFgamW71PMAf7hgIT-J3V?usp=sharing)

You can also run it locally with `tutorial_quickstart.ipynb`.

3. **GUI**

Install `PyQt5` with [pip](https://pypi.org/project/PyQt5/): `pip install PyQt5` or [conda](https://anaconda.org/anaconda/pyqt): `conda install -c anaconda pyqt`

```bash
python gui.py
```



## Model Training

We have trained only mask decoder by freezing the image encode ,prompt encoder .

### Training on multiple GPUs (Recommend)

The model was trained on one v100 gpu .

```bash
sbatch train_multi_gpus.sh
```

When the training process is done, please convert the checkpoint to SAM's format for convenient inference.

```bash
python utils/ckpt_convert.py # Please set the corresponding checkpoint path first
```

### Training on one GPU

```bash
python train_one_gpu.py
```

### Data preprocessing

Download the demo [dataset](http://celltrackingchallenge.net/datasets/) and unzip it to `data/CellTrain/`.

This dataset contains microscopy images of C. elegans. The names of the cell label are available at [CellTrackingCHALLENGE](http://celltrackingchallenge.net/annotations/).



- Split dataset: 80% for training and 20% for testing
- Max-min normalization
- Resample image size to `1024x2014`
- Save the pre-processed images and labels as `png` files

One  can navigate through th folder for the `notebooks/explore_preprocess_cell_tracking_challeneg_dataset.ipynb` for converting the 3d tiff files to microscopy images to png images 



## Acknowledgements
- We highly appreciate all the Google summer of code organizers and dataset owners for providing the public dataset to the community.
- Thanks for the bracly alicea, mayukh deb and mainak deb through entire GSOC period
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this insightful [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/).
- we also thanks for the code MedSAM Code owner for there code on Medical images [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main)

