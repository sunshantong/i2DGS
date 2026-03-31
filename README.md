# i2DGS: Inverting 2D Gaussian Splatting for 6D Pose Estimation
This repository contains a PyTorch implementation for the paper: [i2DGS: Inverting 2D Gaussian Splatting for 6D Pose Estimation]

## Installation

Install environment:
```bash
conda env create --file environment.yml
conda activate i2dgs
```

Install custom CUDA extensions (requires CUDA toolkit and ninja):
```bash


# diff-surfel-rasterization
cd submodules/diff-surfel-rasterization
pip install -e . --no-build-isolation

# simple-knn
cd submodules/simple-knn
pip install -e . --no-build-isolation
```

## Data Preparation

### Tanks&Temples
We use the dataset format of NSVF:

[Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)

Note: The Ignatius object contains a malformed intrinsics.txt. You can find a correctly formatted version. Replace the original file to resolve this issue.

### Mip-NeRF 360°
Download part 1 of the dataset:

[Mip-NeRF 360°](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)

## 2D Gaussian Splatting Model Training

The training script is located in `train.py`. To train a single 2DGS model:

```bash
python train.py -s <path_to_dataset> -m <path_to_output_directory>
```

## Pose Estimation Module
The training and testing script for pose estimation is located in `pretrain_eval_attention.py`: 

```bash
python3 pretrain_eval_attention.py --exp_path <path_to_output_directory> --out_path results.json
```

## Quick Examples
Assuming you have downloaded [Mip-NeRF 360°](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip), the complete estimation pipeline is as follows:
```bash
# Training 2DGS model
python train.py -s <path to m360>/<garden> -m output/garden

# run the pose estimation model
python3 pretrain_eval_attention.py --exp_path ./output/garden --out_path results_garden.json
```

## Acknowledgements
This project builds upon the following foundational works:

- 6DGS for the 6D pose estimation framework
- 2DGS for the 2D Gaussian Splatting techniques

We thank the respective authors for their valuable contributions.

## Citation
```
@INPROCEEDINGS{Bortolon20246dgs,
  author = {Bortolon, Matteo and Tsesmelis, Theodore and James, Stuart and Poiesi, Fabio and Del Bue, Alessio},
  title = {6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model},
  booktitle = {ECCV},
  year = {2024}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
