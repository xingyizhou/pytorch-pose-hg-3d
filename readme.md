# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

This repository is the PyTorch implementation for the network presented in:

> Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
> **Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach**
> ICCV 2017 ([arXiv:1704.02447](https://arxiv.org/abs/1704.02447))

Checkout the original [torch implementation](https://github.com/xingyizhou/pose-hg-3d).

Checkout the clean [2D hourglass network branch](https://github.com/xingyizhou/pytorch-pose-hg-3d/tree/2D).

Contact: [zhouxy2017@gmail.com](mailto:zhouxy2017@gmail.com)

## Requirements
- cudnn
- [PyTorch](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 

## Testing
- Download our pre-trained [model](https://drive.google.com/a/utexas.edu/file/d/1mUEybux3YZ2VhSjs-k4kBadbrT5qx09i/view?usp=sharing) and move it to `models`.
- Run `python demo.py -demo /path/to/image [-loadModel /path/to/image]`. 

We provide example images in `images/`. For testing your own image, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Training
- Prepare the training data:
  - Download our pre-processed Human3.6M dataset [here](https://drive.google.com/open?id=0BxjtxDYaOrYPRlJJeDhfUVAzM00).
  - Run `python GetH36M.py` in `src/tools/` to convert H36M annotations to hdf5 format.
  - Modify `src/ref.py` to setup the dataset path. 

- Stage1: Train the 2D hourglass component for 60 epochs
```
python main.py -expID Stage1
```

Our results of this stage is provided [here](https://drive.google.com/a/utexas.edu/file/d/18IKJyhoZr-oJ7--nGVFOcZi18v-pF5Gh/view?usp=sharing). 

- Stage2: Train without Geometry loss (drop LR at 25 epochs)
```
python main.py -expID Stage2 -ratio3D 1 -regWeigh 0.1 -loadModel ../exp/Stage1/model_60.pth -nEpochs 30 -dropLR 25
```

- Stage3: Train with Geometry loss

```
python main.py -expID Stage3 -ratio3D 1 -regWeigh 0.1 -varWeight 0.01 -loadModel ../exp/Stage2/model_30.pth -LR 2.5e-5 -nEpochs 10
```

## Citation

    @InProceedings{Zhou_2017_ICCV,
    author = {Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
    title = {Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }
