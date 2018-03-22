# Pytorch Stacked Hourglass Network

This repository is the PyTorch re-implementation for the network presented in:

> Alejandro Newell, Kaiyu Yang, and Jia Deng,          
> **Stacked Hourglass Networks for Human Pose Estimation**,        
> [ECCV 2016](arXiv:1603.06937, 2016).

Checkout the augmented [3D human pose estimation branch](https://github.com/xingyizhou/pytorch-pose-hg-3d/tree/master).

Checkout the original [torch implementation](https://github.com/anewell/pose-hg-train). 

The code basically re-produced the results discribed in the original hourglass network paper. Most of the code is a direct translation from lua to python. Thanks to the original authors!

A pre-trained model (2 stacks with 2 residual modules each stack), trained for 180 epochs, with validation acc 88.38\%
is provided [here](https://drive.google.com/a/utexas.edu/file/d/1QgkJ_hRzhTcZyBkEEyz6TAZUhlS9LYy1/view?usp=sharing).

## Requirements
- cudnn
- [PyTorch](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 

## Training
- Modify `src/ref.py` to setup the MPII dataset path. 
- Run
```
python main.py -expID 2D
```

