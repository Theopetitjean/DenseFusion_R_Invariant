# DenseFusion_R_Invariant

## News 
This code is originaly published as : "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion"([arXiv](https://arxiv.org/abs/1901.04780),
[Project](https://sites.google.com/view/densefusion), [Video](https://www.youtube.com/watch?v=SsE5-FuK5jo)) by Wang et al. at
[Stanford Vision and Learning Lab](http://svl.stanford.edu/) and [Stanford People, AI & Robots Group](http://pair.stanford.edu/).

The idea of this project is to modify the Loss Calculation in order to work with rotation invariant in order to reduce the problematic of the symetry
and also the learling discontinuities created by the multiple possibl soluytion embeded by the symetry of an object. This work is done in the fields
of my PhD thesis with the help of SPIE and the university of Burgundy. 

Be sure to check the original work of DenseFusion team to understand the idea of this work. 

## Overview of DenseFusion 

The model takes an RGB-D image as input and predicts the 6D pose of the each object in the frame. This network is implemented using [PyTorch](https://pytorch.org/)
and the rest of the framework is in Python. Since this project focuses on the 6D pose estimation process, we do not specifically limit the choice of the segmentation
models. You can choose your preferred semantic-segmentation/instance-segmentation methods according to your needs. In this repo, we provide our full implementation
code of the DenseFusion model, Iterative Refinement model and a vanilla SegNet semantic-segmentation model used in our real-robot grasping experiment. The ROS code
of the real robot grasping experiment is not included.

## Overview of Rotation Invariant

Based on the Dense Fusion model, the network will take as input the densefeatures provided by the wichbone of the denseF architecture. Then rather than optimizing
the distance between the model with the predicted [R|t] and the GT, we will try to use Rotation invariant and to do our optimization over the t angle. 

The main part of the work is the definition of a new network that output Only a prediction for t and c. Then the New loss is defined in `lib/LossTranslation/` and
the option inside the training parametter are alowing us to play with the original code or the Translation one. 

## Requirements

* Python 2.7/3.5/3.6 (If you want to use Python2.7 to run this repo, please rebuild the `lib/knn/` (with PyTorch 0.4.1).)
* [PyTorch 0.4.1](https://pytorch.org/) ([PyTroch 1.0 branch](<https://github.com/j96w/DenseFusion/tree/Pytorch-1.0>))
* PIL
* scipy
* numpy
* pyyaml
* logging
* matplotlib
* CUDA 7.5/8.0/9.0 (Required. CPU-only will lead to extreme slow training speed because of the loss calculation of the symmetry objects (pixel-wise nearest neighbour loss).)

## Datasets

This work is tested on two 6D object pose estimation datasets:

* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/): Training and Testing sets follow [PoseCNN](https://arxiv.org/abs/1711.00199). The training set includes 80 training videos 0000-0047 & 0060-0091 (choosen by 7 frame as a gap in our training) and synthetic data 000000-079999. The testing set includes 2949 keyframes from 10 testing videos 0048-0059.

* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

Download YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints (You can modify this script according to your needs.):
```	
./download.sh
```

