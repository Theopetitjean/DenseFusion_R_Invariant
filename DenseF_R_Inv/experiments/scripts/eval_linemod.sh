#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/tpetitjean/PhD_Thesis/DenseFusion/tools/eval_linemod.py --dataset_root /home/tpetitjean/PhD_Thesis/DenseFusion/datasets/linemod/Linemod_preprocessed\
  --model /home/tpetitjean/PhD_Thesis/DenseFusion/trained_models/linemod/pose_model_9_0.01310166542980859.pth\
  --refine_model /home/tpetitjean/PhD_Thesis/DenseFusion/trained_models/linemod/pose_refine_model_493_0.006761023565178073.pth
