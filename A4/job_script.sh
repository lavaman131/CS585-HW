#!/bin/bash -l

#$ -P dnn-motion                     # Specify the SCC project name you want to use
#$ -l h_rt=24:00:00                  # Specify the hard time limit for the job
#$ -l gpus=1                         # Specify the number of GPUs
#$ -l gpu_memory=48G                 # Specify the amount of GPU memory
#$ -N fcn_train                      # Give job a name
#$ -j y                              # Merge the error and output streams into a single file

module load miniconda/23.5.2 cuda blender ffmpeg gcc llvm cmake

conda activate fcn

python train.py