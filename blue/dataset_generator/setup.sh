#!/bin/bash
#SBATCH -p dgx2q # alternatively partitions dgx2q (16 qty V100/32GB) or hgx2q (8 qty A100/80GB) or a100q (2 nodes with 2 qty A100/40GB each)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03-23:00
#SBATCH -o output/slurm.dataset_generator.%j.%N.out # STDOUT
#SBATCH -e output/slurm.dataset_generator.%j.%N.err # STDERR
module purge
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load anaconda3/x86_64/2022.05

#Import FFHQ
git clone https://github.com/NVlabs/ffhq-dataset.git

#Import StyleGAN3
git clone https://github.com/NVlabs/stylegan3.git

#Install requirements
conda env create -f stylegan3/environment.yml