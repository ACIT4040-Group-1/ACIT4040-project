#!/bin/bash
#SBATCH -p dgx2q # alternatively partitions dgx2q (16 qty V100/32GB) or hgx2q (8 qty A100/80GB) or a100q (2 nodes with 2 qty A100/40GB each)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03-23:00
#SBATCH -o output/slurm.stylegan3.%j.%N.out # STDOUT
#SBATCH -e output/slurm.stylegan3.%j.%N.err # STDERR
module purge
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load anaconda3/x86_64/2022.05

source activate /home/$USER/.conda/envs/stylegan3

#srun python stylegan3/gen_images.py --outdir="dataset/fake_images" --seeds="0-70000" --network="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl"
